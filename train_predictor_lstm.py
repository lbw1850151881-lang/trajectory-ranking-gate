import os
import glob
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

FEATURE_INDEXES = {
    "x": 0,
    "y": 1,
    "vx": 2,
    "vy": 3,
    "yaw": 4,
    "length": 5,
    "width": 6,
}


class NuPlanLSTMDataset(Dataset):
    def __init__(self, data_root, input_features=("x", "y", "vx", "vy"), augment_noise=0.0):
        self.files = sorted(glob.glob(os.path.join(data_root, "*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_root}")

        self.feature_indices = [FEATURE_INDEXES[key] for key in input_features]
        self.augment_noise = augment_noise

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as data:
            past = data["ego_agent_past"]      # (21, 7)
            future = data["ego_agent_future"]  # (80, 7)

        past_feat = past[:, self.feature_indices].astype(np.float32)
        if self.augment_noise > 0:
            noise = np.random.normal(0.0, self.augment_noise, size=past_feat.shape).astype(np.float32)
            past_feat = past_feat + noise

        future_xy = future[:, :2].astype(np.float32)

        return torch.from_numpy(past_feat), torch.from_numpy(future_xy)


class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        input_dim=4,
        hidden_dim=128,
        num_layers=2,
        future_steps=80,
        dropout=0.2,
        dt=0.1,
        predict_absolute=False,
    ):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.future_steps = future_steps
        self.dt = dt
        self.predict_absolute = predict_absolute
        self.supports_teacher_forcing = True

    def forward(self, past_seq, future_targets=None, teacher_forcing_ratio=0.0):
        if teacher_forcing_ratio > 0 and future_targets is None:
            raise ValueError("future_targets must be provided when teacher_forcing_ratio > 0")

        batch_size = past_seq.size(0)
        if past_seq.size(-1) < 4:
            raise ValueError("Seq2SeqLSTM expects at least four features ordered as (x, y, vx, vy, ...).")

        _, (hidden, cell) = self.encoder(past_seq)

        outputs = []
        prev_state = past_seq[:, -1, :]
        prev_pos = prev_state[:, :2]

        for step in range(self.future_steps):
            prev_pos_before = prev_pos
            decoder_input = prev_state.unsqueeze(1)
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            delta = self.fc(out.squeeze(1))

            if self.predict_absolute:
                pred_pos = delta
                pred_delta = pred_pos - prev_pos_before
            else:
                pred_pos = prev_pos_before + delta
                pred_delta = delta

            outputs.append(pred_pos)

            pred_vel = pred_delta / self.dt
            next_state = torch.zeros_like(prev_state)
            next_state[:, :2] = pred_pos
            next_state[:, 2:4] = pred_vel
            if next_state.shape[1] > 4:
                next_state[:, 4:] = prev_state[:, 4:]

            if teacher_forcing_ratio > 0 and future_targets is not None:
                teacher_mask = torch.rand(batch_size, device=past_seq.device) < teacher_forcing_ratio
                if teacher_mask.any():
                    teacher_pos = future_targets[:, step, :2]
                    teacher_delta = teacher_pos - prev_pos_before
                    teacher_vel = teacher_delta / self.dt
                    next_state = next_state.clone()
                    next_state[teacher_mask, :2] = teacher_pos[teacher_mask]
                    next_state[teacher_mask, 2:4] = teacher_vel[teacher_mask]

                    prev_pos = pred_pos.clone()
                    prev_pos[teacher_mask] = teacher_pos[teacher_mask]
                else:
                    prev_pos = pred_pos
            else:
                prev_pos = pred_pos

            prev_state = next_state

        predictions = torch.stack(outputs, dim=1)
        return predictions


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerTrajectoryPredictor(nn.Module):
    def __init__(
        self,
        input_dim=4,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        future_steps=80,
        dropout=0.1,
        predict_absolute=False,
        dt=0.1,
    ):
        super().__init__()
        self.supports_teacher_forcing = False
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=512)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, future_steps * 2),
        )
        self.future_steps = future_steps
        self.predict_absolute = predict_absolute
        self.dt = dt

    def forward(self, past_seq, future_targets=None, teacher_forcing_ratio=0.0):
        if teacher_forcing_ratio not in (0, 0.0):
            raise ValueError("TransformerTrajectoryPredictor does not support teacher forcing.")

        x = self.input_proj(past_seq)
        x = self.pos_encoding(x)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        delta = self.fc(pooled).view(-1, self.future_steps, 2)

        if self.predict_absolute:
            positions = delta
        else:
            cumulative = torch.cumsum(delta, dim=1)
            last_pos = past_seq[:, -1, :2]
            positions = cumulative + last_pos.unsqueeze(1)

        return positions


def evaluate(model, loader, device):
    model.eval()
    total_ade, total_fde, count = 0.0, 0.0, 0
    with torch.no_grad():
        for past, future in loader:
            past = past.to(device)
            future = future.to(device)
            pred = model(past)
            dist = torch.norm(pred - future, dim=2)
            ade = dist.mean(dim=1)
            fde = dist[:, -1]
            total_ade += ade.sum().item()
            total_fde += fde.sum().item()
            count += past.size(0)
    avg_ade = total_ade / max(count, 1)
    avg_fde = total_fde / max(count, 1)
    return avg_ade, avg_fde


def physics_regularizer(positions, dt, accel_limit=None, jerk_weight=0.0):
    # positions: [B, T, 2]
    vel = torch.diff(positions, dim=1) / dt  # [B, T-1, 2]
    acc = torch.diff(vel, dim=1) / dt  # [B, T-2, 2]
    penalty = acc.pow(2).mean()

    if accel_limit is not None:
        exceed = torch.relu(acc.abs() - accel_limit)
        penalty = penalty + exceed.pow(2).mean()

    if jerk_weight and jerk_weight > 0:
        jerk = torch.diff(acc, dim=1) / dt
        penalty = penalty + jerk_weight * jerk.pow(2).mean()

    return penalty


def make_scheduler(optimizer, scheduler_name, epochs):
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    return None


def train_model(train_loader, val_loader, model, device, args):
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.lr_scheduler, args.epochs)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    best_ade = float("inf")
    best_metrics = (float("inf"), float("inf"))
    best_model_path = args.output_model

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        base_tf_ratio = max(args.teacher_forcing_end, args.teacher_forcing_start * (args.teacher_forcing_decay ** epoch))
        use_tf = getattr(model, "supports_teacher_forcing", False)
        tf_ratio = base_tf_ratio if use_tf else 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for past, future in progress:
            past = past.to(device)
            future = future.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled):
                if use_tf:
                    preds = model(
                        past,
                        future_targets=future if tf_ratio > 0 else None,
                        teacher_forcing_ratio=tf_ratio if tf_ratio > 0 else 0.0,
                    )
                else:
                    preds = model(past)
                loss = criterion(preds, future)
                if args.fde_weight > 0:
                    loss = loss + args.fde_weight * criterion(preds[:, -1], future[:, -1])
                if args.physics_weight > 0:
                    loss = loss + args.physics_weight * physics_regularizer(
                        preds, args.dt, accel_limit=args.physics_acc_limit, jerk_weight=args.physics_jerk_weight
                    )

            scaler.scale(loss).backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{running_loss / (progress.n or 1):.4f}", tf_ratio=f"{tf_ratio:.3f}")

        val_ade, val_fde = evaluate(model, val_loader, device)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_ade)
        elif scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {running_loss / max(len(train_loader), 1):.4f} | "
            f"Val ADE: {val_ade:.3f} | Val FDE: {val_fde:.3f} | "
            f"LR: {current_lr:.2e} | TF Ratio: {tf_ratio:.3f}"
        )

        if val_ade < best_ade:
            best_ade = val_ade
            best_metrics = (val_ade, val_fde)
            torch.save(model.state_dict(), best_model_path)
            print(f"[Best Updated] Saved model with Val ADE {val_ade:.3f} to {best_model_path}")

    print(
        f"Training complete. Best validation metrics -> ADE: {best_metrics[0]:.3f}, "
        f"FDE: {best_metrics[1]:.3f}. Model path: {best_model_path}"
    )


def build_dataloaders(args):
    train_dataset = NuPlanLSTMDataset(
        args.train_set,
        input_features=tuple(args.input_features.split(",")),
        augment_noise=args.noise_std,
    )
    val_dataset = NuPlanLSTMDataset(
        args.valid_set,
        input_features=tuple(args.input_features.split(",")),
        augment_noise=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", type=str, required=True, help="Path to train directory")
    parser.add_argument("--valid_set", type=str, required=True, help="Path to valid directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--fde_weight", type=float, default=1.0)
    parser.add_argument("--teacher_forcing_start", type=float, default=0.6)
    parser.add_argument("--teacher_forcing_end", type=float, default=0.05)
    parser.add_argument("--teacher_forcing_decay", type=float, default=0.9)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument(
        "--input_features",
        type=str,
        default="x,y,vx,vy",
        help="Comma separated feature keys to use from ego_agent_past.",
    )
    parser.add_argument("--future_steps", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--predict_absolute", action="store_true", help="Predict absolute positions instead of deltas.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_model", type=str, default="best_seq2seq_lstm.pth")
    parser.add_argument("--physics_weight", type=float, default=0.0, help="Weight for physics regularization loss.")
    parser.add_argument("--physics_acc_limit", type=float, default=None, help="Optional acceleration limit (m/s^2).")
    parser.add_argument("--physics_jerk_weight", type=float, default=0.0, help="Additional jerk penalty weight.")
    parser.add_argument("--resume_from", type=str, default=None, help="Optional checkpoint to warm start the model.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="lstm",
        choices=["lstm", "transformer"],
        help="Backbone architecture for trajectory forecasting.",
    )
    parser.add_argument("--transformer_d_model", type=int, default=256)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=4)
    parser.add_argument("--transformer_ffn_dim", type=int, default=512)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()

    train_loader, val_loader = build_dataloaders(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp and device.type != "cuda":
        print("AMP requested but CUDA not available; running in FP32 instead.")
    input_dim = len(args.input_features.split(","))
    model_type = args.model_type.lower()
    if model_type == "lstm":
        model = Seq2SeqLSTM(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            future_steps=args.future_steps,
            dropout=args.dropout,
            dt=args.dt,
            predict_absolute=args.predict_absolute,
        ).to(device)
    elif model_type == "transformer":
        model = TransformerTrajectoryPredictor(
            input_dim=input_dim,
            d_model=args.transformer_d_model,
            nhead=args.transformer_heads,
            num_layers=args.transformer_layers,
            dim_feedforward=args.transformer_ffn_dim,
            future_steps=args.future_steps,
            dropout=args.transformer_dropout,
            predict_absolute=args.predict_absolute,
            dt=args.dt,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    if args.resume_from:
        if not os.path.isfile(args.resume_from):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if missing or unexpected:
            print(f"Warning: resume checkpoint missing keys: {missing}, unexpected: {unexpected}")
        else:
            print(f"Loaded weights from {args.resume_from}")

    train_model(train_loader, val_loader, model, device, args)


if __name__ == "__main__":
    main()
