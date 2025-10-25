#!/usr/bin/env python3
"""
Train a tri-expert gate that chooses among LSTM, GameFormer, and SceneConditioned.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tri_expert_model import TriExpertModel, EXPERT_LABELS
from tri_feature_utils import BASE_FEATURE_KEYS, DERIVED_FEATURE_KEYS, compute_additional_features


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray
    feature_keys: List[str]


class TriExpertDataset(Dataset):
    def __init__(self, samples: List[Dict[str, float]], feature_keys: List[str] = None):
        if feature_keys is None:
            self.feature_keys = BASE_FEATURE_KEYS + DERIVED_FEATURE_KEYS
        else:
            self.feature_keys = feature_keys

        self.features = []
        self.labels = []
        self.sample_indices = []

        for sample in samples:
            feature_map = {k: float(sample.get(k, 0.0)) for k in BASE_FEATURE_KEYS}
            feature_map.update(compute_additional_features(sample))
            feature_vec = [feature_map.get(k, 0.0) for k in self.feature_keys]

            lstm_fde = float(sample.get("lstm_fde", 1e6))
            gmf_fde = float(sample.get("gameformer_fde", 1e6))
            sc_fde = float(sample.get("scene_conditioned_fde", 1e6))

            fdes = [lstm_fde, gmf_fde, sc_fde]
            label = int(np.argmin(fdes))

            self.features.append(feature_vec)
            self.labels.append(label)
            self.sample_indices.append(int(sample.get("sample_idx", -1)))

        self.features = np.asarray(self.features, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)

        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0) + 1e-8
        self.features = (self.features - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        feat = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feat, label

    def get_normalization(self) -> NormalizationStats:
        return NormalizationStats(self.mean, self.std, self.feature_keys)

    def class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.labels, minlength=3)
        weights = counts.sum() / np.maximum(counts, 1)
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)


def load_samples(path: Path) -> List[Dict[str, float]]:
    data = json.loads(path.read_text())
    return data


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    preds = []
    labels = []

    with torch.no_grad():
        for features, label in loader:
            features = features.to(device)
            label = label.to(device)
            logits = model(features)
            loss = criterion(logits, label)
            total_loss += loss.item() * features.size(0)
            prediction = logits.argmax(dim=1)
            total_correct += (prediction == label).sum().item()
            total += features.size(0)
            preds.extend(prediction.cpu().numpy())
            labels.extend(label.cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    accuracy = total_correct / max(total, 1)
    return avg_loss, accuracy, confusion_matrix(labels, preds, labels=[0, 1, 2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tri-expert classifier.")
    parser.add_argument(
        "--data",
        type=str,
        default="eval_out/fusion_stats_with_meta_features.json",
        help="Path to stats JSON with per-sample metrics.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of training epochs.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="128,64,32",
        help="Comma separated hidden layer sizes.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_out/tri_gate.pth",
        help="Output checkpoint path.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = load_samples(Path(args.data))
    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)
    train_samples, val_samples = train_test_split(train_samples, test_size=0.15, random_state=42)

    train_dataset = TriExpertDataset(train_samples)
    normalization = train_dataset.get_normalization()

    val_dataset = TriExpertDataset(val_samples, feature_keys=normalization.feature_keys)
    val_dataset.mean = normalization.mean
    val_dataset.std = normalization.std
    val_dataset.features = (val_dataset.features - normalization.mean) / normalization.std

    test_dataset = TriExpertDataset(test_samples, feature_keys=normalization.feature_keys)
    test_dataset.mean = normalization.mean
    test_dataset.std = normalization.std
    test_dataset.features = (test_dataset.features - normalization.mean) / normalization.std

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]
    model = TriExpertModel(
        input_dim=train_dataset.features.shape[1],
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weights().to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0

        for features, label in train_loader:
            features = features.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            total_correct += (logits.argmax(dim=1) == label).sum().item()
            total += features.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = total_correct / max(total, 1)

        val_loss, val_acc, _ = evaluate_model(model, val_loader, device)
        scheduler.step(val_acc)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}"
            )

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, cm = evaluate_model(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(
        classification_report(
            test_dataset.labels,
            model(torch.tensor(test_dataset.features, dtype=torch.float32, device=device)).argmax(dim=1).cpu().numpy(),
            target_names=EXPERT_LABELS,
            digits=4,
        )
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": train_dataset.features.shape[1],
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "normalization": {
            "mean": normalization.mean,
            "std": normalization.std,
            "feature_keys": normalization.feature_keys,
        },
        "class_weights": train_dataset.class_weights().tolist(),
        "history": history,
        "val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "confusion_matrix": cm.tolist(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"\nCheckpoint saved to {output_path}")


if __name__ == "__main__":
    main()
