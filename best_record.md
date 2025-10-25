# Best Model Records

The table below summarises the strongest metrics found in `eval_out`. Records were picked by minimising ADE when available (otherwise by FDE). Numeric columns report lower-is-better errors in metres.

| Model | Best ADE (lower better) | Best FDE (lower better) | Source | Variant / Context | Notes |
|---|---|---|---|---|---|
| LLM Gate | 1.2547 | 2.5673 | `eval_out/llm_gate_results.json` | llm_gate_results | Oracle capture 57.8%; LLM trigger 34.0%; LLM triggers 438/1287 |
| Oracle | 1.5676 | 3.6026 | `eval_out/gate_comparison_results.json` | Oracle (Upper Bound) (gate_comparison) | LSTM usage 27.5%; GMF usage 72.5% |
| GameFormer | 1.7154 | 4.1725 | `eval_out/eval_models_new.log` | eval_models_new.log | |
| Meta-Gate | 1.8720 | 4.3245 | `eval_out/final_gate_comparison.json` | Meta-Gate (Geo+Meta) (final_comparison) | LSTM usage 17.8%; GMF usage 82.2% |
| Two-Stage Gate | 1.8832 | 4.2974 | `eval_out/two_stage_gate_results.json` | two_stage_gate_results | Oracle capture 18.5% |
| Ranking Gate | 1.8841 | 4.3129 | `eval_out/ranking_gate_results.json` | ranking_gate_results | Oracle capture 16.7%; LSTM usage 8.7%; GMF usage 91.3%; Accuracy 0.779 |
| MLP Gate | 1.9081 | 4.4534 | `eval_out/gate_comparison_results.json` | MLP Gate (gate_comparison) | LSTM usage 4.9%; GMF usage 95.1% |
| Regression Gate | 1.9146 | 4.4103 | `eval_out/all_gates_results.json` | Regression Gate (th=0.0) (all_gates) | LSTM usage 6.8%; GMF usage 93.2% |
| Threshold Gate | 1.9235 | 4.4406 | `eval_out/threshold_sweep_results.json` | threshold_0.10 (threshold_sweep) | threshold=0.1; LSTM usage 0.2% |
| Risk-Based Gate | 1.9578 | 4.4891 | `eval_out/risk_based_gate_results.json` | risk_based_gate_results | LSTM usage 21.7%; GMF usage 78.3% |
| Calibrated Gate (Classification) | 2.0984 | 4.8698 | `eval_out/calibrated_gate_classification_results.json` | calibrated_gate_classification_results | Final LSTM 5/300 |
| Calibrated Gate (Test) | 2.4422 | 5.1105 | `eval_out/calibrated_gate_test_results.json` | calibrated_gate_test_results | Final LSTM 0/10 |
| LSTM | 2.6621 | 7.7156 | `eval_out/gate_comparison_results.json` | lstm (gate_comparison) | |
| Calibrated Gate (Regression) | 3.5536 | 9.0345 | `eval_out/calibrated_gate_results.json` | calibrated_gate_results | Final LSTM 87/100 |
| Quantile Threshold Gate | N/A | 4.3576 | `eval_out/quantile_threshold_results.json` | quantile_threshold_results | LSTM usage 10.0%; GMF usage 90.0%; quantile=0.75 |

## Notes

- Parsed files: `gate_comparison_results.json`, `final_gate_comparison.json`, `all_gates_results.json`, `fusion_all_results.json`, `threshold_sweep_results.json`, `ranking_gate_results.json`, `risk_based_gate_results.json`, `two_stage_gate_results.json`, `quantile_threshold_results.json`, `oracle_upperbound_results.json`, `llm_gate_results.json`, `gate_regression_results.json`, `gate_mlp_results.json`, `calibrated_gate_results.json`, `calibrated_gate_classification_results.json`, `calibrated_gate_test_results.json`, and `eval_models_new.log`.
- When multiple variants existed (for example, different thresholds), the variant delivering the lowest ADE was kept. Where ADE was absent, FDE was used instead (e.g., quantile threshold gate).
- Ratios in the notes columns are expressed as percentages of the evaluated split.
