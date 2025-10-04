# Experiment exp0021

## ディレクトリ構成
- `artifacts/`: モデルや中間生成物の保管場所
- `logs/`: 学習・評価ログ
- `submissions/`: 生成した提出ファイル
- `training_with_catboost.ipynb`: Soft MoE 実装を含む学習ノートブック

## 実験ログ
| Run ID | Timestamp (UTC) | CV Weighted RMSE (mean ± std) | OOF Weighted RMSE | Gating AUC | Gating AP | Fold RMSEs | 備考 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `host_soft_moe` | 2025-10-03T23:47:43+00:00 | 0.2476 ± 0.0071 | 0.2477 | 0.8984 | 0.8118 | [0.2372, 0.2588, 0.2447, 0.2463, 0.2509] | `logs/host_soft_moe_metrics.json` |
 2025-10-03T23:47:43+00:00 | 0.2476 ± 0.0071 | 0.2477 | 0.8984 | 0.8118 | `logs/host_soft_moe_metrics.json` |

## メモ
- 文字列関連のtypoが出やすかったため、再利用時はカラム名・キー名の綴りを必ずダブルチェックすること。
