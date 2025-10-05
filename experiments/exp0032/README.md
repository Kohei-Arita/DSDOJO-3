# exp0032: High専門家強化版MoE（対数変換 + Optuna最適化）

## 🎯 実験概要

**提案**: High専門家の性能改善により、MoE全体の精度を向上させる

**ベース**: exp0031（木モデル版MoE）から派生

**課題**: exp0031ではHigh専門家のRMSE=0.392と性能が低く、改善の余地が大きい

---

## 🔬 改善手法

### 1. High専門家の対数変換学習
- **問題**: High領域（y≥0.1）は裾が長い分布で、外れ値の影響を受けやすい
- **解決策**: log1p(y)で学習し、予測時にexpm1で逆変換
- **期待効果**: 値域圧縮により学習が安定し、予測精度が向上

### 2. High専門家専用のOptuna最適化
- **問題**: Low/High専門家が同じハイパーパラメータを使用
- **解決策**: High領域のデータでハイパーパラメータを個別最適化
- **最適化対象**:
  - `num_leaves`: 木の複雑さ
  - `learning_rate`: 学習率
  - `min_child_samples`: ノードの最小サンプル数（過学習抑制）
  - `reg_alpha`: L1正則化
  - `reg_lambda`: L2正則化

### 3. High専門家の正則化強化
- **問題**: High領域はサンプル数が少ない（約30%）ため過学習しやすい
- **解決策**: 正則化パラメータを強化
  - `min_child_samples`: 30 → 50
  - `reg_alpha`: 0.0 → 0.1
  - `reg_lambda`: 0.0 → 1.0

### 4. ゼロ閾値最適化（予測分布の補正）
- **問題**: 予測分布が実際のデータ分布と乖離（真のゼロ率が再現されない）
- **解決策**: 閾値以下の予測を0に補正し、wRMSEを最小化する閾値を探索
- **効果**:
  - 予測分布が実データに近づく
  - 小さい値（0.02, 0.03など）が0.1付近に引き上げられる問題を解消
  - wRMSEの改善

---

## 📊 期待される効果

### スコア改善目標
- **High専門家単体**: RMSE 0.392 → 0.25（目標）
- **MoE全体**: OOF 0.2271 → 0.225（-0.002改善）

### 副次的効果
1. **温度パラメータの最適化余地**: High専門家の性能向上により、τの探索範囲が拡大
2. **ゲート分離の効果最大化**: 分離精度（AUC=0.921）を活かせる
3. **アンサンブル効果の向上**: Low/High専門家の品質バランスが改善

---

## 📝 実装の詳細

### High専門家の学習フロー
```python
# 1. 対数変換
y_high_log = np.log1p(y_train[high_mask])

# 2. Optuna最適化（High領域のみ）
high_params = optuna_optimize(
    X_train[high_mask],
    y_high_log,
    metric=weighted_rmse,
    n_trials=50
)

# 3. 正則化強化
high_params.update({
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
})

# 4. 学習
high_model = lgb.train(high_params, train_high, ...)

# 5. 予測（逆変換）
high_pred_log = high_model.predict(X_val)
high_pred = np.expm1(high_pred_log)
```

---

## 🔍 検証ポイント

- [ ] High専門家のOOFスコアがexp0031比で改善しているか
- [ ] 対数変換により予測分布が適切に変換されているか
- [ ] Optuna最適化が収束しているか（trial数は十分か）
- [ ] MoE全体のOOFスコアがexp0031比で改善しているか
- [ ] 温度パラメータτの最適値が変化しているか

---

## 📂 成果物

- `training_with_nnls.ipynb`: High専門家強化版MoE実装
- `logs/host_moe_tree_003_metrics.json`: メトリクス
- `artifacts/oof_predictions_moe_v2.csv`: OOF詳細
- `submissions/host_moe_tree_003_submission.csv`: 提出ファイル

---

## 🚀 実行方法

```bash
# Jupyter Notebook/Labで実行
jupyter lab experiments/exp0032/training_with_nnls.ipynb

# High専門家のOptuna最適化セルを実行（新規追加）
# その後、既存のMoEセルを実行
```

---

## 💡 次ステップの候補

### さらなる改善案
1. **3専門家MoE**: Low/Mid/Highの階層的分離
2. **CatBoost High専門家**: LightGBMとのアンサンブル
3. **High領域特化特徴量**: キーパス、スルーパス、決定機関連の特徴量追加
4. **Multi-level Isotonic校正**: 各専門家の予測を個別に校正

### 実験の進め方
- High専門家の改善が確認できたら、Low専門家も同様に最適化
- 両専門家の性能が向上したら、ゲート機構の高度化（3専門家化など）

---

## 📖 参考

- **対数変換の効果**: exp0010で学習xTが効果的だったように、値域圧縮は裾の長い分布に有効
- **Optuna最適化**: exp0027でStratifiedGKFoldでCV安定化した実績あり
- **正則化**: High領域はサンプル数が少ないため、過学習抑制が重要

---

## 🏷️ タグ

`MoE` `High-Expert` `Log-Transform` `Optuna` `Regularization` `Expert-Optimization`

---

## 📁 ディレクトリ構成

```
exp0032/
├── .gitignore                      # Git無視設定
├── README.md                       # 実験概要（このファイル）
├── EXPERIMENT_METADATA.yaml        # 実験メタデータ
├── IMPLEMENTATION_SUMMARY.md       # 実装サマリーと実行手順
├── training_with_nnls.ipynb       # メインノートブック
├── logs/                          # 実行ログとメトリクス
│   ├── .gitkeep
│   └── host_moe_high_opt_003_metrics.json  # 実行後に生成
├── artifacts/                     # OOF予測などの中間成果物
│   ├── .gitkeep
│   └── oof_predictions_moe_high_opt.csv    # 実行後に生成
└── submissions/                   # 提出ファイル
    ├── .gitkeep
    └── host_moe_high_opt_003_submission.csv  # 実行後に生成
```

### ファイル説明

- **README.md**: 実験の概要、手法、期待効果、実行方法
- **EXPERIMENT_METADATA.yaml**: 実験の構造化されたメタデータ（日付、ベースライン、目標など）
- **IMPLEMENTATION_SUMMARY.md**: 実装の詳細、トラブルシューティング、次ステップ
- **training_with_nnls.ipynb**: 実験の実行ノートブック
- **logs/**: 実行結果のメトリクス（JSON形式）
- **artifacts/**: OOF予測の詳細データ
- **submissions/**: Kaggle提出用CSVファイル
