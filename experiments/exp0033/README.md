# exp0033: ゼロ閾値最適化版MoE（予測分布補正）

## 🎯 実験概要

**提案**: MoE予測にゼロ閾値最適化を適用し、予測分布を実データに近づける

**ベース**: exp0031（木モデル版MoE）から派生

**課題**: MoE予測の低値域（0.02〜0.08付近）が真のゼロと異なる分布を持ち、wRMSEを悪化させている可能性

---

## 🔬 改善手法

### ゼロ閾値最適化（予測分布の補正）

#### 問題
- **予測分布の乖離**: MoE予測が真のデータ分布と異なる
- **小さい値の過剰**: 0.02, 0.03などの微小な予測値が多数存在
- **ゼロ率の不一致**: 真のゼロ率と予測ゼロ率が一致しない

#### 解決策
```python
# 1. 真のゼロ率を確認
true_zero_rate = (y_train == 0).mean()

# 2. 閾値候補でwRMSEを最小化する閾値を探索
zero_threshold_candidates = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
best_zero_threshold = optimize_threshold(candidates)

# 3. 最適閾値以下を0に補正
moe_oof_preds[moe_oof_preds < best_zero_threshold] = 0.0
moe_test_preds[moe_test_preds < best_zero_threshold] = 0.0
```

#### 期待効果
1. **予測分布の適正化**: 実データに近い分布に補正
2. **wRMSEの改善**: 微小な予測誤差を0に補正することで改善
3. **解釈性の向上**: ゼロ予測の明確化

---

## 📊 期待される効果

### スコア改善目標
- **exp0031 MoE OOF**: 0.2271
- **exp0033目標**: 0.225（-0.002改善）

### 改善メカニズム
1. **分布補正**: 予測分布が実データに近づくことでwRMSE改善
2. **微小値除去**: 0.02〜0.08の微小予測を0に補正し、誤差を削減
3. **ゼロ率の一致**: 真のゼロ率に近づけることで分布の整合性向上

---

## 📝 実装の詳細

### 実装フロー
```python
# Step 5.5: ゼロ閾値最適化
# 1. 訓練データの真のゼロ率を確認
true_zero_rate = (y_train.values == 0).mean()

# 2. MoE予測の低値域分布を確認
for thresh in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]:
    count = (moe_oof_preds < thresh).sum()
    rate = count / len(moe_oof_preds)

# 3. ゼロ閾値の探索
for zero_thresh in zero_threshold_candidates:
    moe_oof_corrected = moe_oof_preds.copy()
    moe_oof_corrected[moe_oof_corrected < zero_thresh] = 0.0
    score = weighted_rmse(y_train.values, moe_oof_corrected)

    if score < best_zero_score:
        best_zero_score = score
        best_zero_threshold = zero_thresh

# 4. 最適閾値で予測を補正
moe_oof_preds[moe_oof_preds < best_zero_threshold] = 0.0
moe_test_preds[moe_test_preds < best_zero_threshold] = 0.0
```

---

## 🔍 検証ポイント

### 補正効果の確認
- [ ] 最適ゼロ閾値が妥当な範囲か（0.02〜0.08を想定）
- [ ] 補正後のwRMSEが改善しているか
- [ ] 補正後のゼロ率が真のゼロ率に近づいているか

### 予測分布の健全性
- [ ] 補正後の予測分布統計（平均、標準偏差、最小値、最大値）が妥当か
- [ ] テスト予測にも同じ閾値を適用できているか
- [ ] Fold別スコアのばらつきが削減されているか

---

## 📂 成果物

- `training_with_zero_threshold.ipynb`: ゼロ閾値最適化版MoE実装
- `logs/host_moe_zero_threshold_001_metrics.json`: メトリクス
- `artifacts/oof_predictions_moe_zero_threshold.csv`: OOF詳細
- `submissions/host_moe_zero_threshold_001_submission.csv`: 提出ファイル

---

## 🚀 実行方法

```bash
# Jupyter Notebook/Labで実行
jupyter lab experiments/exp0033/training_with_zero_threshold.ipynb

# 実行セル順序:
# 1. セル1〜71: exp0031と同じく、データ読み込み〜MoE合成まで実行
# 2. セル72: ゼロ閾値最適化セルを実行（新規追加）
# 3. セル73: メトリクス保存
# 4. セル74: 提出ファイル生成
```

---

## 💡 次ステップの候補

### さらなる改善案
1. **High専門家のOptuna最適化**: 正則化含むハイパーパラメータを個別最適化（exp0034候補）
2. **3専門家MoE**: Low/Mid/Highの階層的分離
3. **Isotonic校正との組み合わせ**: ゼロ閾値補正 + Isotonic校正の相乗効果検証
4. **High領域特化特徴量**: キーパス、スルーパス、決定機関連の特徴量追加

### 実験の進め方
- ゼロ閾値最適化の効果が確認できたら、High専門家の最適化に進む
- 両者の改善が確認できたら、3専門家MoEなど高度化を検討

---

## 📖 参考

- **分布補正の重要性**: exp0032で対数変換が悪化した教訓から、分布の適正化アプローチを変更
- **閾値最適化**: 0.02〜0.08の範囲で微小値を0に補正し、wRMSEを最小化
- **exp0031ベース**: 対数変換を使わず、MoE予測をそのまま閾値補正

---

## 🏷️ タグ

`MoE` `Zero-Threshold` `Distribution-Correction` `Post-Processing` `wRMSE-Optimization`

---

## 📁 ディレクトリ構成

```
exp0033/
├── .gitignore                                    # Git無視設定
├── README.md                                     # 実験概要（このファイル）
├── EXPERIMENT_METADATA.yaml                      # 実験メタデータ
├── training_with_zero_threshold.ipynb           # メインノートブック
├── logs/                                        # 実行ログとメトリクス
│   ├── .gitkeep
│   └── host_moe_zero_threshold_001_metrics.json  # 実行後に生成
├── artifacts/                                   # OOF予測などの中間成果物
│   ├── .gitkeep
│   └── oof_predictions_moe_zero_threshold.csv    # 実行後に生成
└── submissions/                                 # 提出ファイル
    ├── .gitkeep
    └── host_moe_zero_threshold_001_submission.csv  # 実行後に生成
```

### ファイル説明

- **README.md**: 実験の概要、手法、期待効果、実行方法
- **EXPERIMENT_METADATA.yaml**: 実験の構造化されたメタデータ（日付、ベースライン、目標など）
- **training_with_zero_threshold.ipynb**: 実験の実行ノートブック
- **logs/**: 実行結果のメトリクス（JSON形式）
- **artifacts/**: OOF予測の詳細データ
- **submissions/**: Kaggle提出用CSVファイル
