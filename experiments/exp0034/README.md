# exp0034: High専門家Optuna最適化版MoE

## 🎯 実験概要

**提案**: High専門家のハイパーパラメータをOptuna最適化し、MoE全体の精度を向上させる

**ベース**: exp0033（ゼロ閾値最適化版MoE）から派生

**課題**: exp0031でHigh専門家のRMSE=0.392と性能が低く、Low専門家と同じパラメータを使用していた

---

## 🔬 改善手法

### High専門家専用のOptuna最適化

#### 問題
- **パラメータの未最適化**: Low/High専門家が同じハイパーパラメータを使用
- **High領域の特性**: サンプル数が少なく（約30%）、過学習しやすい
- **性能の低さ**: High専門家のRMSE=0.392で改善余地が大きい

#### 解決策
```python
# 1. High領域のデータで最適化（Fold1のみ）
X_high_train = X_fold1_train[y_fold1_train >= 0.1]
y_high_train = y_fold1_train[y_fold1_train >= 0.1]

# 2. Optunaで最適化
def objective_high(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 10, 60),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
    }
    # ... 学習・評価
    return weighted_rmse(y_high_val, preds)

study.optimize(objective_high, n_trials=50)

# 3. 最適パラメータでHigh専門家を学習
high_expert_params.update(study.best_params)
```

#### 最適化対象パラメータ
- `num_leaves`: 木の複雑さ（10〜60）
- `learning_rate`: 学習率（0.01〜0.1、対数スケール）
- `min_child_samples`: ノードの最小サンプル数（20〜100、過学習抑制）
- `reg_alpha`: L1正則化（0.0〜1.0）
- `reg_lambda`: L2正則化（0.0〜2.0）
- `feature_fraction`: 特徴量サンプリング（0.7〜1.0）
- `bagging_fraction`: データサンプリング（0.7〜1.0）

#### 期待効果
1. **High専門家の性能向上**: RMSE 0.392 → 0.25（目標）
2. **MoE全体の改善**: OOF 0.2271 → 0.22以下
3. **過学習の抑制**: High領域に適した正則化パラメータ

---

## 📊 期待される効果

### スコア改善目標
- **exp0031 High専門家**: wRMSE 0.392
- **exp0034 High専門家目標**: 0.25（-0.142改善）
- **exp0031 MoE OOF**: 0.2271
- **exp0034 MoE OOF目標**: 0.22（-0.007改善）

### 改善メカニズム
1. **High領域専用最適化**: High領域のデータ特性に合わせたパラメータ
2. **正則化の適正化**: サンプル数が少ないHigh領域に最適な正則化
3. **ゼロ閾値との相乗効果**: exp0033のゼロ閾値最適化 + High専門家改善

---

## 📝 実装の詳細

### 実装フロー

#### Step 4.5（新規）: High専門家のOptuna最適化
```python
# Fold1でHigh領域の最適化
X_high_train = X_fold1_train[y_fold1_train >= 0.1]
y_high_train = y_fold1_train[y_fold1_train >= 0.1]

# Optuna最適化（50 trials）
study = optuna.create_study(direction='minimize')
study.optimize(objective_high, n_trials=50)

# 最適パラメータ取得
best_high_params = study.best_params
high_expert_params.update(best_high_params)
```

#### Step 4（更新）: High専門家の学習
```python
# high_expert_paramsを使用（Optuna最適化済み）
high_model = lgb.train(high_expert_params, train_high, ...)
```

#### Step 5.5: ゼロ閾値最適化（exp0033から継承）
```python
# MoE予測にゼロ閾値を適用
moe_oof_preds[moe_oof_preds < best_zero_threshold] = 0.0
moe_test_preds[moe_test_preds < best_zero_threshold] = 0.0
```

---

## 🔍 検証ポイント

### High専門家の改善確認
- [ ] Optuna最適化が収束しているか（50 trialsで十分か）
- [ ] 最適パラメータが妥当な範囲か
- [ ] High専門家のOOF wRMSEがexp0031比で改善しているか

### MoE全体の改善確認
- [ ] MoE OOF wRMSEがexp0031/exp0033比で改善しているか
- [ ] ゲート分離精度（AUC/AP）が維持されているか
- [ ] ゼロ閾値最適化との相乗効果があるか

### 予測の健全性
- [ ] OOF予測の分布が訓練データと類似しているか
- [ ] テスト予測の統計量が妥当か
- [ ] Fold別スコアのばらつきが削減されているか

---

## 📂 成果物

- `training_with_high_optuna.ipynb`: High専門家Optuna最適化版MoE実装
- `logs/host_moe_high_optuna_001_metrics.json`: メトリクス
- `artifacts/oof_predictions_moe_high_optuna.csv`: OOF詳細
- `submissions/host_moe_high_optuna_001_submission.csv`: 提出ファイル

---

## 🚀 実行方法

```bash
# Jupyter Notebook/Labで実行
jupyter lab experiments/exp0034/training_with_high_optuna.ipynb

# 実行セル順序:
# 1. セル1〜68: exp0033と同じく、データ読み込み〜Low専門家まで実行
# 2. セル69: High専門家のOptuna最適化（新規追加、50 trials）
# 3. セル70: High専門家の学習（最適パラメータ使用）
# 4. セル71: ゲート分類器の学習
# 5. セル72: MoE予測の合成
# 6. セル73: ゼロ閾値最適化（exp0033から継承）
# 7. セル74: メトリクス保存
# 8. セル75: 提出ファイル生成
```

### 実行時間の目安
- High専門家Optuna最適化: 約15〜20分（50 trials、GPU使用時）
- 全体実行時間: 約30〜40分

---

## 💡 次ステップの候補

### さらなる改善案
1. **Low専門家のOptuna最適化**: High専門家と同様にLow専門家も最適化（exp0035候補）
2. **3専門家MoE**: Low/Mid/Highの階層的分離
3. **温度パラメータτの再最適化**: 専門家性能向上後のτ探索
4. **CatBoost High専門家**: LightGBMとのアンサンブル
5. **High領域特化特徴量**: キーパス、スルーパス、決定機関連の特徴量追加

### 実験の進め方
- High専門家の改善が確認できたら、Low専門家も最適化
- 両専門家の性能が向上したら、3専門家MoEなど高度化を検討

---

## 📖 参考

- **Optuna最適化**: exp0027でStratifiedGKFoldでCV安定化した実績あり
- **High領域の特性**: サンプル数が少なく（約30%）、過学習しやすい
- **正則化の重要性**: min_child_samples、reg_alpha、reg_lambdaが過学習抑制に効果的
- **exp0032の教訓**: 対数変換は悪化したため、パラメータ最適化に注力

---

## 🏷️ タグ

`MoE` `High-Expert` `Optuna` `Hyperparameter-Optimization` `Regularization` `Zero-Threshold`

---

## 📁 ディレクトリ構成

```
exp0034/
├── .gitignore                                   # Git無視設定
├── README.md                                    # 実験概要（このファイル）
├── EXPERIMENT_METADATA.yaml                     # 実験メタデータ
├── training_with_high_optuna.ipynb             # メインノートブック
├── logs/                                       # 実行ログとメトリクス
│   ├── .gitkeep
│   └── host_moe_high_optuna_001_metrics.json    # 実行後に生成
├── artifacts/                                  # OOF予測などの中間成果物
│   ├── .gitkeep
│   └── oof_predictions_moe_high_optuna.csv      # 実行後に生成
└── submissions/                                # 提出ファイル
    ├── .gitkeep
    └── host_moe_high_optuna_001_submission.csv  # 実行後に生成
```

### ファイル説明

- **README.md**: 実験の概要、手法、期待効果、実行方法
- **EXPERIMENT_METADATA.yaml**: 実験の構造化されたメタデータ（日付、ベースライン、目標など）
- **training_with_high_optuna.ipynb**: 実験の実行ノートブック
- **logs/**: 実行結果のメトリクス（JSON形式）
- **artifacts/**: OOF予測の詳細データ
- **submissions/**: Kaggle提出用CSVファイル
