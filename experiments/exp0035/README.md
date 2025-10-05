# EXP0035: Low専門家Optuna最適化版MoE

## 🎯 実験目的

**Primary**: Low専門家のハイパーパラメータをOptuna最適化し、MoE全体の精度を向上させる

**Secondary**:
- Low領域専用のパラメータで過学習を抑制
- High専門家最適化（exp0034）との相乗効果を実現

## 📊 ベースライン比較

| 指標 | exp0031 | exp0034 | exp0035目標 |
|------|---------|---------|-------------|
| Low Expert wRMSE | TBD | - | TBD |
| High Expert wRMSE | 0.392 | 0.25 | 0.25 |
| MoE OOF wRMSE | 0.2271 | 0.22 | **0.21** |

## 🔬 手法

### Optuna最適化パラメータ

```python
optimization_space = {
    'num_leaves': [10, 60],
    'learning_rate': [0.01, 0.1],  # log scale
    'min_child_samples': [20, 100],
    'reg_alpha': [0.0, 1.0],
    'reg_lambda': [0.0, 2.0],
    'feature_fraction': [0.7, 1.0],
    'bagging_fraction': [0.7, 1.0]
}
```

### 実装ステップ

1. **Step 4.5**: Low専門家のOptuna最適化
   - Fold1のLow領域でハイパーパラメータ探索
   - 50 trials でTPESampler使用
   - 最適パラメータを low_expert_params に反映

2. **Step 4**: Low専門家の学習
   - Optuna最適化済みパラメータを使用
   - 全Foldで学習

3. **Step 5.5**: ゼロ閾値最適化（exp0033から継承）

4. **Step 6**: メトリクス保存

## 📁 成果物

```
exp0035/
├── artifacts/
│   └── oof_predictions_moe_low_optuna.csv
├── catboost_info/
├── logs/
│   └── host_moe_low_optuna_001_metrics.json
├── submissions/
│   └── host_moe_low_optuna_001_submission.csv
├── training_with_low_optuna.ipynb
└── training_with_low_optuna.py
```

## ✅ 検証ポイント

- [ ] Optuna最適化が収束しているか（50 trialsで十分か）
- [ ] 最適パラメータが妥当な範囲か
- [ ] Low専門家のOOF wRMSEがexp0031比で改善しているか
- [ ] MoE OOF wRMSEがexp0034比で改善しているか
- [ ] ゲート分離精度（AUC/AP）が維持されているか

## 🚀 次のステップ

- **exp0036**: 両専門家同時最適化版MoE
- **Future**: 3専門家MoE（Low/Mid/High）
- **Future**: 温度パラメータτの再最適化

## 📚 参考実験

- **exp0034**: High専門家Optuna最適化版MoE
- **exp0031**: 木モデル版MoE（ベースライン）
- **exp0033**: ゼロ閾値最適化版MoE
- **exp0027**: StratifiedGKFoldでCV安定化（Optuna最適化実績）

## 📝 注意事項

- Low領域はサンプル数が多い（約70%）ため、パラメータ最適化の効果が大きい
- 正則化パラメータ（min_child_samples、reg_alpha、reg_lambda）が重要
- High専門家の最適化成功後に実施することで、相乗効果を期待
