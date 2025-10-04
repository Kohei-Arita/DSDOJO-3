# exp0030: Tweedie損失によるゼロ過剰データ対応

## 🎯 実験の目的

ゼロ過剰データ（zero-inflated data）に対応するため、LightGBMにTweedie損失を適用し、従来のL2損失（squared error）との性能を比較する。

## 📊 ベースライン (exp0027)

- **CV mean**: 0.2315 (std 0.0052)
- **OOF RMSE**: 0.2316
- **使用損失**: L2 (squared error)
- **データ特性**: 約69%が xAG < 0.1（ゼロ過剰）

## 🔧 実装内容

### 1. Tweedie損失の適用

**対象**: LightGBMのみ（CatBoostはTweedie非対応）

**パラメータ**:
```python
{
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1 ~ 1.9,  # Optunaで最適化
    'metric': 'rmse',  # 評価はRMSE
}
```

### 2. variance_powerの探索

- **探索範囲**: [1.1, 1.9]
- **最適化手法**: Optuna TPE sampler
- **目的関数**: weighted RMSE

### 3. 期待される効果

- ゼロ近傍の予測精度向上
- 正領域での過剰予測の抑制
- ゼロ過剰データに適した確率分布のモデリング

## 📁 ファイル構成

```
exp0030/
├── training_with_tweedie.ipynb   # メインノートブック
├── logs/
│   └── host_baseline_002_metrics.json
├── submissions/
│   └── submission_exp0030.csv
├── artifacts/
├── README.md                      # このファイル
└── IMPLEMENTATION_PLAN.md         # 詳細実装計画
```

## 🚀 実行方法

1. Jupyter Labで `training_with_tweedie.ipynb` を開く
2. 全セルを順番に実行
3. Tweedie vs L2の比較結果を確認
4. 最終スコアを `logs/host_baseline_002_metrics.json` に記録

## 📝 評価指標

- **主要指標**: weighted RMSE (wRMSE)
- **補助指標**: 
  - 正領域（y >= 0.1）のRMSE
  - 負領域（y < 0.1）のRMSE
  - 予測分布の比較

## 🔍 検証ポイント

- [ ] variance_powerの最適値を確認
- [ ] Tweedie vs L2 のOOFスコアを比較
- [ ] ゼロ近傍の予測精度を確認
- [ ] CatBoostとのブレンド効果を確認
- [ ] README実験台帳に結果を記録

## ⚠️ 注意事項

- CatBoostはTweedie損失をサポートしていないため、L2のまま使用
- 評価指標（wRMSE）は変更しない
- StratifiedGroupKFoldの分割は維持

## 📚 参考文献

- Tweedie distribution: https://en.wikipedia.org/wiki/Tweedie_distribution
- LightGBM Tweedie objective: https://lightgbm.readthedocs.io/en/latest/Parameters.html#tweedie_variance_power

