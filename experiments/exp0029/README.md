# exp0029: 非負制約スタッキング (NNLS)

## 🎯 実験の目的

Phase 3として、LGBMとCatBoostのOOF予測を非負最小二乗法（NNLS）でスタッキングし、さらに派生特徴（`max(ŷ - 0.1, 0)`）を加えることで予測精度を向上させる。

## 📊 ベースライン (exp0028)

- **CV mean**: 0.2315 (std 0.0052)
- **OOF RMSE**: 0.2316
- **特徴**: StratifiedGroupKFold + Isotonic校正

## 🔧 実装内容

### 1. 非負制約スタッキング (NNLS)
- **入力特徴量**:
  - LGBM OOF予測
  - CatBoost OOF予測
  - `max(LGBM - 0.1, 0)` （正領域の強調）
  - `max(CatBoost - 0.1, 0)` （正領域の強調）
- **最適化**: `scipy.optimize.nnls` で非負制約付き線形回帰
- **目的関数**: weighted MSE

### 2. 期待される効果
- グリッドサーチより柔軟な重み最適化
- 派生特徴により正領域（y >= 0.1）の予測精度向上
- 非負制約により解釈性の維持

## 📁 ファイル構成

```
exp0029/
├── training_with_nnls.ipynb      # メインノートブック
├── logs/
│   └── host_baseline_002_metrics.json
├── submissions/
│   └── submission_exp0029.csv
├── artifacts/
└── README.md                      # このファイル
```

## 🚀 実行方法

1. Jupyter Labで `training_with_nnls.ipynb` を開く
2. 全セルを順番に実行
3. NNLSスタッキングセクションで学習された係数を確認
4. 最終スコアを `logs/host_baseline_002_metrics.json` に記録

## 📝 評価指標

- **主要指標**: weighted RMSE (wRMSE)
- **補助指標**: 
  - NNLS学習係数
  - グリッドサーチとの比較
  - OOF改善量

## 🔍 検証ポイント

- [ ] NNLS係数が非負であることを確認
- [ ] グリッドサーチ vs NNLS のスコア比較
- [ ] 派生特徴の寄与度を確認
- [ ] README実験台帳に結果を記録

