# exp0028: Isotonic校正実装

## 🎯 実験の目的

Phase 2として、ブレンド後の予測値に対してIsotonic Regression校正を適用し、特に正領域（xAG >= 0.1）の予測精度を向上させる。

## 📊 ベースライン (exp0027)

- **CV mean**: 0.2315 (std 0.0052)
- **OOF RMSE**: 0.2316
- **特徴**: StratifiedGroupKFoldによる安定したCV分割

## 🔧 実装内容

### 1. Isotonic校正の適用
- **対象**: ブレンド後の予測値（LGBM + CatBoost）
- **方法**: 正領域（y >= 0.1）のみを校正
- **重み**: wRMSEと同じ重み（正例: 5.0, 負例: 1.0）

### 2. 期待される効果
- 正領域での予測精度向上
- wRMSEスコアの改善
- モデルのキャリブレーション向上

## 📁 ファイル構成

```
exp0028/
├── training_with_isotonic.ipynb   # メインノートブック
├── logs/
│   └── host_baseline_002_metrics.json
├── submissions/
│   └── submission_exp0028.csv
├── artifacts/
├── README.md                       # このファイル
└── HANDOFF.md                      # 引き継ぎドキュメント
```

## 🚀 実行方法

1. Jupyter Labで `training_with_isotonic.ipynb` を開く
2. 全セルを順番に実行
3. Isotonic校正セクションで校正前後の比較を確認
4. 最終スコアを `logs/host_baseline_002_metrics.json` に記録

## 📝 評価指標

- **主要指標**: weighted RMSE (wRMSE)
- **補助指標**: 
  - 正領域（y >= 0.1）のRMSE
  - 負領域（y < 0.1）のRMSE
  - OOF改善量

## 🔍 検証ポイント

- [ ] 校正前後のwRMSEを比較
- [ ] 正領域・負領域別のスコアを確認
- [ ] テスト予測の分布を確認
- [ ] README実験台帳に結果を記録

