# exp0024 Data Leakage Fix - 実験ノート

## 概要

- ベース: `exp0020_catboost_blend`
- 目的: 特徴量の追加とリーク監査の修正
- 結果: CV mean 0.2309 (std 0.0057) / OOF 0.2310 / Optuna best 0.2303

## 考察

本実験では、eΔxT系およびプレー連鎖に関連する特徴群を強化し、リーク監査レポート（`DATA_LEAKAGE_AUDIT_REPORT.md`）で指摘された分割上の注意点に基づく点検を行いました。特に`GCA_1`や`SCA_1`のようなプレーメイク関連指標、および`nstep_to_shot`のようなプレー連鎖の短さを示す特徴が大きく寄与しました。

### 特徴量重要度 Top 10（CV平均）

```
feature                              mean          std
GCA_1                             5435.034380   314.715236
SCA_1                             2412.881013   150.499600
Squad                             2144.977234   334.492667
nstep_to_shot                     2026.499182   199.879277
openplay_edxt_positive_sum        1330.696491   160.229099
pass_edxt_max                      993.121604   156.286109
xt_learned_positive_delta_mean     803.519231    85.251470
openplay_edxt_scaled_positive_sum  543.271319   166.256540
cross_edxt_max                     420.539780   133.515785
pass_edxt_scaled_max               396.010502   128.496748
```

これらの重要度は、プレーの創出力（GCA/SCA）、連鎖の効率（nstep_to_shot）、および学習型xT由来の増分や最大値が、xAG予測に対して顕著な説明力を持つことを示しています。特に`xt_learned_positive_delta_mean`は、固定マトリクスxTでは捉えにくい文脈依存の価値増分を反映し、他の位置・進行系特徴と相補的に作用しています。

## 参考

- ログ: `logs/host_baseline_002_metrics.json`
- リーク監査: `DATA_LEAKAGE_AUDIT_REPORT.md`
- 提出物: `logs/submission_lgbm_only.csv`, `logs/submission_catboost_only.csv`, `logs/submission_blend_lgbm_catboost.csv`


