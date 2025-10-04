# exp0031: 木モデル版MoE（Low/High専門家 + 温度ゲート）

## 概要

**提案元アイデア**（exp0035として提示されていたもの）:
> MoE を"木モデル版"で：Low/High 専門家 + 温度ゲート  
> NN ではなく、低域（<0.1）専用 LGBM と 高域（≥0.1）専用 LGBM を作る 2専門家。  
> ゲートは exp0031 の校正済み p̂ を使用、Top-1（hard） と soft（温度 τ） を OOF で比較。

本実験では、上記アイデアを**exp0031**で実装しました。

---

## 目的

- **Low専門家**: y < 0.1 のサンプルのみで学習したLGBM回帰モデル
- **High専門家**: y ≥ 0.1 のサンプルのみで学習したLGBM回帰モデル
- **ゲート分類器**: y ≥ 0.1 を判別するLGBM分類器（OOF + Isotonic校正）
- **合成方式**:
  - **Soft合成**: \( \hat{y} = g^\tau \cdot \hat{\mu}_{high} + (1 - g^\tau) \cdot \hat{\mu}_{low} \)
  - **Hard合成**: \( \hat{y} = \mathbb{1}[g \geq t] \cdot \hat{\mu}_{high} + \mathbb{1}[g < t] \cdot \hat{\mu}_{low} \)
  - OOFで温度τ（Soft）と閾値t（Hard）を最適化し、ベストを選択

---

## 実装手順

### Step 1: ゲート分類器の学習

- **入力**: 全特徴量（`all_features`）
- **ラベル**: `is_positive = (y >= 0.1)`
- **手法**: LGBMClassifier（`objective=binary`, `scale_pos_weight`で不均衡対応）
- **出力**: OOF確率 `gate_oof_raw`, テスト確率 `gate_test_raw`
- **評価**: AUC, AP

### Step 2: ゲート確率のIsotonic校正

- **入力**: `gate_oof_raw`, `is_positive`
- **手法**: `IsotonicRegression(increasing=True, out_of_bounds="clip")`
- **出力**: 校正済みOOF確率 `gate_oof_cal`, テスト確率 `gate_test_cal`
- **確認**: 校正曲線（10分位）、AUC/AP

### Step 3: Low専門家の学習

- **入力**: Low領域（y < 0.1）のサンプルのみ
- **手法**: LGBMRegressor（`objective=regression`, wRMSE重み適用）
- **出力**: OOF予測 `low_oof_preds`, テスト予測 `low_test_preds`
- **評価**: Low領域のみのwRMSE

### Step 4: High専門家の学習

- **入力**: High領域（y ≥ 0.1）のサンプルのみ
- **手法**: LGBMRegressor（`objective=regression`, wRMSE重み適用）
- **出力**: OOF予測 `high_oof_preds`, テスト予測 `high_test_preds`
- **評価**: High領域のみのwRMSE

### Step 5: MoE合成の最適化

#### Soft合成の温度τ最適化
- **探索範囲**: τ ∈ {0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0}
- **評価**: OOF全体のwRMSE
- **出力**: ベストτ, ベストスコア

#### Hard合成の閾値t最適化
- **探索範囲**: t ∈ {0.3, 0.4, 0.5, 0.6, 0.7}
- **評価**: OOF全体のwRMSE
- **出力**: ベストt, ベストスコア

#### 最終選択
- Soft vs Hard を比較し、より良いwRMSEの方を採用

### Step 6: ベースライン比較・メトリクス保存

- **比較対象**:
  - LGBM単体
  - NNLS統合（exp0029）
  - Isotonic校正（exp0028）
- **保存**:
  - `logs/host_moe_tree_002_metrics.json`: CV/OOF/ゲート/専門家スコア、改善量
  - `artifacts/oof_predictions_moe.csv`: OOF詳細（ゲート確率、Low/High予測、MoE予測）

### Step 7: 提出ファイル作成

- **出力**: `submissions/host_moe_tree_002_submissions.csv`
- **可視化**: OOF/テスト予測分布、ゲート確率分布

---

## 期待される効果

1. **専門化**: Low/High専門家が各領域に特化し、単一モデルより精度向上
2. **校正**: ゲート確率をIsotonicで校正することで、合成の安定性向上
3. **温度制御**: τ>1でゲート確率を縮小→過剰なHigh依存を抑制、τ<1で拡大→明確な分離
4. **柔軟性**: Soft/Hard合成をOOFで選択し、データに最適な方式を採用

---

## 技術的留意点

- **リーク防止**: ゲート確率/専門家予測は全てOOFで生成
- **重み整合**: wRMSE評価と学習重みを統一（閾値0.1、正例重み5.0）
- **非負クリップ**: 全予測を `np.clip(preds, 0, None)` で非負化
- **乱数**: SEED=42で再現性確保（専門家は`SEED+fold`, `SEED+fold+100`で分離）
- **型一貫性**: `match_id`, `player_id`はstring型を維持

---

## ファイル構成

```
experiments/exp0031/
├── training_with_nnls.ipynb       # 実装ノートブック（MoE追加済み）
├── IMPLEMENTATION_PLAN.md         # 本ファイル
├── logs/
│   └── host_moe_tree_002_metrics.json  # メトリクス
├── artifacts/
│   └── oof_predictions_moe.csv    # OOF詳細
└── submissions/
    └── host_moe_tree_002_submissions.csv  # 提出ファイル
```

---

## 実装完了

- ✅ ゲート分類器（LGBM binary）をOOFで学習してp_raw_oof/p_raw_testを取得
- ✅ IsotonicRegressionでゲート確率を校正（p_cal_oof/p_cal_test）し、AUC/APを確認
- ✅ Low専門家（y<0.1のみ学習）をOOFで学習してy_low_oof/y_low_testを取得
- ✅ High専門家（y≥0.1のみ学習）をOOFで学習してy_high_oof/y_high_testを取得
- ✅ Soft合成（温度τでOOFグリッド探索）とHard合成（閾値tでOOF探索）を比較しベスト設定を決定
- ✅ OOF wRMSE（fold別/平均/std）、テスト予測、メトリクスをJSONで保存
- ✅ 提出ファイル（host_moe_tree_002_submissions.csv）を作成

---

## 次のステップ

1. ノートブックを実行してスコアを確認
2. ベースライン（NNLS/Isotonic）との差分を評価
3. 改善が見られた場合、README.mdの実験履歴に追記
4. さらなる改善案（例: τの細かい探索、専門家のハイパーパラメータ最適化）を検討

