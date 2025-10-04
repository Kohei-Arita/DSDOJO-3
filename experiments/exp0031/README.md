# exp0031: 木モデル版MoE（Low/High専門家 + 温度ゲート）

## 🎯 実験概要

**提案**: Low域（y<0.1）専門家とHigh域（y≥0.1）専門家を作り、校正済みゲート確率で温度制御されたソフト合成を行う「木モデル版MoE」

**ベース**: exp0029（NNLS統合）から派生

---

## 🔬 手法

### 1. ゲート分類器（y≥0.1の判別）
- **モデル**: LGBMClassifier（binary）
- **不均衡対応**: `scale_pos_weight`
- **校正**: IsotonicRegression（OOFで学習）
- **評価**: AUC, AP

### 2. 専門家回帰
- **Low専門家**: y<0.1のサンプルのみで学習（LGBM）
- **High専門家**: y≥0.1のサンプルのみで学習（LGBM）
- **重み**: wRMSE整合（閾値0.1、正例重み5.0）
- **評価**: 各領域のwRMSE

### 3. MoE合成
#### Soft合成
\[
\hat{y} = g^\tau \cdot \hat{\mu}_{high} + (1 - g^\tau) \cdot \hat{\mu}_{low}
\]
- **温度τ**: {0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0} をOOF探索
- **τ>1**: ゲート確率縮小（High依存を抑制）
- **τ<1**: ゲート確率拡大（明確な分離）

#### Hard合成
\[
\hat{y} = \mathbb{1}[g \geq t] \cdot \hat{\mu}_{high} + \mathbb{1}[g < t] \cdot \hat{\mu}_{low}
\]
- **閾値t**: {0.3, 0.4, 0.5, 0.6, 0.7} をOOF探索

#### 最終選択
- Soft vs Hard を比較し、wRMSE最小の方を採用

---

## 📊 期待される効果

1. **専門化**: 各領域に特化したモデルで精度向上
2. **校正**: ゲート確率の校正で合成の安定性向上
3. **温度制御**: データに応じた柔軟な合成比率
4. **MoE思想**: 条件付き計算（専門家の使い分け）

---

## 📝 評価指標

- **主要指標**: weighted RMSE (wRMSE)
- **補助指標**: 
  - ゲートAUC/AP
  - Low/High専門家の各領域wRMSE
  - Fold別CV安定性

---

## 🔍 検証ポイント

- [ ] ゲート分類器のAUC/APが十分高いか（0.85以上が目安）
- [ ] Low/High専門家が各領域で単一モデルより優れているか
- [ ] Soft/Hard合成のどちらが良いか
- [ ] ベースライン（NNLS/Isotonic）比で改善しているか
- [ ] Fold間の安定性（CV std）

---

## 📂 成果物

- `training_with_nnls.ipynb`: MoE実装追加済み（Cell 66-72）
- `logs/host_moe_tree_002_metrics.json`: メトリクス
- `artifacts/oof_predictions_moe.csv`: OOF詳細
- `submissions/host_moe_tree_002_submissions.csv`: 提出ファイル
- `IMPLEMENTATION_PLAN.md`: 実装計画書

---

## 🚀 実行方法

```python
# Jupyter Notebook/Labで実行
jupyter lab experiments/exp0031/training_with_nnls.ipynb

# Cell 66-72がMoE実装部分
# 既存セル（〜65）を実行後、MoEセルを実行
```

---

## 💡 改善案（次ステップ）

1. **温度τの細分化**: τ∈[0.8, 1.2] を0.05刻みで探索
2. **専門家のハイパラ最適化**: Low/Highで別々にOptuna最適化
3. **log1p変換**: High専門家をlog1p(y)で学習（裾の取り扱い改善）
4. **3専門家MoE**: Low/Mid/High の3領域分割
5. **CatBoost専門家**: LGBMと並行してCatBoost版も学習

---

## 📖 参考

- **MoE (Mixture of Experts)**: 条件付き計算で専門家を使い分ける手法
- **Isotonic Regression**: 単調性を保ちながら校正（scikit-learn）
- **温度スケーリング**: NN分類器の校正でよく使われる手法をMoEに応用

---

## 🏷️ タグ

`MoE` `Mixture-of-Experts` `LightGBM` `Isotonic` `Calibration` `Temperature-Scaling` `Expert-Specialization`

