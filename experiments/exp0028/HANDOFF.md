# exp0028 引き継ぎドキュメント - Isotonic校正実装

## 📋 現在の状況

### 完了済み (exp0027)
- ✅ **Phase 1: StratifiedGroupKFold** - 完了
  - `scripts/cv.py` に `make_stratified_group_folds()` を実装
  - CV標準偏差: 0.0058 → 0.0052 に改善
  - 正例率の均一化に成功 (Val正例率std=0.0029)

### 次のタスク (exp0028)
- 🎯 **Phase 2: Isotonic校正** - これから実装
  - ブレンド後の予測に対して、正領域（y >= 0.1）のみIsotonic Regressionを適用
  - `scripts/calibration.py` は既に実装済み

---

## 🎯 Phase 2 実装計画: Isotonic校正

### 目的
ブレンド後の予測値を正領域（xAG >= 0.1）でのみ校正し、予測精度を向上させる。

### 実装内容

#### 1. 使用する関数（既に実装済み）
`scripts/calibration.py` に以下の関数が用意されています：

```python
from scripts.calibration import fit_isotonic_positive, apply_isotonic_positive

# 学習
iso_model = fit_isotonic_positive(
    y_oof_true,      # OOF真値
    y_oof_pred,      # OOF予測値
    threshold=0.1,   # 正例の閾値
    pos_weight=5.0   # 正例の重み（wRMSEと一致）
)

# 適用
calibrated_pred = apply_isotonic_positive(
    predictions,     # 校正前の予測値
    iso_model,       # 学習済みモデル
    threshold=0.1    # 正例の閾値
)
```

#### 2. ノートブックへの追加箇所

**場所**: ブレンド予測の作成後、提出ファイル作成前

既存のノートブックでは以下のセルの後：
```python
# 最適な重みでブレンド予測を作成
blended_test_preds = best_lgbm_weight * lgbm_test_preds + best_catboost_weight * catboost_test_preds
```

**新規セルを追加**:

```python
# [exp0028] Isotonic校正: 正領域（y >= 0.1）のみを校正
print("=" * 80)
print("Isotonic校正を適用中...")
print("=" * 80)

from scripts.calibration import fit_isotonic_positive, apply_isotonic_positive

# OOF予測で校正モデルを学習
print("\n1. OOF予測でIsotonicモデルを学習中...")
iso_model = fit_isotonic_positive(
    y_oof_true=y_train.values,
    y_oof_pred=blended_oof_preds,  # ブレンド後のOOF予測
    threshold=0.1,
    pos_weight=5.0
)

# OOF予測を校正して評価
print("\n2. OOF予測を校正中...")
calibrated_oof_preds = apply_isotonic_positive(
    blended_oof_preds,
    iso_model,
    threshold=0.1
)

# 校正前後のスコア比較
before_rmse = weighted_rmse(y_train.values, blended_oof_preds)
after_rmse = weighted_rmse(y_train.values, calibrated_oof_preds)

print(f"\n【校正効果】")
print(f"  校正前 wRMSE: {before_rmse:.6f}")
print(f"  校正後 wRMSE: {after_rmse:.6f}")
print(f"  改善量: {before_rmse - after_rmse:.6f} ({(before_rmse - after_rmse) / before_rmse * 100:.2f}%)")

# 正領域・負領域別の分析
pos_mask = y_train.values >= 0.1
neg_mask = y_train.values < 0.1

print(f"\n【正領域（y >= 0.1）】")
print(f"  サンプル数: {pos_mask.sum()}")
print(f"  校正前 RMSE: {np.sqrt(mean_squared_error(y_train[pos_mask], blended_oof_preds[pos_mask])):.6f}")
print(f"  校正後 RMSE: {np.sqrt(mean_squared_error(y_train[pos_mask], calibrated_oof_preds[pos_mask])):.6f}")

print(f"\n【負領域（y < 0.1）】")
print(f"  サンプル数: {neg_mask.sum()}")
print(f"  校正前 RMSE: {np.sqrt(mean_squared_error(y_train[neg_mask], blended_oof_preds[neg_mask])):.6f}")
print(f"  校正後 RMSE: {np.sqrt(mean_squared_error(y_train[neg_mask], calibrated_oof_preds[neg_mask])):.6f}")

# 3. テストデータに適用
print("\n3. テストデータに校正を適用中...")
calibrated_test_preds = apply_isotonic_positive(
    blended_test_preds,
    iso_model,
    threshold=0.1
)

print(f"\n【テスト予測の統計】")
print(f"  校正前:")
print(f"    Mean: {blended_test_preds.mean():.6f}")
print(f"    Std:  {blended_test_preds.std():.6f}")
print(f"    Min:  {blended_test_preds.min():.6f}")
print(f"    Max:  {blended_test_preds.max():.6f}")
print(f"  校正後:")
print(f"    Mean: {calibrated_test_preds.mean():.6f}")
print(f"    Std:  {calibrated_test_preds.std():.6f}")
print(f"    Min:  {calibrated_test_preds.min():.6f}")
print(f"    Max:  {calibrated_test_preds.max():.6f}")

# 4. ブレンド予測を校正版に置き換え
blended_test_preds = calibrated_test_preds
blended_oof_preds = calibrated_oof_preds

print("\n✅ Isotonic校正完了")
```

---

## 📝 Cursorでのノートブック編集方法

### 1. セル番号の確認方法

```bash
# ノートブック内のセル数を確認
python -c "import json; nb = json.load(open('experiments/exp0027/training_with_stratified_cv.ipynb')); print(f'Total cells: {len(nb[\"cells\"])}')"

# 特定のキーワードを含むセルを検索
python -c "import json; nb = json.load(open('experiments/exp0027/training_with_stratified_cv.ipynb')); [(print(f'Cell {i}: {c[\"source\"][0][:80]}...')) for i, c in enumerate(nb['cells']) if 'blended_test_preds' in ''.join(c.get('source', []))]"
```

### 2. edit_notebook ツールの使い方

#### 新しいセルを追加する場合

```xml
<invoke name="edit_notebook">
<parameter name="target_notebook">experiments/exp0028/training_with_isotonic.ipynb