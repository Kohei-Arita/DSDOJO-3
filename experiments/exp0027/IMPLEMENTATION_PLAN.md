# EXP0027 実装プラン

## 実装方針

### ノートブック編集の最小化
- exp0026 の `training_with_catboost.ipynb` は8000行超の大規模ノート
- 直接編集は困難なため、**追加セルで段階的に拡張**する方式
- 実装ロジックは `scripts/` に分離し、テスト・再利用性を確保

### 実験の段階的実施
各Phaseを独立したセクションとして実装し、相互干渉を防ぐ：

1. **Phase 1（層化CV）**: 既存のGroupKFold生成セルの直後に追加
2. **Phase 2（Isotonic）**: ブレンド予測生成セルの直後に追加
3. **Phase 3（NNLS）**: Phase 2との比較セルを追加
4. **Phase 4（Tweedie）**: LGBM学習セクションを複製・修正
5. **Phase 5（平滑重み）**: Phase 4との比較で検証

## Phase 1: StratifiedGroupKFold導入

### 挿入位置
既存の「GroupKFoldでfold割当」セル（行4706付近）の直後

### 追加セル構成

#### セル1: 層化fold生成
```python
# StratifiedGroupKFold による fold 割当
from scripts.cv import make_stratified_group_folds, evaluate_fold_balance

print("StratifiedGroupKFold による fold 再割当...")
train_df['fold_sgkf'] = make_stratified_group_folds(
    train_df,
    y_col='xAG',
    group_col='match_id',
    threshold=0.1,
    n_splits=5,
    n_bins=5,
    seed=SEED
)

# fold balance評価
fold_balance_gkf = evaluate_fold_balance(train_df, fold_col='fold', y_col='xAG')
fold_balance_sgkf = evaluate_fold_balance(train_df, fold_col='fold_sgkf', y_col='xAG')

print("\n=== GroupKFold (既存) ===")
display(fold_balance_gkf)
print("\n=== StratifiedGroupKFold (新) ===")
display(fold_balance_sgkf)

# 正例率の標準偏差を比較
std_gkf = fold_balance_gkf['pos_rate'].std()
std_sgkf = fold_balance_sgkf['pos_rate'].std()
print(f"\n正例率の標準偏差:")
print(f"  GroupKFold:           {std_gkf:.6f}")
print(f"  StratifiedGroupKFold: {std_sgkf:.6f}")
print(f"  改善率: {(1 - std_sgkf/std_gkf)*100:.1f}%")
```

#### セル2: 層化CVでの学習（LGBM）
既存のLGBM CVループを複製し、`fold` → `fold_sgkf` に変更

#### セル3: 層化CVでの学習（CatBoost）
既存のCatBoost CVループを複製し、`fold` → `fold_sgkf` に変更

#### セル4: 結果比較
```python
# GroupKFold vs StratifiedGroupKFold の OOF wRMSE 比較
print("=== CV方式の比較 ===")
print(f"GroupKFold OOF wRMSE:           {oof_score_gkf:.4f}")
print(f"StratifiedGroupKFold OOF wRMSE: {oof_score_sgkf:.4f}")
print(f"差分: {oof_score_sgkf - oof_score_gkf:+.4f}")

# fold別の安定性
cv_std_gkf = np.std(cv_scores_gkf)
cv_std_sgkf = np.std(cv_scores_sgkf)
print(f"\nCV標準偏差:")
print(f"  GroupKFold:           {cv_std_gkf:.4f}")
print(f"  StratifiedGroupKFold: {cv_std_sgkf:.4f}")
```

### 期待される効果
- fold別正例率の標準偏差が減少（目標: 50%以上改善）
- CV標準偏差が減少（fold間の安定性向上）
- OOF wRMSEは同等か微改善

---

## Phase 2: 正領域Isotonic校正

### 挿入位置
既存のブレンド予測生成セル（行7200付近）の直後

### 追加セル構成

#### セル1: Isotonic校正の学習・適用
```python
from scripts.calibration import fit_isotonic_positive, apply_isotonic_positive

print("=== 正領域Isotonic校正 ===")

# OOFで校正モデルを学習
iso_model = fit_isotonic_positive(
    y_true=y_train.values,
    y_pred=blended_oof_preds,
    threshold=0.1,
    pos_weight=5.0
)

# OOFに適用
blended_oof_calib = apply_isotonic_positive(
    blended_oof_preds,
    iso_model,
    threshold=0.1
)

# テスト予測に適用
blended_test_calib = apply_isotonic_positive(
    blended_test_preds,
    iso_model,
    threshold=0.1
)

# 評価
oof_score_raw = weighted_rmse(y_train, blended_oof_preds)
oof_score_calib = weighted_rmse(y_train, blended_oof_calib)

print(f"ブレンドOOF（校正前）: {oof_score_raw:.4f}")
print(f"ブレンドOOF（校正後）: {oof_score_calib:.4f}")
print(f"改善: {oof_score_calib - oof_score_raw:+.4f}")
```

#### セル2: 校正効果の可視化
```python
# 正領域の予測分布比較
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 校正前
axes[0].scatter(blended_oof_preds, y_train, alpha=0.3, s=5)
axes[0].plot([0, 0.6], [0, 0.6], 'r--', label='Perfect')
axes[0].axhline(0.1, color='orange', linestyle='--', alpha=0.5)
axes[0].axvline(0.1, color='orange', linestyle='--', alpha=0.5)
axes[0].set_xlabel('予測値（校正前）')
axes[0].set_ylabel('真値')
axes[0].legend()

# 校正後
axes[1].scatter(blended_oof_calib, y_train, alpha=0.3, s=5)
axes[1].plot([0, 0.6], [0, 0.6], 'r--', label='Perfect')
axes[1].axhline(0.1, color='orange', linestyle='--', alpha=0.5)
axes[1].axvline(0.1, color='orange', linestyle='--', alpha=0.5)
axes[1].set_xlabel('予測値（校正後）')
axes[1].set_ylabel('真値')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### 期待される効果
- OOF wRMSE: -0.001〜-0.003 の改善
- 正領域での予測歪みの補正（y=x線への収束）

---

## Phase 3: 非負スタッキング

### 挿入位置
Phase 2 の直後（または並列比較セクションとして）

### 追加セル構成

#### セル1: スタッキング特徴の生成
```python
from scripts.stacking import create_stack_features, fit_nnls_stack, predict_nnls_stack

print("=== 非負制約スタッキング ===")

# OOFでの特徴生成
base_preds_oof = {
    'lgbm': oof_preds,  # LGBM OOF
    'catboost': catboost_oof_preds  # CatBoost OOF
}

X_stack_oof, feature_names = create_stack_features(
    base_preds_oof,
    threshold=0.1,
    include_positive_excess=True
)

print(f"スタッキング特徴: {feature_names}")

# 重み付きNNLS学習
sample_weights = make_sample_weight(y_train.values)
nnls_coefs = fit_nnls_stack(X_stack_oof, y_train.values, sample_weights)

print(f"\nNNLS係数:")
for name, coef in zip(feature_names, nnls_coefs):
    print(f"  {name}: {coef:.6f}")

# OOF予測
nnls_oof_preds = predict_nnls_stack(X_stack_oof, nnls_coefs)
nnls_oof_score = weighted_rmse(y_train, nnls_oof_preds)

print(f"\n=== OOFスコア比較 ===")
print(f"グリッドブレンド:     {best_blend_score:.4f}")
print(f"NNLSスタッキング:    {nnls_oof_score:.4f}")
print(f"差分: {nnls_oof_score - best_blend_score:+.4f}")
```

#### セル2: テスト予測
```python
# テスト特徴生成
base_preds_test = {
    'lgbm': lgbm_test_preds,
    'catboost': catboost_test_preds
}

X_stack_test, _ = create_stack_features(
    base_preds_test,
    threshold=0.1,
    include_positive_excess=True
)

# テスト予測
nnls_test_preds = predict_nnls_stack(X_stack_test, nnls_coefs)

# 負値チェック
print(f"テスト予測の最小値: {nnls_test_preds.min():.6f}")
print(f"負値の数: {(nnls_test_preds < 0).sum()}")
```

### 期待される効果
- 非負制約により、ゼロ域での過剰相殺を防止
- OOF wRMSE: グリッドブレンドと同等以上

---

## Phase 4: Tweedie損失

### 挿入位置
既存のLGBM Optuna最適化セクションを複製・修正

### 追加セル構成

#### セル1: Tweedie最適化
```python
print("=== Tweedie損失でのLGBM学習 ===")

def objective_tweedie(trial):
    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
        'metric': 'rmse',
        'num_leaves': trial.suggest_int('num_leaves', 10, 64),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'random_state': SEED
    }
    
    # Fold 1でのCV
    trn_mask = (train_df['fold_sgkf'] != 0)
    val_mask = (train_df['fold_sgkf'] == 0)
    
    # ... (既存のCV実装を利用) ...
    
    return val_rmse

study_tweedie = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
study_tweedie.optimize(objective_tweedie, n_trials=30, show_progress_bar=True)

print(f"\n最適パラメータ:")
print(f"  tweedie_variance_power: {study_tweedie.best_params['tweedie_variance_power']:.3f}")
```

#### セル2: Tweedie CVと評価
既存のCV実装を複製し、Tweedieパラメータで学習

### 期待される効果
- ゼロインフレ構造への適合性向上
- OOF wRMSE: baseline比で-0.002以上の改善（期待）

---

## Phase 5: 平滑化重み（optional）

### 挿入位置
Phase 4 の直後（または並列実験）

### 追加セル構成

#### セル1: 重みスキームの比較
```python
from scripts.weights import make_sample_weight as make_weight_custom
from scripts.weights import compare_weight_schemes

print("=== 重みスキームの比較 ===")

# 統計比較
weight_stats = compare_weight_schemes(
    y_train.values,
    threshold=0.1,
    pos_weight=5.0,
    tau_values=[0.02, 0.03, 0.05]
)

for scheme, stats in weight_stats.items():
    print(f"\n{scheme}:")
    print(f"  平均重み: {stats['mean_weight']:.3f}")
    print(f"  有効サンプル数: {stats['effective_n']:.1f} / {stats['total_samples']}")
```

#### セル2: 平滑重みでの学習
```python
# tau=0.03 での学習
lgbm_params_smooth = lgbm_params.copy()
# (学習時にweightを差し替え)

# ... CV実装 ...
```

### 期待される効果
- fold間のCV標準偏差がさらに減少
- 閾値近傍での予測精度向上

---

## 実装順序とマイルストーン

### Week 1: 基盤整備
- [x] scripts/ モジュール実装
- [ ] exp0027 ディレクトリ・README作成
- [ ] Phase 1（層化CV）の実装・検証

### Week 2: 校正・スタッキング
- [ ] Phase 2（Isotonic）の実装・検証
- [ ] Phase 3（NNLS）の実装・検証
- [ ] Phase 1-3の結果まとめ

### Week 3: 損失関数・重み
- [ ] Phase 4（Tweedie）の実装・検証
- [ ] Phase 5（平滑重み）の実装・検証（optional）
- [ ] 全体比較と最終モデル選定

### Week 4: 最終化
- [ ] 最良手法の組み合わせ検証
- [ ] 提出ファイル生成
- [ ] ドキュメント整備

---

## 技術的注意点

### ノートブック編集
- ipynb は JSON形式なので、`edit_notebook` ツールを使用
- セル追加は `is_new_cell=True` で対応
- 既存セルの修正は最小限に

### 実験の再現性
- 全実験で `SEED=42` を統一
- fold割当は Phase 1 で固定し、以降共通利用
- OOF予測は毎回保存（後続の校正・スタッキングで利用）

### デバッグ戦略
- 各Phaseで小規模テスト（1 foldのみ）を先行実施
- ログ・メトリクスは必ず保存（後で比較可能に）
- 異常値検出（負値、NaN、外れ値）を各ステップで実施

### 失敗時の対応
- **層化失敗**: n_bins を減らす、または GroupKFold に戻す
- **Isotonic劣化**: 閾値を調整、または適用を見送る
- **NNLS係数異常**: positive_excess特徴を外す
- **Tweedie効果なし**: variance_power を固定して他パラメータのみ探索

