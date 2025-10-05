# exp0035ノートブックへの追加手順

## 📍 追加位置

既存の応用特徴量生成セクション（`build_pass_network_centrality` などの後）に以下のセルを追加してください。

---

## 🔧 追加セル1: 新規GCA特徴量のインポートと生成

```python
# ============================================================
# 新規GCA特徴量の追加 (exp0035拡張)
# ============================================================

from scripts.advanced_features import (
    build_gca_spatial_features,
    build_linebreak_packing_features,
    build_pass_chain_quality_features,
    build_box_entry_receiving_features,
    build_setplay_bodypart_features,
)

print("=" * 60)
print("🚀 新規GCA特徴量の生成開始")
print("=" * 60)

# 1. GCA空間特徴量
print("\n[1/5] GCA空間特徴量（ゾーン14、ハーフスペース、カットバック）")
gca_spatial_train = build_gca_spatial_features(
    actions=action_data_train,
    match_col="match_id",
    player_col="player_id",
    team_col="team_id",
    type_col="type_name",
    result_col="result_name",
)
gca_spatial_test = build_gca_spatial_features(
    actions=action_data_test,
    match_col="match_id",
    player_col="player_id",
    team_col="team_id",
    type_col="type_name",
    result_col="result_name",
)
print(f"   ✅ 生成完了: {gca_spatial_train.shape[1]-2}列 (train: {len(gca_spatial_train)} rows)")

# 2. ラインブレイク/パッキング特徴量
print("\n[2/5] ラインブレイク/パッキング特徴量")
linebreak_train = build_linebreak_packing_features(
    actions=action_data_train,
    match_col="match_id",
    player_col="player_id",
    type_col="type_name",
    result_col="result_name",
)
linebreak_test = build_linebreak_packing_features(
    actions=action_data_test,
    match_col="match_id",
    player_col="player_id",
    type_col="type_name",
    result_col="result_name",
)
print(f"   ✅ 生成完了: {linebreak_train.shape[1]-2}列 (train: {len(linebreak_train)} rows)")

# 3. パス連鎖品質特徴量
print("\n[3/5] パス連鎖品質特徴量（1-2、三人目、速攻）")
pass_chain_train = build_pass_chain_quality_features(
    actions=action_data_train,
    match_col="match_id",
    player_col="player_id",
    team_col="team_id",
    type_col="type_name",
    time_col="time_seconds",
)
pass_chain_test = build_pass_chain_quality_features(
    actions=action_data_test,
    match_col="match_id",
    player_col="player_id",
    team_col="team_id",
    type_col="type_name",
    time_col="time_seconds",
)
print(f"   ✅ 生成完了: {pass_chain_train.shape[1]-2}列 (train: {len(pass_chain_train)} rows)")

# 4. PA進入受け手文脈特徴量
print("\n[4/5] PA進入受け手文脈特徴量")
box_entry_train = build_box_entry_receiving_features(
    actions=action_data_train,
    match_col="match_id",
    player_col="player_id",
    type_col="type_name",
    result_col="result_name",
)
box_entry_test = build_box_entry_receiving_features(
    actions=action_data_test,
    match_col="match_id",
    player_col="player_id",
    type_col="type_name",
    result_col="result_name",
)
print(f"   ✅ 生成完了: {box_entry_train.shape[1]-2}列 (train: {len(box_entry_train)} rows)")

# 5. セットプレー/ボディパート特徴量
print("\n[5/5] セットプレー/ボディパート特徴量")
setplay_bp_train = build_setplay_bodypart_features(
    actions=action_data_train,
    match_col="match_id",
    player_col="player_id",
    type_col="type_name",
    bodypart_col="bodypart_name",
)
setplay_bp_test = build_setplay_bodypart_features(
    actions=action_data_test,
    match_col="match_id",
    player_col="player_id",
    type_col="type_name",
    bodypart_col="bodypart_name",
)
print(f"   ✅ 生成完了: {setplay_bp_train.shape[1]-2}列 (train: {len(setplay_bp_train)} rows)")

print("\n" + "=" * 60)
print("✅ 全ての新規GCA特徴量生成完了")
print("=" * 60)
```

---

## 🔧 追加セル2: train_df/test_dfへのマージ

```python
# ============================================================
# 新規特徴量のマージ
# ============================================================

print("\n📊 新規特徴量をtrain_df/test_dfにマージ中...")

# train_dfに統合
train_df = train_df.merge(gca_spatial_train, on=["match_id", "player_id"], how="left")
train_df = train_df.merge(linebreak_train, on=["match_id", "player_id"], how="left")
train_df = train_df.merge(pass_chain_train, on=["match_id", "player_id"], how="left")
train_df = train_df.merge(box_entry_train, on=["match_id", "player_id"], how="left")
train_df = train_df.merge(setplay_bp_train, on=["match_id", "player_id"], how="left")

# test_dfに統合
test_df = test_df.merge(gca_spatial_test, on=["match_id", "player_id"], how="left")
test_df = test_df.merge(linebreak_test, on=["match_id", "player_id"], how="left")
test_df = test_df.merge(pass_chain_test, on=["match_id", "player_id"], how="left")
test_df = test_df.merge(box_entry_test, on=["match_id", "player_id"], how="left")
test_df = test_df.merge(setplay_bp_test, on=["match_id", "player_id"], how="left")

# 新規特徴量リストの定義
new_gca_features = [
    # GCA空間特徴量 (10列)
    "zone14_origin_pass_count",
    "zone14_origin_pass_success_rate",
    "zone14_preGCA_count",
    "halfspace_L_to_box_count",
    "halfspace_L_to_box_success_rate",
    "halfspace_R_to_box_count",
    "halfspace_R_to_box_success_rate",
    "cutback_count",
    "cutback_success_rate",
    "cutback_next_shot_rate",

    # ラインブレイク/パッキング (5列)
    "linebreak_third_transition_count",
    "linebreak_third_transition_rate",
    "through_channel_pass_count",
    "through_channel_pass_rate",
    "packing_approx_score_mean",

    # パス連鎖品質 (3列)
    "one_two_chain_count",
    "third_man_release_count",
    "burst_window_SCA_rate",

    # PA進入受け手文脈 (5列)
    "box_entry_from_zone14_count",
    "box_entry_from_halfspace_L_count",
    "box_entry_from_halfspace_R_count",
    "facing_forward_share_in_box",
    "first_touch_shot_rate_in_box",

    # セットプレー/ボディパート (5列)
    "setplay_GCA_share",
    "openplay_GCA_share",
    "bodypart_on_key_pass_rate_right",
    "bodypart_on_key_pass_rate_left",
    "bodypart_on_key_pass_rate_head",
]

# 実際に存在する列のみフィルタ
new_gca_features = [f for f in new_gca_features if f in train_df.columns]

# NaN埋め（念のため）
for col in new_gca_features:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(0.0)
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(0.0)

print(f"\n✅ マージ完了:")
print(f"   - 新規特徴量: {len(new_gca_features)}列")
print(f"   - train_df shape: {train_df.shape}")
print(f"   - test_df shape: {test_df.shape}")

# 特徴量リスト表示
print(f"\n📋 新規GCA特徴量リスト:")
for i, feat in enumerate(new_gca_features, 1):
    print(f"   {i:2d}. {feat}")
```

---

## 🔧 追加セル3: 特徴量リストへの追加

```python
# ============================================================
# LightGBM/CatBoost特徴量リストに新規特徴量を追加
# ============================================================

# 既存の lgbm_features に追加
lgbm_features = lgbm_features + new_gca_features
print(f"✅ LightGBM特徴量: {len(lgbm_features)}列 (新規{len(new_gca_features)}列追加)")

# 既存の catboost_features に追加
catboost_features = catboost_features + new_gca_features
print(f"✅ CatBoost特徴量: {len(catboost_features)}列 (新規{len(new_gca_features)}列追加)")

# カテゴリカル変数は変更なし（新規特徴量は全て数値型）
# categorical_features はそのまま使用
print(f"ℹ️  カテゴリカル特徴量: {len(categorical_features)}列 (変更なし)")
```

---

## 🔧 追加セル4: 単調性制約の更新（任意）

```python
# ============================================================
# 単調性制約の更新（任意）
# ============================================================

# 新規GCA特徴量で単調増加制約を適用するもの
new_monotone_increase = [
    "zone14_origin_pass_count",
    "zone14_preGCA_count",
    "halfspace_L_to_box_count",
    "halfspace_R_to_box_count",
    "cutback_count",
    "linebreak_third_transition_count",
    "through_channel_pass_count",
    "one_two_chain_count",
    "third_man_release_count",
    "box_entry_from_zone14_count",
    "box_entry_from_halfspace_L_count",
    "box_entry_from_halfspace_R_count",
    "openplay_GCA_share",  # オープンプレーGCA比率
]

# 既存の単調増加特徴量リストに追加
# ※既にmonotone_increase_featuresが定義されている場合
lgbm_monotone_increase_features = lgbm_monotone_increase_features + new_monotone_increase

# LightGBM制約ベクトル再生成
monotone_constraints = [
    1 if feat in lgbm_monotone_increase_features else 0
    for feat in lgbm_features
]

print(f"✅ LightGBM単調増加制約: {sum(monotone_constraints)}個の特徴量")

# CatBoost用（任意）
catboost_monotone_increase_features = catboost_monotone_increase_features + new_monotone_increase
print(f"✅ CatBoost単調増加制約: {len(catboost_monotone_increase_features)}個の特徴量")
```

---

## ✅ 統合完了チェックリスト

以下を確認してください:

- [ ] 新規特徴量が正常に生成された（エラーなし）
- [ ] `train_df.shape[1]` が28列増加している
- [ ] `test_df.shape[1]` も28列増加している
- [ ] `lgbm_features` と `catboost_features` に新規特徴量が追加されている
- [ ] `categorical_features` に新規特徴量が**含まれていない**（全て数値型のため）
- [ ] 単調性制約を使用する場合、適切に更新されている

---

## 🎯 次のアクション

1. **Optunaハイパーパラメータ再調整**
   - 新規特徴量を含めて最適化
   - `num_leaves`, `learning_rate`, `min_child_samples` などを再探索

2. **特徴量重要度の確認**
   ```python
   # 学習後に実行
   importance_df = pd.DataFrame({
       'feature': lgbm_features,
       'importance': lgbm_model.feature_importance(importance_type='gain')
   }).sort_values('importance', ascending=False)

   new_feat_importance = importance_df[importance_df['feature'].isin(new_gca_features)]
   print(new_feat_importance.head(10))
   ```

3. **CV/OOFスコアの比較**
   - ベースライン（新規特徴量なし）との差分を確認
   - 改善がない場合は特徴量選択を検討

---

## 📚 詳細ドキュメント

詳細な説明は以下を参照:
- `experiments/exp0035/FEATURE_INTEGRATION_GUIDE.md` - 統合ガイド
- `scripts/advanced_features.py` - 実装コード
