"""
新規GCA特徴量の統合コード
exp0035ノートブックに追加するセル用
"""

from scripts.advanced_features import (
    build_gca_spatial_features,
    build_linebreak_packing_features,
    build_pass_chain_quality_features,
    build_box_entry_receiving_features,
    build_setplay_bodypart_features,
)

# ==========================================
# 新規GCA特徴量の生成
# ==========================================

print("🏗️ GCA空間特徴量を生成中...")
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
print(f"✅ GCA空間特徴量: {gca_spatial_train.shape[1]-2}列")

print("🏗️ ラインブレイク/パッキング特徴量を生成中...")
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
print(f"✅ ラインブレイク/パッキング特徴量: {linebreak_train.shape[1]-2}列")

print("🏗️ パス連鎖品質特徴量を生成中...")
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
print(f"✅ パス連鎖品質特徴量: {pass_chain_train.shape[1]-2}列")

print("🏗️ PA進入受け手文脈特徴量を生成中...")
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
print(f"✅ PA進入受け手文脈特徴量: {box_entry_train.shape[1]-2}列")

print("🏗️ セットプレー/ボディパート特徴量を生成中...")
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
print(f"✅ セットプレー/ボディパート特徴量: {setplay_bp_train.shape[1]-2}列")

# ==========================================
# 既存データフレームへのマージ
# ==========================================

print("\n📊 新規特徴量をマージ中...")

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

# NaN埋め
new_feature_cols = (
    list(gca_spatial_train.columns) +
    list(linebreak_train.columns) +
    list(pass_chain_train.columns) +
    list(box_entry_train.columns) +
    list(setplay_bp_train.columns)
)
new_feature_cols = [c for c in new_feature_cols if c not in ["match_id", "player_id"]]

for col in new_feature_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(0.0)
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(0.0)

print(f"\n✅ 新規特徴量統合完了: {len(new_feature_cols)}列追加")
print(f"📈 train_df shape: {train_df.shape}")
print(f"📈 test_df shape: {test_df.shape}")

# ==========================================
# 新規特徴量リストの記録
# ==========================================

new_gca_features = [
    # GCA空間特徴量
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

    # ラインブレイク/パッキング
    "linebreak_third_transition_count",
    "linebreak_third_transition_rate",
    "through_channel_pass_count",
    "through_channel_pass_rate",
    "packing_approx_score_mean",

    # パス連鎖品質
    "one_two_chain_count",
    "third_man_release_count",
    "burst_window_SCA_rate",

    # PA進入受け手文脈
    "box_entry_from_zone14_count",
    "box_entry_from_halfspace_L_count",
    "box_entry_from_halfspace_R_count",
    "facing_forward_share_in_box",
    "first_touch_shot_rate_in_box",

    # セットプレー/ボディパート
    "setplay_GCA_share",
    "openplay_GCA_share",
    "bodypart_on_key_pass_rate_right",
    "bodypart_on_key_pass_rate_left",
    "bodypart_on_key_pass_rate_head",
]

# 実際に存在する列のみフィルタ
new_gca_features = [f for f in new_gca_features if f in train_df.columns]

print(f"\n📋 新規GCA特徴量リスト ({len(new_gca_features)}個):")
for feat in new_gca_features:
    print(f"  - {feat}")

# 既存の特徴量リストに追加
# ※ノートブック内の all_features または lgbm_features に追加してください
# 例: all_features.extend(new_gca_features)
