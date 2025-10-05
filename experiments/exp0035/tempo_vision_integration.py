"""
攻撃テンポ + 視野・認知特徴量の統合コード
exp0035ノートブックに追加するセル用
"""

from scripts.advanced_features import (
    build_attack_tempo_features,
    build_vision_cognition_features,
)

# ============================================================
# 新規特徴量の生成
# ============================================================

print("=" * 60)
print("🚀 攻撃テンポ + 視野・認知特徴量の生成開始")
print("=" * 60)

# 1. 攻撃テンポ特徴量
print("\n[1/2] 攻撃テンポ・リズム特徴量（5列）")
tempo_train = build_attack_tempo_features(
    actions=action_data_train,
    match_col="match_id",
    player_col="player_id",
    team_col="team_id",
    type_col="type_name",
    time_col="time_seconds",
)
tempo_test = build_attack_tempo_features(
    actions=action_data_test,
    match_col="match_id",
    player_col="player_id",
    team_col="team_id",
    type_col="type_name",
    time_col="time_seconds",
)
print(f"   ✅ 生成完了: {tempo_train.shape[1]-2}列 (train: {len(tempo_train)} rows)")

# 2. 視野・認知特徴量
print("\n[2/2] 視野・認知系特徴量（4列）")
vision_train = build_vision_cognition_features(
    actions=action_data_train,
    match_col="match_id",
    player_col="player_id",
    team_col="team_id",
    type_col="type_name",
)
vision_test = build_vision_cognition_features(
    actions=action_data_test,
    match_col="match_id",
    player_col="player_id",
    team_col="team_id",
    type_col="type_name",
)
print(f"   ✅ 生成完了: {vision_train.shape[1]-2}列 (train: {len(vision_train)} rows)")

print("\n" + "=" * 60)
print("✅ 全ての新規特徴量生成完了")
print("=" * 60)

# ============================================================
# 新規特徴量のマージ
# ============================================================

print("\n📊 新規特徴量をtrain_df/test_dfにマージ中...")

# train_dfに統合
train_df = train_df.merge(tempo_train, on=["match_id", "player_id"], how="left")
train_df = train_df.merge(vision_train, on=["match_id", "player_id"], how="left")

# test_dfに統合
test_df = test_df.merge(tempo_test, on=["match_id", "player_id"], how="left")
test_df = test_df.merge(vision_test, on=["match_id", "player_id"], how="left")

# 新規特徴量リストの定義
tempo_vision_features = [
    # 攻撃テンポ特徴量 (5列)
    "possession_duration_before_shot",
    "pass_tempo_variance",
    "acceleration_phase_count",
    "quick_transition_rate",
    "slow_buildup_gca_rate",

    # 視野・認知特徴量 (4列)
    "switch_play_gca",
    "blind_side_pass_count",
    "cross_field_progression",
    "vision_angle_wide_pass",
]

# 実際に存在する列のみフィルタ
tempo_vision_features = [f for f in tempo_vision_features if f in train_df.columns]

# NaN埋め（念のため）
for col in tempo_vision_features:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(0.0)
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(0.0)

print(f"\n✅ マージ完了:")
print(f"   - 新規特徴量: {len(tempo_vision_features)}列")
print(f"   - train_df shape: {train_df.shape}")
print(f"   - test_df shape: {test_df.shape}")

# 特徴量リスト表示
print(f"\n📋 新規特徴量リスト:")
for i, feat in enumerate(tempo_vision_features, 1):
    print(f"   {i:2d}. {feat}")
