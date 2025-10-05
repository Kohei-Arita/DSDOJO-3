# exp0035ノートブックへの追加手順（攻撃テンポ+視野・認知）

## 📍 追加位置

**既存の新規GCA特徴量セクション（セル43-44）の後**に以下のセルを追加してください。

---

## 🔧 追加セル1: 攻撃テンポ + 視野・認知特徴量のインポートと生成

```python
# ============================================================
# 攻撃テンポ + 視野・認知特徴量の追加 (exp0035拡張 Phase 2)
# ============================================================

from scripts.advanced_features import (
    build_attack_tempo_features,
    build_vision_cognition_features,
)

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
```

---

## 🔧 追加セル2: train_df/test_dfへのマージ

```python
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
```

---

## 🔧 追加セル3: 特徴量リストへの追加

```python
# ============================================================
# LightGBM/CatBoost特徴量リストに新規特徴量を追加
# ============================================================

# all_features に追加（既存のnew_gca_featuresの後）
all_features = all_features + tempo_vision_features
print(f"✅ 特徴量合計: {len(all_features)}列 (攻撃テンポ+視野・認知{len(tempo_vision_features)}列追加)")

# カテゴリカル変数は変更なし（新規特徴量は全て数値型）
print(f"ℹ️  カテゴリカル特徴量: {len(categorical_features)}列 (変更なし)")
```

---

## 🔧 追加セル4: 単調性制約の更新（任意）

```python
# ============================================================
# 単調性制約の更新（任意）
# ============================================================

# 新規特徴量で単調増加制約を適用するもの
new_monotone_increase = [
    "acceleration_phase_count",    # 加速フェーズは多いほど良い
    "switch_play_gca",             # サイドチェンジGCAは多いほど良い
    "blind_side_pass_count",       # ブラインドサイドパスは多いほど良い
    "cross_field_progression",     # 対角線的前進は多いほど良い
    "vision_angle_wide_pass",      # 広角視野パスは多いほど良い
]

# 既存の単調増加特徴量リストに追加（LightGBM）
if 'lgbm_monotone_increase_features' in globals():
    lgbm_monotone_increase_features = lgbm_monotone_increase_features + new_monotone_increase

    # LightGBM制約ベクトル再生成
    if 'lgbm_features' in globals():
        monotone_constraints = [
            1 if feat in lgbm_monotone_increase_features else 0
            for feat in lgbm_features
        ]
        print(f"✅ LightGBM単調増加制約: {sum(monotone_constraints)}個の特徴量")

# CatBoost用（任意）
if 'catboost_monotone_increase_features' in globals():
    catboost_monotone_increase_features = catboost_monotone_increase_features + new_monotone_increase
    print(f"✅ CatBoost単調増加制約: {len(catboost_monotone_increase_features)}個の特徴量")
```

---

## ✅ 統合完了チェックリスト

- [ ] 新規特徴量が正常に生成された（エラーなし）
- [ ] `train_df.shape[1]` が9列増加している
- [ ] `test_df.shape[1]` も9列増加している
- [ ] `all_features` に新規特徴量が追加されている
- [ ] `categorical_features` に新規特徴量が**含まれていない**（全て数値型のため）
- [ ] 単調性制約を使用する場合、適切に更新されている
- [ ] カーネルクラッシュが発生しない（メモリ効率化済み）

---

## 📊 実装済み特徴量一覧 (9列)

### 攻撃テンポ特徴量 (5列)
```
possession_duration_before_shot  # シュート前ポゼッション時間平均
pass_tempo_variance              # パステンポ分散（予測不可能性）
acceleration_phase_count         # 加速フェーズ回数
quick_transition_rate            # 5秒以内速攻率
slow_buildup_gca_rate           # 15秒以上ビルドアップGCA率
```

### 視野・認知特徴量 (4列)
```
switch_play_gca                  # サイドチェンジ(40m+)からのGCA
blind_side_pass_count            # ブラインドサイドパス
cross_field_progression          # 対角線的前進
vision_angle_wide_pass           # 広角視野パス(120度+)
```

---

## 🎯 期待される効果

### 攻撃テンポ特徴量
- 速攻 vs ポゼッション攻撃の判別
- テンポ変化による守備崩しの評価
- カウンター能力の明示化

### 視野・認知特徴量
- 戦術的知性の評価
- 視野の広さ・創造性の捕捉
- 高度なパス選択能力の明示化

---

## 📚 詳細ドキュメント

- `experiments/exp0035/DATA_LEAK_VERIFICATION.md` - リーク検証レポート
- `experiments/exp0035/tempo_vision_integration.py` - 統合コード
- `scripts/advanced_features.py` - 実装コード（行1226-1470）
