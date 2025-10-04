# 新特徴量実装ガイド - EXP0025

## 📋 実装済み新特徴量

### 1. 時間帯別パフォーマンス特徴量
**関数**: `build_time_based_features()`

**特徴量**:
- `first_half_actions`: 前半アクション数
- `second_half_actions`: 後半アクション数
- `final_15min_actions`: ラスト15分アクション数
- `early_10min_actions`: 序盤10分アクション数
- `time_weighted_intensity`: 時間重み付き強度(後半ほど重要)

**使用方法**:
```python
from scripts.advanced_features import build_time_based_features

time_features = build_time_based_features(
    relevant_actions,
    match_col="match_id",
    player_col="player_id",
    time_col="time_seconds",
    period_col="period_id"
)

train_df = train_df.merge(time_features, on=["match_id", "player_id"], how="left")
test_df = test_df.merge(time_features, on=["match_id", "player_id"], how="left")
```

---

### 2. ゾーン別アクション密度特徴量
**関数**: `build_zone_based_features()`

**特徴量**:
- `defensive_zone_actions`: 自陣ゾーン(x < 35)
- `middle_zone_actions`: 中盤ゾーン(35 <= x < 70)
- `attacking_zone_actions`: 敵陣ゾーン(x >= 70)
- `halfspace_left_actions`: 左ハーフスペース(y < 22.67)
- `halfspace_right_actions`: 右ハーフスペース(y > 45.33)
- `central_corridor_actions`: 中央レーン(22.67 <= y <= 45.33)
- `final_third_penetrations`: 敵陣最終ライン進入(x >= 70)
- `box_entries`: ペナルティエリア進入(x >= 88.5, 13.84 <= y <= 54.16)

**使用方法**:
```python
from scripts.advanced_features import build_zone_based_features

zone_features = build_zone_based_features(
    relevant_actions,
    match_col="match_id",
    player_col="player_id"
)

train_df = train_df.merge(zone_features, on=["match_id", "player_id"], how="left")
test_df = test_df.merge(zone_features, on=["match_id", "player_id"], how="left")
```

---

### 3. パスネットワーク中心性特徴量
**関数**: `build_pass_network_centrality()`

**特徴量**:
- `betweenness_centrality`: 媒介中心性(攻撃の中継点度)
- `closeness_centrality`: 近接中心性(攻撃への近さ)
- `degree_centrality`: 次数中心性(パス接続数)
- `pass_receiver_diversity`: パス先の多様性(エントロピー)
- `unique_pass_partners`: ユニークなパス相手数

**使用方法**:
```python
from scripts.advanced_features import build_pass_network_centrality

network_features = build_pass_network_centrality(
    relevant_actions,
    match_col="match_id",
    player_col="player_id",
    team_col="team_id",
    type_col="type_name",
    time_col="time_seconds"
)

train_df = train_df.merge(network_features, on=["match_id", "player_id"], how="left")
test_df = test_df.merge(network_features, on=["match_id", "player_id"], how="left")
```

**注意**: NetworkXライブラリが必要です。
```bash
pip install networkx
```

---

### 4. 拡張シーケンス連鎖特徴量 (5-10手先)
**関数**: `build_extended_chain_features()`

**特徴量**:
- `longchain_to_shot`: 7手先までのシュート到達割引和
- `longchain_xt_delta`: 7手先までのxT増加割引和

**使用方法**:
```python
from scripts.advanced_features import build_extended_chain_features

extended_chain = build_extended_chain_features(
    relevant_actions,
    match_col="match_id",
    player_col="player_id",
    team_col="team_id",
    type_col="type_name",
    n_steps=7,  # 7手先まで
    gamma=0.6   # 割引率
)

train_df = train_df.merge(extended_chain, on=["match_id", "player_id"], how="left")
test_df = test_df.merge(extended_chain, on=["match_id", "player_id"], how="left")
```

---

### 5. 動的ポジショニング特徴量
**関数**: `build_dynamic_positioning_features()`

**特徴量**:
- `position_variance_x`: 前後方向の活動範囲(分散)
- `position_variance_y`: 左右方向の活動範囲(分散)
- `position_range_x`: 前後方向の最大移動距離
- `position_range_y`: 左右方向の最大移動距離
- `avg_action_distance`: アクション間平均移動距離

**使用方法**:
```python
from scripts.advanced_features import build_dynamic_positioning_features

positioning_features = build_dynamic_positioning_features(
    relevant_actions,
    match_col="match_id",
    player_col="player_id"
)

train_df = train_df.merge(positioning_features, on=["match_id", "player_id"], how="left")
test_df = test_df.merge(positioning_features, on=["match_id", "player_id"], how="left")
```

---

## 🚀 一括統合の例

全ての新特徴量を一度に追加する方法:

```python
from scripts.advanced_features import (
    build_time_based_features,
    build_zone_based_features,
    build_pass_network_centrality,
    build_extended_chain_features,
    build_dynamic_positioning_features,
)

# 全特徴量を計算
print("新特徴量を計算中...")

time_feats = build_time_based_features(
    relevant_actions,
    time_col="time_seconds",
    period_col="period_id"
)

zone_feats = build_zone_based_features(relevant_actions)

network_feats = build_pass_network_centrality(
    relevant_actions,
    team_col="team_id",
    type_col="type_name",
    time_col="time_seconds"
)

extended_chain = build_extended_chain_features(
    relevant_actions,
    team_col="team_id",
    type_col="type_name",
    n_steps=7,
    gamma=0.6
)

positioning_feats = build_dynamic_positioning_features(relevant_actions)

# 統合
new_feature_blocks = [
    time_feats,
    zone_feats,
    network_feats,
    extended_chain,
    positioning_feats,
]

for block in new_feature_blocks:
    if block is not None and not block.empty:
        train_df = train_df.merge(block, on=["match_id", "player_id"], how="left")
        test_df = test_df.merge(block, on=["match_id", "player_id"], how="left")

# 欠損値処理
new_feature_cols = []
for block in new_feature_blocks:
    if block is not None and not block.empty:
        cols = [c for c in block.columns if c not in ["match_id", "player_id"]]
        new_feature_cols.extend(cols)

for col in new_feature_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(0.0)
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(0.0)

print(f"追加された新特徴量: {len(new_feature_cols)}個")
print(f"新特徴量リスト: {new_feature_cols}")
```

---

## 📊 特徴量の追加先

ノートブックの既存セクションに以下のように追加してください:

### 追加場所: `## 特徴量エンジニアリング - 応用特徴量` セクションの後

```python
## 新規追加特徴量 (EXP0025)

### 時間帯別・ゾーン別・ネットワーク特徴量
[上記の一括統合コードを挿入]
```

---

## 🎯 期待される効果

### 最優先効果 (GCA/SCAが最重要なため)
1. **パスネットワーク中心性** → GCA/SCAを深掘り、ネットワーク上の役割を捉える
2. **ゾーン別アクション密度** → 位置情報を高度活用、敵陣侵入度を評価
3. **拡張シーケンス連鎖** → nstep_to_shotの成功を7手先に拡張

### 中期的効果
4. **時間帯別パフォーマンス** → 試合の流れでの貢献度変化を捉える
5. **動的ポジショニング** → 移動パターンからプレースタイルを推定

---

## ⚠️ 注意事項

### NetworkX依存
パスネットワーク中心性はNetworkXが必要です:
```bash
pip install networkx
```

### 計算コスト
- **パスネットワーク中心性**: 試合×チーム数だけグラフ計算 → やや重い
- **その他**: 高速 (既存特徴量と同等)

### 欠損値処理
全ての新特徴量は欠損値を0.0で埋めます。アクションが少ない選手は0になります。

---

## 🔬 検証方法

特徴量重要度を確認:
```python
# 学習後
feature_importance_mean = feature_importance.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
feature_importance_mean = feature_importance_mean.sort_values('mean', ascending=False)

# 新特徴量のみ抽出
new_feature_importance = feature_importance_mean[
    feature_importance_mean['feature'].isin(new_feature_cols)
]
print("新特徴量の重要度:")
display(new_feature_importance.head(10))
```
