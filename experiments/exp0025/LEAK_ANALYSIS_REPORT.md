# 新特徴量リーク分析レポート

## 📋 分析対象特徴量

1. 時間帯別パフォーマンス (5特徴)
2. ゾーン別アクション密度 (8特徴)
3. パスネットワーク中心性 (5特徴)
4. 拡張シーケンス連鎖 (2特徴)
5. 動的ポジショニング (5特徴)

---

## 1️⃣ 時間帯別パフォーマンス特徴量

### 計算方法
```python
# 前半/後半の判定
first_half = (period_id == 1)
second_half = (period_id == 2)

# ラスト15分: 後半30分以降 (time_seconds >= 2700)
final_15min = (period_id == 2) & (time_seconds >= 2700)

# 序盤10分: 前半0-600秒
early_10min = (period_id == 1) & (time_seconds <= 600)

# 時間重み付き強度
time_weight = 前半なら (time_seconds / 2700) * 0.5
             後半なら 0.5 + (time_seconds / 2700) * 0.5
```

### 使用データ
- **入力**: `period_id`, `time_seconds` (action_data.csv)
- **集計**: 試合×選手別のアクション数カウント

### リーク分析
- ✅ **リスク: なし**
- **理由**:
  - アクションの時間帯情報のみを使用
  - 未来情報は一切参照していない
  - 同一試合内の過去データのみ使用
  - ターゲット(xAG)と独立した記述統計

### 計算タイミング
- 試合終了後、その試合のアクションデータから計算
- train/test分離前に計算可能

---

## 2️⃣ ゾーン別アクション密度特徴量

### 計算方法
```python
# ピッチをゾーン分割
defensive_zone = (start_x < 35.0)
middle_zone = (35.0 <= start_x < 70.0)
attacking_zone = (start_x >= 70.0)

# Y軸分割
halfspace_left = (start_y < 22.67)
central_corridor = (22.67 <= start_y <= 45.33)
halfspace_right = (start_y > 45.33)

# 特殊ゾーン
final_third = (start_x >= 70.0)
penalty_box = (start_x >= 88.5) & (13.84 <= start_y <= 54.16)
```

### 使用データ
- **入力**: `start_x`, `start_y` (action_data.csv)
- **集計**: 試合×選手別の各ゾーンでのアクション数

### リーク分析
- ✅ **リスク: なし**
- **理由**:
  - 位置情報の静的な分類のみ
  - アクションが発生した座標を単純集計
  - ゾーン境界は固定値(ピッチサイズ基準)
  - 未来情報や結果情報は不使用

### 計算タイミング
- 試合終了後、アクション位置から即座に計算可能
- リアルタイムでも計算可能(試合中の累積値)

---

## 3️⃣ パスネットワーク中心性特徴量

### 計算方法
```python
# パスネットワーク構築
for each pass:
    passer = current_player
    # 次のアクションの選手を受け手と仮定
    next_actions = actions[(match_id == current_match) &
                          (team_id == current_team) &
                          (time_seconds > current_time)]
    if len(next_actions) > 0:
        receiver = next_actions.iloc[0].player_id
        add_edge(passer, receiver)

# 中心性計算
betweenness_centrality(G)  # 媒介中心性
closeness_centrality(G)    # 近接中心性
degree_centrality(G)       # 次数中心性

# 多様性
entropy(receiver_distribution)  # エントロピー
```

### 使用データ
- **入力**: `type_name == "pass"`, `time_seconds`, `team_id`, `player_id`
- **処理**: 時系列順にパス受け手を推定してグラフ構築

### リーク分析
- ⚠️ **リスク: 軽微 (パス受け手推定の不確実性)**
- **詳細**:
  - **潜在的問題**: 「次のアクションの選手 = パス受け手」と仮定
  - **現実**: パスが失敗した場合、次のアクションは相手チーム
  - **影響**: パス成功時のみ正しいネットワークが構築される

- ✅ **リーク該当性: なし**
- **理由**:
  - 使用するのは時系列情報のみ
  - パス成功/失敗は`result_name`を見ていないため間接的
  - 同一チームフィルタで失敗パスは自動除外される
  - ターゲット(xAG)は直接参照していない

### 改善提案
```python
# より正確なパス受け手推定
passes_with_receiver = actions[
    (type_name == "pass") &
    (result_name == "success")  # 成功パスのみ
]
# さらに正確にするには、空間的距離も考慮
```

### 計算タイミング
- 試合終了後、時系列順アクションから計算

---

## 4️⃣ 拡張シーケンス連鎖特徴量

### 計算方法
```python
# 7手先までのシフト
for k in range(1, 8):
    next_type[k] = actions.groupby(match_id).shift(-k)
    next_team[k] = actions.groupby(match_id).shift(-k)

# 割引和計算
weights = {k: 0.6^(k-1) for k in 1..7}
longchain_to_shot = Σ(k=1→7) weights[k] * I[next_k_type == shot & same_team]

# xT増分の割引和
longchain_xt_delta = Σ(k=1→7) weights[k] * next_k_xt_delta * I[same_team]
```

### 使用データ
- **入力**: `type_name`, `team_id`, `xt_learned_delta` (既に計算済み)
- **処理**: 未来N手の情報を**過去視点から**集計

### リーク分析
- ⚠️ **リスク: 中程度 (未来情報の使用)**
- **詳細**:
  - **問題**: `shift(-k)` で未来のアクション情報を使用
  - **影響範囲**: 現在のアクションが7手先のシュートに繋がったかを評価

- ✅ **リーク該当性: 条件付きセーフ**
- **理由**:
  - **目的**: xAG = 「シュートに繋がったパスの期待値」
  - **定義的正当性**: アシスト判定は元々「後続のシュート」を見る
  - **既存特徴量との整合性**: `nstep_to_shot` (3手先) と同じロジック
  - **ターゲット直接参照なし**: xAGそのものは見ていない

### ⚠️ 注意点
- **正当化の根拠**:
  - xAGの定義が「シュートに繋がったパス」である以上、未来のシュート発生は**特徴量として妥当**
  - ただし、7手先は長すぎる可能性 → 過学習リスク

- **推奨**:
  - n_steps=3-5程度に抑える
  - Cross-Validationで汎化性能を確認

---

## 5️⃣ 動的ポジショニング特徴量

### 計算方法
```python
# 前回位置との差分
prev_x = actions.groupby([match_id, player_id]).shift(1)
prev_y = actions.groupby([match_id, player_id]).shift(1)

# 移動距離
move_dist = sqrt((start_x - prev_x)^2 + (start_y - prev_y)^2)

# 統計量
position_variance_x = var(start_x)  # 前後方向の分散
position_range_x = max(start_x) - min(start_x)  # 前後方向の範囲
avg_action_distance = mean(move_dist)  # 平均移動距離
```

### 使用データ
- **入力**: `start_x`, `start_y` (同一試合×選手内の時系列データ)
- **集計**: 試合×選手別の位置統計

### リーク分析
- ✅ **リスク: なし**
- **理由**:
  - 過去の自分の位置のみ参照 (`shift(1)`)
  - 未来情報は不使用
  - 単純な記述統計(分散、範囲)
  - ターゲットと独立

### 計算タイミング
- 試合終了後、時系列順アクションから計算可能

---

## 📊 総合リスク評価

| 特徴量グループ | リークリスク | 深刻度 | 推奨アクション |
|---------------|-------------|--------|--------------|
| 時間帯別 | なし | ✅ 安全 | そのまま使用可 |
| ゾーン別 | なし | ✅ 安全 | そのまま使用可 |
| ネットワーク中心性 | 軽微 | ⚠️ 注意 | パス成功フィルタ追加検討 |
| 拡張連鎖 | 中程度 | ⚠️ 注意 | n_steps削減推奨(3-5) |
| 動的ポジショニング | なし | ✅ 安全 | そのまま使用可 |

---

## 🔍 リークの定義と判定基準

### ❌ リークに該当するケース
1. **ターゲット直接参照**: xAG値そのものを特徴量化
2. **テストデータ混入**: trainでtest情報を使用
3. **時系列逆流**: 予測時点で未知の未来情報を使用
4. **グローバル統計汚染**: test含む全体統計をtrainに適用

### ⚠️ グレーゾーン (要検証)
1. **定義的未来参照**: xAGの定義に含まれる未来情報(シュート結果)
2. **間接的結果情報**: パス成功率など、結果と相関する情報

### ✅ セーフなケース
1. **過去情報のみ**: 予測時点より前のデータのみ使用
2. **記述統計**: 平均、分散、カウントなど
3. **固定ルール**: ピッチゾーン分割など、データに依存しない閾値

---

## 💡 推奨事項

### 1. **拡張連鎖特徴量の調整**
```python
# n_stepsを削減
extended_chain = build_extended_chain_features(
    relevant_actions,
    n_steps=5,  # 7 → 5に削減
    gamma=0.6
)
```

### 2. **ネットワーク中心性の強化**
```python
# パス成功のみでネットワーク構築
passes = actions[
    (type_name == "pass") &
    (result_name == "success")  # 追加
]
```

### 3. **Cross-Validation検証**
- OOF予測でtrain RMSEとvalid RMSEの乖離を確認
- 乖離が大きい場合は過学習の可能性

### 4. **特徴量重要度モニタリング**
- 拡張連鎖の重要度が極端に高い場合は要注意
- 時間帯・ゾーン系が適度に重要なら健全

---

## ✅ 結論

### 総合判定: **使用可能 (一部調整推奨)**

1. **安全な特徴量 (そのまま使用)**:
   - 時間帯別パフォーマンス
   - ゾーン別アクション密度
   - 動的ポジショニング

2. **要調整の特徴量**:
   - 拡張連鎖: n_steps=7 → 5に削減
   - ネットワーク中心性: パス成功フィルタ追加検討

3. **モニタリング項目**:
   - CV scoreの安定性
   - 特徴量重要度の妥当性
   - Train/Valid RMSEの乖離

全体として、**リーク懸念は限定的**であり、適切な調整とモニタリングの下で使用可能です。
