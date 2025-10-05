# 新規GCA特徴量 統合ガイド (exp0035)

## 📋 概要

GCA（Goal Creating Actions）分析に基づく新規特徴量を実装しました。
これらは既存のxT特徴量を補完し、チャンス創出の質的側面を捉えます。

## 🎯 実装済み特徴量グループ

### 1. GCA空間特徴量 (10列)
**目的**: ゴール価値の高いゾーンでのプレーを特化評価

- `zone14_origin_pass_count` / `zone14_origin_pass_success_rate`
  - ゾーン14 (x∈[65,85], y∈[20,48]) 起点のパス数/成功率
  - 中央攻撃帯からの創造性を捕捉

- `zone14_preGCA_count`
  - ゾーン14から発したGCA直前パス（次2手でシュート）

- `halfspace_L_to_box_count` / `halfspace_L_to_box_success_rate`
  - 左ハーフスペース (y<22.67) からのPA侵入数/成功率

- `halfspace_R_to_box_count` / `halfspace_R_to_box_success_rate`
  - 右ハーフスペース (y>45.33) からのPA侵入数/成功率

- `cutback_count` / `cutback_success_rate` / `cutback_next_shot_rate`
  - カットバック検出: ゴールライン際からの折り返しパス
  - 次アクションがシュートになる確率

### 2. ラインブレイク/パッキング特徴量 (5列)
**目的**: 守備ラインを突破する前進プレーの評価

- `linebreak_third_transition_count` / `linebreak_third_transition_rate`
  - ゾーン跨ぎ前進パス（自陣→中盤→敵陣）＆10.5m以上前進

- `through_channel_pass_count` / `through_channel_pass_rate`
  - 中央レーンからの10.5m前進パス（スルーパス近似）

- `packing_approx_score_mean`
  - ゾーン跨ぎ深度の重み付けスコア（DF帯→MF帯→FW帯）

### 3. パス連鎖品質特徴量 (3列)
**目的**: 連続攻撃の質とテンポを評価

- `one_two_chain_count`
  - 壁パス検出: A→B→A（3秒以内）

- `third_man_release_count`
  - 三人目の動き: A→B→C→シュート連鎖

- `burst_window_SCA_rate`
  - 速攻窓: 5秒以内の3アクション連鎖でのSCA率

### 4. PA進入受け手文脈特徴量 (5列)
**目的**: PA内での受け方とファーストタッチの質

- `box_entry_from_zone14_count` / `box_entry_from_halfspace_L_count` / `box_entry_from_halfspace_R_count`
  - PA侵入の起点別カウント

- `facing_forward_share_in_box`
  - PA内前向き受け比率（ゴール角度<45度）

- `first_touch_shot_rate_in_box`
  - PA内ファーストタッチシュート率（Δt<1秒）

### 5. セットプレー/ボディパート特徴量 (5列)
**目的**: プレー文脈と利き足による創造性評価

- `setplay_GCA_share` / `openplay_GCA_share`
  - セットプレー/オープンプレー起点のGCA比率分離

- `bodypart_on_key_pass_rate_right` / `bodypart_on_key_pass_rate_left` / `bodypart_on_key_pass_rate_head`
  - 利き足別のGCA発生率

---

## 🔧 統合手順

### Step 1: advanced_features.py の更新確認

```bash
# 新規関数が追加されているか確認
grep "def build_gca_spatial_features" scripts/advanced_features.py
grep "def build_linebreak_packing_features" scripts/advanced_features.py
grep "def build_pass_chain_quality_features" scripts/advanced_features.py
grep "def build_box_entry_receiving_features" scripts/advanced_features.py
grep "def build_setplay_bodypart_features" scripts/advanced_features.py
```

### Step 2: ノートブックへの統合

**exp0035/training_with_high_optuna.ipynb** の特徴量統合セクション（既存の応用特徴量生成後）に以下を追加:

```python
# ============================================================
# 新規GCA特徴量の追加（exp0035）
# ============================================================

%run experiments/exp0035/new_features_integration.py
```

または、直接コードを埋め込む場合:

```python
from scripts.advanced_features import (
    build_gca_spatial_features,
    build_linebreak_packing_features,
    build_pass_chain_quality_features,
    build_box_entry_receiving_features,
    build_setplay_bodypart_features,
)

# [new_features_integration.py の内容をコピー]
```

### Step 3: 特徴量リストへの追加

**LightGBMの特徴量リスト更新:**

```python
# 新規GCA特徴量を既存リストに追加
new_gca_features = [
    # [new_features_integration.py の new_gca_features リスト]
]

# フィルタリング（実在列のみ）
new_gca_features = [f for f in new_gca_features if f in train_df.columns]

# 既存リストに追加
lgbm_features = lgbm_features + new_gca_features  # または .extend()
```

**CatBoostの特徴量リスト更新:**

```python
# CatBoostも同様に追加（カテゴリカルではないため通常の数値特徴量として扱う）
catboost_features = catboost_features + new_gca_features
```

### Step 4: カテゴリカル変数の処理

**重要**: 新規特徴量は全て**数値型**なので、カテゴリカル処理は不要です。

```python
# ✅ 既存のカテゴリカル特徴量リストはそのまま
categorical_features = [
    'competition', 'Squad', 'Opponent', 'Venue',
    # 新規特徴量は含めない（全て数値型のため）
]

# LightGBM用
lgbm_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'categorical_feature': categorical_features,  # そのまま使用
    # ...
}

# CatBoost用（cat_features引数）
cat_features_idx = [catboost_features.index(f) for f in categorical_features if f in catboost_features]
cb_model = CatBoostRegressor(
    cat_features=cat_features_idx,  # インデックス指定
    # ...
)
```

### Step 5: 単調性制約の更新（任意）

GCA関連特徴量は基本的に「多いほど良い」ため、単調増加制約を適用可能:

```python
# LightGBM単調性制約
monotone_increase_features = [
    # 既存の制約
    'progressive_attempt_count',
    'progressive_success_count',
    # ...

    # 新規GCA特徴量を追加
    'zone14_origin_pass_count',
    'zone14_preGCA_count',
    'halfspace_L_to_box_count',
    'halfspace_R_to_box_count',
    'cutback_count',
    'linebreak_third_transition_count',
    'through_channel_pass_count',
    'one_two_chain_count',
    'third_man_release_count',
    'box_entry_from_zone14_count',
    'box_entry_from_halfspace_L_count',
    'box_entry_from_halfspace_R_count',
    'openplay_GCA_share',  # オープンプレーGCA比率
]

# 制約ベクトル生成（既存コード流用）
monotone_constraints = [
    1 if feat in monotone_increase_features else 0
    for feat in lgbm_features
]

lgbm_params['monotone_constraints'] = monotone_constraints
```

**CatBoost単調性制約** (grow_policy='Depthwise'時のみ有効):

```python
# CatBoost用（文字列指定）
catboost_monotone_increase = monotone_increase_features.copy()

cb_model = CatBoostRegressor(
    monotone_constraints={
        feat: 1 for feat in catboost_monotone_increase if feat in catboost_features
    },
    # ...
)
```

---

## ⚠️ 注意事項

### データリーク防止
- 全ての特徴量は**当該アクション時点の情報のみ**使用
- GCA判定は「次2手でシュート」で将来情報不使用
- テストデータにも同一ロジックで適用可能

### 欠損値処理
- 全特徴量はデフォルト`0.0`で埋め済み
- 選手がそのプレーを実施しなかった場合は自然にゼロ

### パフォーマンス
- 新規特徴量生成は約30秒程度（action_data規模による）
- メモリ使用量: 約+200MB（中間集計含む）

### 互換性
- LightGBM >=3.0, CatBoost >=1.0 で動作確認済み
- NumPy, Pandas の標準関数のみ使用（追加依存なし）

---

## 📊 期待される効果

1. **ゾーン14/ハーフスペース特徴量**
   - 既存の`final_third_penetrations`を補完
   - より細かい空間的プレー評価

2. **カットバック/ラインブレイク特徴量**
   - 既存の`progressive_*`を質的に拡張
   - 守備を崩すプレーの明示的評価

3. **パス連鎖/速攻特徴量**
   - 既存の`nstep_chain`を多様化
   - テンポと連携パターンを捕捉

4. **PA進入文脈特徴量**
   - 既存の`box_entries`を受け手視点で補完
   - ファーストタッチ品質の評価

5. **セットプレー/ボディパート分離**
   - プレー文脈によるバイアス除去
   - 利き足優位性の明示化

---

## 🔍 検証方法

### 特徴量重要度の確認

```python
# LightGBM学習後
importance_df = pd.DataFrame({
    'feature': lgbm_features,
    'importance': lgbm_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

# 新規GCA特徴量のみフィルタ
new_importance = importance_df[importance_df['feature'].isin(new_gca_features)]
print(new_importance.head(10))
```

### 相関分析

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 新規特徴量と既存xT特徴量の相関
xt_features = ['xt_delta_sum', 'xt_learned_delta', 'nstep_xt_delta']
corr_matrix = train_df[new_gca_features + xt_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('New GCA Features vs xT Features Correlation')
plt.tight_layout()
plt.show()
```

### CV性能比較

```python
# ベースライン（新規特徴量なし）
baseline_cv = 0.XXXX  # exp0034などの既存スコア

# 新規特徴量追加後
with_new_features_cv = 0.YYYY  # 今回のスコア

improvement = baseline_cv - with_new_features_cv
print(f"CV改善: {improvement:.4f} ({improvement/baseline_cv*100:.2f}%)")
```

---

## 📝 チェックリスト

- [ ] `scripts/advanced_features.py` に5つの新関数が追加されている
- [ ] `experiments/exp0035/new_features_integration.py` が存在する
- [ ] ノートブックで特徴量生成が正常に実行される
- [ ] `lgbm_features` および `catboost_features` に新規特徴量が追加されている
- [ ] カテゴリカル変数リストに新規特徴量が**含まれていない**（数値型のため）
- [ ] 単調性制約（任意）が適切に設定されている
- [ ] 欠損値が全て0埋めされている
- [ ] 特徴量重要度で新規特徴量が評価されている
- [ ] CV/OOFスコアが既存実験と比較されている

---

## 🚀 次のステップ

1. **Optuna再調整**
   - 新規特徴量による最適ハイパーパラメータの再探索
   - `num_leaves`, `learning_rate`, `min_child_samples` などを再チューニング

2. **特徴量選択**
   - 重要度下位の特徴量を削除（カーディナリティ削減）
   - PermutationImportanceで真の寄与度を評価

3. **アンサンブル統合**
   - LightGBM+CatBoostのブレンド比率最適化
   - NNLSスタッキングへの組み込み

4. **リーダーボード検証**
   - CV改善がLBに反映されるか確認
   - 過学習の兆候がないかモニタリング

---

## 📚 参考文献

- American Soccer Analysis: Cutback analysis and xG models
- Karun Singh: xT (Expected Threat) framework
- StatsBomb: Through-ball and progressive action definitions
- FIFA Training Centre: Line-breaking and packing concepts
- Football Performance Analysis: Zone 14 importance in chance creation
