# CatBoost実装 - exp0020/training_with_catboost.ipynb

## 📋 概要

`training_with_catboost.ipynb`は、元の`training.ipynb`にCatBoostモデルを追加し、LightGBMとのブレンディングによって予測精度を向上させる実装です。

## 🎯 主要な改善点

### 1. **CatBoost導入による多様性確保**
- **Ordered Target Statistics**: リークを回避しながらカテゴリ変数を効果的に処理
- **高次交互作用**: カテゴリ変数間の複雑な交互作用を自動学習
- **対称木構造**: より安定した予測を実現

### 2. **LGBMとの相補性最大化**
- 異なるseed値 (`SEED + 100`) でランダム性を変更
- 単調性制約を緩和（LGBMの21個→CatBoostの6個）
- アーキテクチャの違いによる予測の多様性

### 3. **モデルブレンディング**
- OOF予測でブレンド比率を最適化
- グリッドサーチで最適な重みを自動探索
- 3つの提出ファイルを生成（ブレンド、LGBM単独、CatBoost単独）

## 📁 ファイル構成

```
experiments/exp0020/
├── training.ipynb                          # 元のLightGBMのみの実装
├── training_with_catboost.ipynb            # CatBoost追加版（新規）
└── README_CATBOOST.md                      # このファイル
```

## 🔧 実装の詳細

### 追加されたセクション

#### **Cell 48-50: CatBoost モデル学習**
```python
# カテゴリカル特徴量の明示的指定
catboost_categorical_features = ['Comp', 'Squad', 'Venue']

# Optunaでハイパーパラメータ最適化
catboost_params = {
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
    'depth': trial.suggest_int('depth', 4, 10),
    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
    'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
    'border_count': trial.suggest_int('border_count', 32, 255),
    'random_seed': SEED + 100,  # 多様性確保
}
```

#### **Cell 51-52: CatBoost Cross Validation**
```python
# 単調性制約（LGBMより緩和）
catboost_monotone_increase_features = [
    'goal_count',
    'pass_leads_to_shot',
    'progressive_pass_count',
    'deep_completion_count',
    'penalty_area_entry_count',
    'zone_attacking_actions',
]

# サンプル重み付きで学習
train_pool = cb.Pool(
    data=X_tr,
    label=y_tr,
    weight=train_weights,  # Weighted RMSE対応
    cat_features=cat_features_idx
)
```

#### **Cell 53-58: モデルブレンディング**
```python
# OOF予測でブレンド比率を最適化
for lgb_weight in np.arange(0.0, 1.01, 0.05):
    catboost_weight = 1.0 - lgb_weight
    blended_oof = lgb_weight * oof_preds + catboost_weight * catboost_oof_preds
    blend_score = weighted_rmse(y_train, blended_oof)
    # 最小スコアを探索

# テストデータに対する推論
blended_test_preds = (
    best_blend_weight * lgbm_test_preds +
    best_catboost_weight * catboost_test_preds
)
```

## 📊 期待される効果

### モデルの相補性

| 側面 | LightGBM | CatBoost |
|------|----------|----------|
| **アーキテクチャ** | Leaf-wise成長 | Symmetric tree |
| **カテゴリ処理** | One-hot/Target Encoding | Ordered Target Statistics |
| **単調性制約** | 21個の特徴量 | 6個の特徴量（緩和） |
| **ランダムシード** | 42 | 142 |
| **特徴** | 速度重視、細かい分割 | 安定性重視、高次交互作用 |

### パフォーマンス向上の仕組み

1. **アーキテクチャの違い**
   - LightGBM: Leaf-wise成長で局所的な最適化
   - CatBoost: Symmetric treeで全体的な安定性

2. **カテゴリ処理の違い**
   - LightGBM: OOF Target Encoding（手動実装）
   - CatBoost: Ordered Target Statistics（自動・ビルトイン）

3. **制約の違い**
   - LightGBM: 多くの単調性制約（ドメイン知識重視）
   - CatBoost: 最小限の制約（柔軟性重視）

## 🚀 実行方法

### 1. 環境セットアップ
```bash
pip install catboost optuna lightgbm pandas numpy matplotlib seaborn
```

### 2. Notebook実行
```bash
jupyter notebook training_with_catboost.ipynb
```

または、Colabで実行:
- Cell 3で自動的に`catboost`がインストールされます

### 3. 実行の流れ

```
データ読み込み
↓
特徴量エンジニアリング（基本・応用）
↓
LightGBM学習（Optuna最適化 + 5-Fold CV）
↓
CatBoost学習（Optuna最適化 + 5-Fold CV）  ← 追加
↓
ブレンディング比率の最適化                ← 追加
↓
テストデータ推論（ブレンドモデル）         ← 追加
↓
提出ファイル作成（3種類）                 ← 更新
```

## 📈 出力ファイル

実行後、以下のファイルが`logs/`ディレクトリに生成されます:

1. **submission_blend_lgbm_catboost.csv** ← **推奨**
   - LightGBMとCatBoostの最適ブレンド
   - 最も高い汎化性能が期待される

2. **submission_lgbm_only.csv**
   - LightGBM単独の予測
   - ベースライン比較用

3. **submission_catboost_only.csv**
   - CatBoost単独の予測
   - アーキテクチャ比較用

## 🔍 デバッグ・検証

### モデル間相関の確認
```python
correlation = np.corrcoef(lgbm_test_preds, catboost_test_preds)[0, 1]
print(f"LGBMとCatBoostの予測相関: {correlation:.4f}")
```

**期待値:**
- 相関 < 0.95: ブレンディングによる改善効果が大きい
- 相関 > 0.98: モデルが似すぎている（多様性不足）

### OOFスコアの比較
```python
print(f"LightGBM OOF: {oof_score:.4f}")
print(f"CatBoost OOF: {catboost_oof_score:.4f}")
print(f"ブレンドOOF: {best_blend_score:.4f}")
```

**期待される改善:**
- ブレンドOOF < min(LightGBM OOF, CatBoost OOF)

## ⚠️ 注意事項

### 1. 計算時間
- CatBoostの追加により、全体の実行時間が約1.5-2倍に増加
- Optuna最適化: LightGBM（30試行）+ CatBoost（30試行）

### 2. メモリ使用量
- 2つのモデルセット（各5 folds）を保持
- 推定メモリ増加: +2-3GB

### 3. 再現性
- CatBoostは`random_seed`を`SEED + 100`に設定
- 完全な再現にはseedの固定が必要

## 📚 参考資料

### CatBoost公式ドキュメント
- [CatBoost公式サイト](https://catboost.ai/)
- [Ordered Target Statistics論文](https://arxiv.org/abs/1706.09516)

### Kaggleでの活用事例
- [CatBoost Tutorial](https://www.kaggle.com/code/alexisbcook/categorical-variables)
- [Model Blending Best Practices](https://mlwave.com/kaggle-ensembling-guide/)

## 🎓 技術的詳細

### Ordered Target Statistics とは

CatBoostの特徴的なカテゴリ処理手法:

1. **従来のTarget Encoding の問題**
   ```python
   # リークのリスク
   category_mean = df.groupby('category')['target'].mean()
   df['encoded'] = df['category'].map(category_mean)
   # → 同じカテゴリの全サンプルが同じ値に
   ```

2. **Ordered Target Statistics**
   ```python
   # 時間順序を考慮（疑似コード）
   for i in range(n):
       # i番目のサンプルより前のデータのみを使用
       prior_data = df.iloc[:i]
       category_mean = prior_data[prior_data['category'] == df.iloc[i]['category']]['target'].mean()
       df.iloc[i, 'encoded'] = category_mean
   ```

   **利点:**
   - 未来の情報を使わない（リーク回避）
   - カテゴリごとの統計を効果的に学習
   - 高次交互作用の自動検出

### 単調性制約の緩和理由

```python
# LightGBM: 21個の制約
lgbm_monotone_features = [
    'progressive_pass_count', 'progressive_pass_success',
    'progressive_pass_distance_total', 'progressive_pass_distance_mean',
    'progressive_carry_count', 'progressive_carry_success',
    # ... (合計21個)
]

# CatBoost: 6個の制約（緩和）
catboost_monotone_features = [
    'goal_count', 'pass_leads_to_shot',
    'progressive_pass_count', 'deep_completion_count',
    'penalty_area_entry_count', 'zone_attacking_actions',
]
```

**緩和の理由:**
- モデル間の多様性を確保
- CatBoostの柔軟性を活かす
- 過度な制約によるアンダーフィットを回避

## 🤝 貢献

改善提案やバグ報告は歓迎です。

## 📝 変更履歴

- **2025-10-03**: CatBoost実装を追加
  - Optuna最適化
  - 5-Fold Cross Validation
  - モデルブレンディング
  - 3種類の提出ファイル生成

---

**作成者**: Arita Kohei
**日付**: 2025-10-03
**ベース**: experiments/exp0020/training.ipynb
