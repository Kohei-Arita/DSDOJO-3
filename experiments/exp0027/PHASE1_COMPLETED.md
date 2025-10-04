# Phase 1 完了: StratifiedGroupKFold実装

## 実装内容

### 1. 新規スクリプト
- `scripts/cv.py`: `make_stratified_group_folds()` 関数を実装
  - `match_id` 単位でグループ化
  - 各試合の「正例率」(`xAG >= 0.1` の割合) でビン分割
  - StratifiedKFold で層化分割を実施

### 2. ノートブック修正
- `experiments/exp0027/training_with_stratified_cv.ipynb` を作成
- Cell 37: GroupKFold → StratifiedGroupKFold に置き換え
- Cell 38 (新規): 層化品質検証セルを追加
  - 各foldの正例率、サンプル数、試合数、平均xAGを表示
  - 標準偏差を計算して層化の品質を定量評価

## 期待される効果

1. **CV安定性の向上**: 各foldの正例率が均等になることで、fold間のスコア変動が減少
2. **wRMSE推定の精度向上**: 重み付き評価指標に対して、より代表性の高いCV分割
3. **ハイパーパラメータ最適化の信頼性向上**: より安定したCVスコアによる信頼性の高い選択

## 次のステップ

Phase 1が完了したので、次は以下のいずれかを実施できます:

- **実行テスト**: このノートブックを実行して、層化CVの品質を確認
- **Phase 2移行**: Isotonic校正の実装に進む
- **Phase 3移行**: 非負スタッキングの実装に進む
- **Phase 4移行**: Tweedie損失の実装に進む

## 検証方法

```bash
# Jupyter Labでノートブックを開く
jupyter lab experiments/exp0027/training_with_stratified_cv.ipynb

# または、Pythonスクリプトとして実行（変換が必要な場合）
# jupytext --to py experiments/exp0027/training_with_stratified_cv.ipynb
# python experiments/exp0027/training_with_stratified_cv.py
```

## 注意事項

- 既存のモデル学習ロジック（LGBM, CatBoost）はそのまま維持
- `train_df["fold"]` の値が変更されるため、CV結果は変わる可能性があります
- baseline (exp0026) との比較のため、CVスコアを記録することを推奨

