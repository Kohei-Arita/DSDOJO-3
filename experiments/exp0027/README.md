# EXP0027: ゼロインフレ対応と層化CV・ポスト校正の統合

## 実験目的

exp0026（LGBM+CatBoostブレンド）をベースに、ゼロインフレ構造と重み付きRMSEの不安定性に対処する5つの手法を段階的に検証：

1. **StratifiedGroupKFold**: 試合単位で正例率を層化し、fold間のwRMSE揺れを抑制
2. **正領域Isotonic校正**: 負領域を歪めず、正領域のみ単調回帰で校正
3. **非負スタッキング**: NNLS制約でゼロ域での過剰相殺を防止
4. **Tweedie損失**: ゼロインフレに適した損失関数での学習
5. **平滑化重み**: 境界付近の学習不安定性を緩和

## 実装モジュール

### `scripts/cv.py`
- `make_stratified_group_folds()`: StratifiedGroupKFoldでfold割当
- `evaluate_fold_balance()`: fold別の正例率・統計を評価

### `scripts/calibration.py`
- `fit_isotonic_positive()`: 正領域のみIsotonic学習
- `apply_isotonic_positive()`: 閾値以上のみ校正適用
- `optimize_isotonic_threshold()`: 閾値の最適化（optional）

### `scripts/stacking.py`
- `fit_nnls_stack()`: 非負制約の線形スタッキング
- `predict_nnls_stack()`: 係数を使った予測
- `create_stack_features()`: 基礎特徴＋ReLU派生特徴の生成

### `scripts/weights.py`
- `make_sample_weight()`: 'step'/'smooth' の重み生成
- `evaluate_weight_statistics()`: 重み分布の統計
- `compare_weight_schemes()`: 複数τでの比較

## 実験手順

### Phase 1: 層化CV導入（exp0027a）
- GroupKFold → StratifiedGroupKFold に置き換え
- fold別の正例率・wRMSEを比較
- baseline（exp0026）との差分を記録

### Phase 2: Isotonic校正（exp0027b）
- ブレンド後の予測に正領域Isotonicを適用
- OOF wRMSEの改善を評価
- 閾値固定（0.1）での検証

### Phase 3: 非負スタッキング（exp0027c）
- LGBM/CatBoostのOOFを `[raw, relu(raw-0.1)]` で拡張
- NNLS係数を学習・テスト適用
- 既存グリッドブレンドとの比較

### Phase 4: Tweedie損失（exp0027d）
- LGBMの `objective='tweedie'` で学習
- `tweedie_variance_power ∈ [1.1, 1.9]` を探索
- OOF wRMSEを従来手法と比較

### Phase 5: 平滑化重み（exp0027e、optional）
- `scheme='smooth'`, `tau ∈ [0.02, 0.03, 0.05]` でCV
- fold間の安定性と閾値近傍の誤差分布を確認

## 評価指標

すべての実験で以下を記録：
- **CV wRMSE**: 平均・標準偏差・各fold
- **OOF wRMSE**: 全体
- **fold別正例率**: 層化の効果確認
- **実行時間**: 訓練時間の記録
- **パラメータ**: 最適化結果の保存

## ログ・成果物

### `logs/`
- `exp0027{a-e}_metrics.json`: CV/OOF結果・パラメータ
- `exp0027{a-e}_fold_balance.csv`: fold別統計
- `exp0027{a-e}_training.log`: 実行ログ

### `submissions/`
- `submission_exp0027{a-e}.csv`: 各手法の提出ファイル
- `submission_exp0027_final.csv`: 最良手法の最終提出

### `artifacts/`
- `isotonic_model.pkl`: 学習済みIsotonicモデル
- `nnls_coefs.json`: スタッキング係数
- `feature_importance_*.csv`: 各手法の特徴量重要度

## ベースライン比較

| 実験 | CV wRMSE | OOF wRMSE | 手法 |
|------|----------|-----------|------|
| exp0026 | 0.2303±0.0134 | 0.2295 | LGBM+CB グリッドブレンド |
| exp0027a | TBD | TBD | + StratifiedGroupKFold |
| exp0027b | TBD | TBD | + Isotonic校正 |
| exp0027c | TBD | TBD | + NNLSスタッキング |
| exp0027d | TBD | TBD | + Tweedie損失 |
| exp0027e | TBD | TBD | + 平滑化重み |

## 注意事項

- **乱数種の統一**: 全実験で `SEED=42`（CatBoostは +100）
- **fold一貫性**: 層化CVで生成したfoldを全手法で共通利用
- **リーク防止**: Isotonic/NNLSはOOFのみで学習
- **閾値固定**: 評価指標の閾値（0.1）は固定、校正閾値のみ探索可能

## 次のステップ

1. 各Phase完了後、結果を本READMEに追記
2. 有効な手法を積み上げて最終モデルを構築
3. 最良の組み合わせで提出ファイルを生成
4. 必要に応じCatBoostにもTweedieを適用

