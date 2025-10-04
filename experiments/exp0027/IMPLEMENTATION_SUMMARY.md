# EXP0027 実装サマリー

## 完了した準備作業

### 1. モジュール実装（`scripts/`）

#### ✅ `scripts/cv.py`
- **機能**: StratifiedGroupKFold による試合単位の層化分割
- **主要関数**:
  - `make_stratified_group_folds()`: match_id単位で正例率をビン化し、層化したfold割当
  - `evaluate_fold_balance()`: fold別の正例率・統計を集計
- **特徴**: 
  - ユニーク数が少ない場合の自動縮退
  - 量子点（quantile）ベースのビン化で偏りを回避

#### ✅ `scripts/calibration.py`
- **機能**: 正領域のみのIsotonic回帰校正
- **主要関数**:
  - `fit_isotonic_positive()`: 閾値以上の予測のみで単調回帰学習
  - `apply_isotonic_positive()`: 閾値未満は素通し、以上のみ校正
  - `optimize_isotonic_threshold()`: 閾値のグリッドサーチ最適化
- **特徴**: 
  - 負領域（<0.1）を歪めない設計
  - wRMSEと整合する重み付き学習

#### ✅ `scripts/stacking.py`
- **機能**: 非負制約の線形スタッキング
- **主要関数**:
  - `fit_nnls_stack()`: scipy.optimize.nnls による重み付き最小二乗
  - `predict_nnls_stack()`: 係数を用いた予測合成
  - `create_stack_features()`: 基礎予測 + ReLU派生特徴の生成
- **特徴**: 
  - バイアス項なし（ゼロ予測を保持）
  - 正領域強調特徴 `relu(ŷ - 0.1)` の自動生成

#### ✅ `scripts/weights.py`
- **機能**: ステップ／平滑化重みの生成
- **主要関数**:
  - `make_sample_weight()`: 'step'（段差）/'smooth'（sigmoid平滑化）
  - `evaluate_weight_statistics()`: 重み分布の統計（有効サンプル数含む）
  - `compare_weight_schemes()`: 複数τでの比較テーブル生成
- **特徴**: 
  - 評価指標（step）との単調同値性を保証
  - τ（温度）パラメータで境界の急峻さを調整

---

## 実験設計

### Phase 1: StratifiedGroupKFold（層化CV）
**目的**: fold間の正例率を揃え、wRMSEの揺れを抑制

**実装方針**:
- 既存の `GroupKFold` 生成セル（行4706付近）の直後に追加セルを挿入
- `train_df['fold_sgkf']` として別名で保存（既存foldと比較可能に）
- fold別の正例率・CV標準偏差を比較表示

**期待効果**:
- 正例率の標準偏差: 50%以上削減
- CV標準偏差: 10〜20% 削減
- OOF wRMSE: 同等〜微改善

---

### Phase 2: 正領域Isotonic校正
**目的**: 負領域を保護しつつ、正領域の予測歪みを補正

**実装方針**:
- ブレンド予測生成セル（行7200付近）の直後に追加
- OOFで校正モデルを学習 → テストに適用
- 校正前後の散布図・分布比較を可視化

**期待効果**:
- OOF wRMSE: -0.001〜-0.003 の改善
- 正領域での y=x 線への収束

---

### Phase 3: 非負スタッキング
**目的**: ゼロ域での過剰相殺を防止、非負制約で安定化

**実装方針**:
- LGBM/CatBoostのOOFを `[raw, relu(raw-0.1)]` で拡張（4特徴）
- NNLS係数を学習 → テスト予測に適用
- グリッドブレンドとの横並び比較

**期待効果**:
- 負値予測の完全排除
- OOF wRMSE: グリッドブレンドと同等以上

---

### Phase 4: Tweedie損失
**目的**: ゼロインフレ構造に適した損失での学習

**実装方針**:
- LGBMの `objective='tweedie'` で学習
- `tweedie_variance_power ∈ [1.1, 1.9]` を Optuna で探索
- 既存の RMSE ベース学習と OOF wRMSE で比較

**期待効果**:
- ゼロ近傍の予測精度向上
- OOF wRMSE: -0.002以上の改善（期待）

---

### Phase 5: 平滑化重み（optional）
**目的**: 境界付近の学習不安定性を緩和

**実装方針**:
- `scheme='smooth'`, `tau ∈ [0.02, 0.03, 0.05]` でCV
- 評価は従来の段差wRMSEで一貫評価
- fold間の標準偏差と閾値近傍の誤差分布を確認

**期待効果**:
- CV標準偏差のさらなる減少
- 閾値境界での予測安定化

---

## ノートブック編集戦略

### 最小介入の原則
- **ipynb形式の課題**: 8000行超のJSON、直接編集は困難
- **対策**: 追加セルで段階的に拡張、既存セルは原則不変
- **実装場所**: 各Phaseで論理的に適切な位置にセルを挿入

### セル追加の技術仕様
- `edit_notebook` ツールの `is_new_cell=True` を使用
- セル言語: `python` / `markdown`
- インデックス指定: 既存セルの直後に挿入

### 実験の切り替え
- 各Phaseで変数名を分離（例: `oof_preds_gkf` / `oof_preds_sgkf`）
- 後続セクションで比較表を生成
- 最終提出はベストスコアの手法を選択

---

## 評価・ログ仕様

### 記録対象メトリクス
すべてのPhaseで以下を記録：

1. **CV wRMSE**: 平均・標準偏差・各fold
2. **OOF wRMSE**: 全体スコア
3. **fold別統計**: 正例率・サンプル数・平均/標準偏差
4. **パラメータ**: 最適化結果（Optuna trial、NNLS係数等）
5. **実行時間**: 訓練時間の記録

### ログファイル構成

#### `logs/`
```
exp0027a_metrics.json       # Phase 1（層化CV）
exp0027a_fold_balance.csv   # fold別統計
exp0027a_training.log       # 実行ログ

exp0027b_metrics.json       # Phase 2（Isotonic）
exp0027b_isotonic_model.pkl # 学習済みモデル

exp0027c_metrics.json       # Phase 3（NNLS）
exp0027c_nnls_coefs.json    # スタッキング係数

exp0027d_metrics.json       # Phase 4（Tweedie）
exp0027d_optuna_study.pkl   # Optunaスタディ

exp0027e_metrics.json       # Phase 5（平滑重み）
```

#### `submissions/`
```
submission_exp0027a.csv     # 各Phaseの提出ファイル
submission_exp0027b.csv
submission_exp0027c.csv
submission_exp0027d.csv
submission_exp0027e.csv
submission_exp0027_final.csv  # 最良手法
```

---

## ベースライン（exp0026）仕様

### 既存の構成
- **モデル**: LightGBM + CatBoost
- **ブレンド**: グリッドサーチで重み探索（0.45 LGBM / 0.55 CB）
- **CV**: GroupKFold（n_splits=5, match_id単位）
- **特徴量**: 315個（xT系、ターゲットエンコーディング、進攻系等）
- **評価**: 重み付きRMSE（閾値0.1、重み5.0）

### ベースラインスコア
- **CV wRMSE**: 0.2303 ± 0.0134
- **OOF wRMSE**: 
  - LGBM単独: 0.2311
  - CatBoost単独: 0.2304
  - ブレンド: 0.2295

### 改善目標
- **Phase 1**: CV標準偏差を 0.0134 → 0.010 以下
- **Phase 2-3**: OOF wRMSE を 0.2295 → 0.226 以下（-0.003）
- **Phase 4**: さらに -0.002 の改善（Tweedie効果）
- **最終目標**: OOF wRMSE 0.224 以下

---

## 技術的注意点

### データリークの防止
- **Isotonic/NNLS**: OOFのみで学習、テストは推論のみ
- **ターゲットエンコーディング**: 既存実装を維持（fold外で計算）
- **fold一貫性**: Phase 1で生成した `fold_sgkf` を全Phaseで共通利用

### 再現性の保証
- **乱数種**: 全実験で `SEED=42`（CatBoostは +100）
- **fold固定**: `random_state=SEED` を明示
- **パラメータ保存**: JSON形式で全ハイパーパラメータを記録

### デバッグ戦略
- **段階的実行**: 各Phaseで1 foldのみを先行テスト
- **異常値検出**: 負値・NaN・外れ値を各ステップで確認
- **ログ詳細化**: print文で中間結果を逐次表示

### 失敗時のフォールバック
1. **層化失敗**: n_bins減少 or GroupKFold に戻す
2. **Isotonic劣化**: 閾値調整 or 適用見送り
3. **NNLS係数異常**: positive_excess特徴を削除
4. **Tweedie効果なし**: variance_power固定 or 他損失を試行

---

## 次のステップ

### 即座に実施可能
- Phase 1（層化CV）の実装：ノートブックに追加セルを挿入

### 準備完了後
- Phase 1の結果を評価 → 有効なら Phase 2へ
- 各Phase完了後、README にスコアを追記
- 最良の組み合わせで最終提出ファイルを生成

### 必要に応じて
- CatBoostにもTweedieを適用（Phase 4で有効なら）
- Isotonic閾値の最適化（Phase 2で改善余地があれば）
- 複数手法の積み上げ（例: 層化CV + Isotonic + NNLS）

---

## まとめ

### 実装済み
✅ 4つのモジュール（cv, calibration, stacking, weights）  
✅ 実験計画とドキュメント  
✅ ディレクトリ構造（exp0027/）

### 次のアクション
🔄 Phase 1: ノートブックへの層化CV追加セル挿入  
⏳ Phase 2-5: 順次実装・評価

### 期待される成果
- **安定性**: fold間のwRMSE揺れを大幅削減
- **精度**: OOF wRMSE で -0.005〜-0.007 の改善
- **汎用性**: モジュール化により他実験でも再利用可能

