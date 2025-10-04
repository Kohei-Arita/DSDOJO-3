# exp0030: Tweedie損失実装計画

## 🎯 実験の目的

ゼロ過剰データ（zero-inflated data）に対応するため、LightGBMにTweedie損失を適用し、従来のL2損失との性能を比較する。

## 📊 ベースライン (exp0027)

- **CV mean**: 0.2315 (std 0.0052)
- **OOF RMSE**: 0.2316
- **使用損失**: L2 (squared error)

## 🔬 Tweedie損失について

### 理論的背景

Tweedie分布は、Poissonとガンマ分布を組み合わせた分布族で、ゼロ過剰データのモデリングに適しています：

- **variance_power = 1.0**: Poisson分布（カウントデータ）
- **variance_power = 1.5**: Compound Poisson-Gamma（保険請求データなど）
- **variance_power = 2.0**: ガンマ分布（連続正値データ）

xAGデータの特性：
- ゼロが多い（約69%が xAG < 0.1）
- 正値は連続値
- → variance_power = 1.1 ~ 1.9 の範囲で最適化

### LightGBMでの実装

```python
params = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.5,  # 最適化対象
    'metric': 'rmse',  # 評価はRMSEで統一
    # その他のパラメータは既存と同じ
}
```

## 📋 実装ステップ

### Step 1: ディレクトリとドキュメント準備
- [x] `experiments/exp0030/` ディレクトリ作成
- [x] `README.md` 作成
- [x] `IMPLEMENTATION_PLAN.md` 作成（このファイル）

### Step 2: ベースノートブックのコピーと確認
- [ ] exp0027のノートブックをexp0030にコピー
- [ ] ファイルサイズとセル数を確認
- [ ] GroupKFoldセルを特定

### Step 3: Tweedie損失の実装（LGBMのみ）

#### 3.1 Optunaハイパーパラメータ探索の修正
- [ ] Cell: Optuna探索空間の定義
  - `objective`: 'tweedie' に固定
  - `tweedie_variance_power`: trial.suggest_float(1.1, 1.9) を追加
  - その他のパラメータは既存を維持

#### 3.2 LightGBM学習パラメータの更新
- [ ] Cell: 5-Fold CV学習
  - paramsに `objective: 'tweedie'` を追加
  - `tweedie_variance_power` をOptunaの最適値に設定
  - `metric: 'rmse'` は維持（評価指標は変更しない）

#### 3.3 比較用にL2損失の結果も保存
- [ ] 既存のL2結果（exp0027）と比較できるようにメトリクスを記録

### Step 4: 結果の分析と可視化

- [ ] OOFスコアの比較
  - Tweedie vs L2
  - 正領域（y >= 0.1）と負領域（y < 0.1）で分けて評価
- [ ] 予測分布の比較
  - ヒストグラム
  - ゼロ近傍の予測精度
- [ ] Tweedie variance_powerの影響分析

### Step 5: CatBoostとのブレンド（Tweedie LGBM + 既存CB）

- [ ] CatBoostは既存のまま使用
- [ ] Tweedie-LGBMとCatBoostのブレンド
  - グリッドサーチで最適比率を探索
  - または前回のNNLS係数を参考

### Step 6: メトリクスの保存とREADME更新

- [ ] `logs/host_baseline_002_metrics.json` に結果を保存
- [ ] README実験台帳に結果を追記

## ⚠️ 注意事項

### 1. CatBoostについて
- CatBoostはTweedie損失をサポートしていないため、**LGBMのみ**にTweedieを適用
- CatBoostは既存のL2損失のままで学習済みモデルを使用

### 2. 評価指標
- 学習時の損失: Tweedie
- 評価指標: wRMSE（変更なし）
- Optunaの最適化目標: wRMSE（変更なし）

### 3. パラメータチューニング
- variance_powerを含めた再最適化が理想
- ただし計算時間を考慮し、既存のnum_leaves/learning_rate/min_child_samplesは固定も検討

### 4. データリーク防止
- StratifiedGroupKFoldは維持
- foldの分割は変更しない

## 🔍 期待される効果

### ポジティブシナリオ
- ゼロ近傍の予測精度向上
- 正領域での過剰な予測を抑制
- wRMSEスコアの改善

### ネガティブリスク
- Tweedieの性質上、予測が保守的になりすぎる可能性
- ハイパーパラメータの再調整が必要になる可能性
- スコアが横ばいまたは悪化する可能性

## 📝 実装時のチェックリスト

### コーディング品質
- [ ] 変数名が明確で一貫している
- [ ] コメントが適切に記載されている
- [ ] エラーハンドリングが適切
- [ ] 出力が読みやすい

### テスト・検証
- [ ] 既存のfold分割と一致しているか
- [ ] OOF予測のサイズが正しいか
- [ ] テスト予測のサイズが正しいか
- [ ] メトリクスが正しく計算されているか

### ドキュメント
- [ ] コードの意図が明確か
- [ ] 結果の解釈が記載されているか
- [ ] 次のステップが明確か

## 🚀 実装開始前の確認

以下を確認してから実装を開始します：

1. ✅ exp0027が完了し、結果が保存されている
2. ✅ StratifiedGroupKFoldが適用されている
3. ✅ 既存のLGBM/CatBoostのOOF予測が存在する
4. ✅ セッションに十分なトークンが残っている（90万トークン以上）

---

**準備が整い次第、Step 2から順番に実装を進めます。**

