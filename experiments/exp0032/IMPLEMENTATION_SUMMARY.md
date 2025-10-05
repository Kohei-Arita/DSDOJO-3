# exp0032: High専門家強化版MoE 実装サマリー

## 📋 実装完了内容

### ✅ 実装した改善

#### 1. **対数変換による学習安定化**
```python
# High領域のターゲットを対数変換
y_tr_high_log = np.log1p(y_tr_high)
y_val_high_log = np.log1p(y_val_high)

# 予測時に逆変換
high_oof_preds_log = high_model.predict(X_val_high)
high_oof_preds[val_mask] = np.expm1(high_oof_preds_log)
```

**効果**:
- 裾の長い分布を圧縮し、外れ値の影響を軽減
- 学習の安定性が向上
- exp0010で学習xTが効果的だったのと同じ原理

#### 2. **正則化の強化**
```python
high_params = {
    'min_child_samples': 50,  # 30 → 50（過学習抑制）
    'reg_alpha': 0.1,  # L1正則化追加
    'reg_lambda': 1.0,  # L2正則化強化
    # ... その他のパラメータ
}
```

**効果**:
- High領域はサンプル数が少ない（約30%）ため過学習しやすい
- 正則化により汎化性能が向上

### 📊 期待される改善

#### exp0031からの改善目標
| 指標 | exp0031 | exp0032目標 | 改善幅 |
|------|---------|-------------|--------|
| High専門家 wRMSE | 0.392 | 0.25 | -0.142 |
| MoE全体 OOF | 0.2271 | 0.225 | -0.002 |

#### 改善メカニズム
1. **対数変換**: High領域の予測精度が向上
2. **正則化**: 過学習を抑制し、CV/OOFの乖離を削減
3. **MoE効果**: High専門家の改善がMoE全体に波及

## 🔧 修正したファイル

### 主要な変更
- `experiments/exp0032/training_with_nnls.ipynb`
  - Step 4: High専門家セクションを対数変換版に置き換え
  - Step 6: メトリクス保存部分を exp0032 用に更新
  - Step 7: 提出ファイル名を更新

### 成果物の命名
- メトリクス: `logs/host_moe_high_opt_003_metrics.json`
- OOF詳細: `artifacts/oof_predictions_moe_high_opt.csv`
- 提出ファイル: `submissions/host_moe_high_opt_003_submission.csv`

## 🚀 実行手順

### 1. Jupyter Notebookで実行
```bash
cd /Users/aritakohei/DSDOJO-3/experiments/exp0032
jupyter lab training_with_nnls.ipynb
```

### 2. 実行するセル
- **既存セル（〜65）**: exp0031と同じく、データ読み込み〜NNLS統合まで実行
- **セル74**: Isotonic校正セルを実行（`calibrated_oof_preds`を定義）
- **MoEセル（66〜73）**: ゲート分類器、Low/High専門家、MoE合成、メトリクス保存

### 3. 実行順序の注意点
⚠️ **重要**: セル74（Isotonic校正）を実行してから、セル72（Step 6）を実行してください。

理由: Step 6で`calibrated_oof_preds`を使用しますが、これはセル74で定義されます。
→ `try-except`で対応済みなので、エラーは出ませんが、正しいスコアを得るには順序が重要です。

## 📈 検証ポイント

実行後、以下を確認してください：

### High専門家の改善
- [ ] Fold別のwRMSEがexp0031より改善しているか
- [ ] 対数変換後の分布統計が適切か（平均・標準偏差）
- [ ] 逆変換後の予測値が妥当な範囲か（非負、上限チェック）

### MoE全体の改善
- [ ] MoE OOF wRMSEがexp0031（0.2271）より改善しているか
- [ ] 温度パラメータτの最適値が変化しているか
- [ ] ゲート分離精度（AUC/AP）が維持されているか

### 予測の健全性
- [ ] OOF予測の分布が訓練データと類似しているか
- [ ] テスト予測の統計量が妥当か
- [ ] 負の予測値がクリップされているか

## 🔍 トラブルシューティング

### よくある問題

#### 1. `calibrated_oof_preds`が未定義エラー
**原因**: セル74より前にセル72を実行した
**解決策**: セル74を実行してから、セル72を再実行

#### 2. 対数変換でNaN/Inf
**原因**: 負の値に対してlog1pを適用した
**解決策**: High領域のフィルタ（`high_mask = y >= 0.1`）が正しいか確認

#### 3. High専門家のスコアが悪化
**原因**: 正則化が強すぎる、または対数変換が不適切
**解決策**:
- 正則化パラメータを調整（`reg_alpha`, `reg_lambda`を減らす）
- 対数変換を無効化して確認

## 💡 次のステップ候補

### さらなる改善案（優先度順）

#### A. Optuna最適化の追加
```python
# High専門家専用のOptuna最適化
def objective_high(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 10, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'min_child_samples': trial.suggest_int('min_child_samples', 30, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
    }
    # ... High領域で学習・評価
    return cv_score
```

#### B. 3専門家MoE
- Low（y < 0.05）、Mid（0.05 ≤ y < 0.2）、High（y ≥ 0.2）に分割
- 2段階ゲート機構の導入

#### C. CatBoost High専門家
- LightGBMとCatBoostのアンサンブル
- High領域での多様性向上

## 📝 実験ノート

### 作業履歴
- 2025-10-05: exp0032作成、High専門家の対数変換・正則化強化を実装
- セル順序の検証完了（セル74 → セル72の順で実行）
- README.md, IMPLEMENTATION_SUMMARY.md作成

### 次回実行時のメモ
- [ ] High専門家のOptuna最適化を追加（exp0033候補）
- [ ] Low専門家も同様に最適化（対数変換は不要の可能性あり）
- [ ] 温度パラメータτの探索範囲を拡大

---

**作成日**: 2025-10-05
**実験ID**: exp0032
**ベース**: exp0031（木モデル版MoE）
**改善内容**: High専門家の対数変換 + 正則化強化
