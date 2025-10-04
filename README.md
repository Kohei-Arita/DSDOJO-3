# xAG Prediction Competition - Kaggle Grandmaster Template

Kaggleグランドマスター流の実験管理構成を使用したxAG（アシスト期待値）予測コンペテンプレートです。

## 🎯 競技概要

- **競技名**: アシスト期待値（xAG）予測コンペティション
- **評価指標**: 重み付きRMSE (Position-Weighted RMSE)
- **提出フォーマット**: row_id, xAG
- **データ期間**: 2017-18シーズン 欧州主要リーグ
- **データ**: プレー単位のアクションデータ → 試合単位のxAG予測

## 🏗️ プロジェクト構成

```
xag-prediction/
├── experiments/           # 実験ディレクトリ（1実験=1ディレクトリ）
│   └── exp0001/          # ベースライン実験
│       ├── training.ipynb      # 学習ノートブック
│       ├── evaluation.ipynb    # OOF分析・CV品質チェック
│       ├── inference.ipynb     # 推論・提出
│       ├── config.yaml         # 実験設定スナップショット
│       ├── cv_folds.parquet    # CV分割固定
│       ├── oof.parquet         # OOF予測
│       ├── metrics.json        # CV/OOF指標
│       ├── model/              # fold別モデル
│       ├── logs/               # 実行ログ・CVメトリクス
│       ├── submissions/        # 提出ファイル
│       ├── env/requirements.lock # 固定環境
│       └── notes.md            # 実験ノート
├── data/                 # コンペデータ
│   ├── action_data.csv           # プレー単位アクションデータ (591MB)
│   ├── match_train_data.csv      # 試合単位訓練データ (4.4MB)
│   ├── match_test_data.csv       # 試合単位テストデータ (2.0MB)
│   └── sample_submission.csv     # 提出テンプレート (114KB)
├── configs/             # 基底設定
│   ├── data.yaml       # データ・前処理設定
│   ├── cv.yaml         # CV戦略設定
│   ├── lgbm.yaml       # LightGBM設定
│   └── features.yaml   # 特徴量設定
├── scripts/             # ユーティリティ
├── experiments.csv      # 実験台帳（自動追記）
├── dvc.yaml            # DVCパイプライン
└── README.md           # このファイル
```

## 🧪 @experiments 実験サマリー

| 実験ID / ブランチ                      | 元ファイル                            | 実施日        | 試したこと                                    | 精度への影響 (CV / LB etc.)                                                       | 結果                                                                      | 考察                                                                                                                                                        | 根拠・スクリーンショット                                                                                                  |
| :------------------------------- | :------------------------------- | :--------- | :--------------------------------------- | :-------------------------------------------------------------------------- | :---------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------ |
| exp0001_baseline                 | -                                | 2025-10-02 | LightGBMベースライン構築                         | CV: 0.246 → 0.231 (−6.1%)                                                   | ✅ 改善                                                                    | Optuna調整により過学習が抑制され汎化性能が向上                                                                                                                                | `/Users/aritakohei/Library/CloudStorage/Dropbox/スクリーンショットスクリーンショット 2025-10-02 16.08.39.png`                   |
| exp0001_host_baseline_002        | exp0001_baseline                 | 2025-10-02 | Optuna調整 (num_leaves=17, lr≈0.0196)      | CV mean: 0.2687 (OOF: 0.2688)                                               | -                                                                       | 比較基準となるベースライン。基本特徴量のみでの性能                                                                                                                                 | `experiments/exp0001/logs/host_baseline_002_metrics.json`                                                     |
| exp0002_host_baseline_002        | exp0001_host_baseline_002        | 2025-10-02 | アクション派生特徴量追加 + 時間正規化 + ターゲットエンコーディング     | CV mean: 0.2683 (std 0.0061) / OOF: 0.2684                                  | ✅ 改善                                                                    | 選手の行動パターンと時間要素の正規化が予測精度に寄与。わずかな改善(−0.0004)                                                                                                                | `experiments/exp0002/logs/host_baseline_002_metrics.json`, `experiments/exp0002/training.ipynb`               |
| exp0003_host_baseline_002        | exp0002_host_baseline_002        | 2025-10-02 | プログレッシブ/ディープ指標の集約特徴 + pass→shot拡張        | CV mean: 0.2662 (std 0.0060) / OOF: 0.2663                                  | ✅ 改善                                                                    | 攻撃的プレー連鎖の特徴量化が効果的。累積で−0.0025の改善                                                                                                                           | `experiments/exp0003/logs/host_baseline_002_metrics.json`, `experiments/exp0003/training.ipynb`               |
| exp0004_two_stage_hurdle         | exp0003_host_baseline_002        | 2025-10-02 | xAG>0分類→回帰の2段階LightGBM + 既存特徴群           | CV mean: 0.2889 (std 0.0059) / OOF: 0.2890                                  | ❌ 悪化                                                                    | 分類確率の縮小効果で高xAG試合を過小評価。キャリブレーション不足により+0.0227悪化                                                                                                             | `experiments/exp0004/logs/host_baseline_002_metrics.json`, `experiments/exp0004/training.ipynb`               |
| exp0005_squad_opponent_te        | exp0003_host_baseline_002        | 2025-10-03 | Squad×Opponent交互作用のOOFターゲットエンコーディング追加    | CV mean: 0.2659 (std 0.0061) / OOF: 0.2660                                  | ✅ 改善                                                                    | 対戦カード別のxAG傾向を捕捉。exp0003から−0.0003の改善でベストスコア更新                                                                                                              | `experiments/exp0005/logs/host_baseline_002_metrics.json`, `experiments/exp0005/training.ipynb`               |
| exp0006_monotone_constraints     | exp0005_squad_opponent_te        | 2025-10-03 | LightGBM単調性制約（monotone_constraints）の導入   | CV mean: 0.2657 (std 0.0061) / OOF: 0.2658                                  | ✅ 改善                                                                    | 14特徴量に単調増加制約を適用。exp0005から−0.0002改善で**新ベストスコア更新** (0.2657)。過学習抑制とドメイン知識の組み込みが効果的                                                                           | `experiments/exp0006/logs/host_baseline_002_metrics.json`, `experiments/exp0006/training.ipynb`               |
| exp0007_xt_features              | exp0006_monotone_constraints     | 2025-10-03 | xT (Expected Threat) 特徴量の追加              | CV mean: 0.2653 (std 0.0063) / OOF: 0.2654                                  | ✅ 改善                                                                    | ΔxT特徴が攻撃・位置指標を補完し、exp0006から−0.0004で新ベスト更新。fold2で高リフトを確認                                                                                                   | `experiments/exp0007/logs/host_baseline_002_metrics.json`, `experiments/exp0007/training.ipynb`               |
| exp0008_monotone_constraints_fix | exp0007_xt_features              | 2025-10-03 | 単調性制約対象列の整合性と存在チェック                      | CV mean: 0.2649 (std 0.0063) / OOF: 0.2650                                  | ✅ 改善                                                                    | 既存制約の列名ズレを修正し、攻撃系特徴量への単調増加制約を適正化。exp0007から−0.0004更新でベスト継続                                                                                                 | `experiments/exp0008/logs/host_baseline_002_metrics.json`, `experiments/exp0008/training.ipynb`               |
| exp0009_tweedie_quantile         | exp0008_monotone_constraints_fix | 2025-10-03 | Tweedie目的 + 分位LGBMブレンド (Isotonic校正含む)    | CV mean: 0.2657 (std 0.0070) / OOF: 0.2657 → 分位ブレンド OOF 0.2657 → 校正後 0.2735 | ❌ 悪化                                                                    | Tweedieは右裾重視でも基礎モデル比 +0.0007。分位ブレンドは0.1近傍のバイアス補正狙いも校正で逆に悪化。Tail強調は別の損失設計が必要                                                                               | `experiments/exp0009/logs/host_baseline_002_metrics.json`, `experiments/exp0009/training.ipynb`               |
| exp0010_learned_xt               | exp0009_tweedie_quantile         | 2025-10-03 | 学習型xT (value iteration) + ΔxT派生集約        | CV mean: 0.2560 (std 0.0060) / OOF: 0.2561                                  | ✅ 大幅改善                                                                  | 固定マトリクスxTをMarkov学習値へ置き換え、成功期待値×ΔxT等を選手×試合に集約。fold1で-0.020、平均でも-0.009改善し過去最良を更新                                                                            | `experiments/exp0010/logs/host_baseline_002_metrics.json`, `experiments/exp0010/training.ipynb`               |
| exp0011_possession_progression   | exp0010_learned_xt               | 2025-10-03 | 学習xT + ポゼッション進行速度/直進性集約                  | CV mean: 0.2530 (std 0.0064) / OOF: 0.2530                                  | ✅ 改善                                                                    | 学習xTに連動してポゼッション速度・Δx/秒・ファイナルサード到達ステップ等を追加。fold1で-0.0027、平均で-0.003改善と過去最良を更新                                                                               | `experiments/exp0011/logs/host_baseline_002_metrics.json`, `experiments/exp0011/training.ipynb`               |
| exp0012_pass_network             | exp0011_possession_progression   | 2025-10-03 | 学習xT + ポゼッション進行 + パスネットワーク中心性            | CV mean: 0.2533 (std 0.0064) / OOF: 0.2534                                  | ➖ 微差                                                                    | 中心性・受け口を追加しfold平均は横ばい。`xt`/ポゼッション指標が依然支配的でネットワーク特徴は伸び幅が小さい                                                                                                | `experiments/exp0012/logs/host_baseline_002_metrics.json`, `experiments/exp0012/training.ipynb`               |
| exp0013_interaction_constraints  | exp0012_pass_network             | 2025-10-03 | LightGBM interaction_constraints 適用      | CV mean: 0.3061 (std 0.0097) / OOF: 0.3063                                  | ❌ 大幅悪化                                                                  | 許可グループが狭く木の分割が制限され、学習xT/ポゼッション特徴を活かせず大幅劣化。制約設計の再検討が必要                                                                                                     | `experiments/exp0013/logs/host_baseline_002_metrics.json`, `experiments/exp0013/training.ipynb`               |
| exp0014_edxt_lambda              | exp0013_interaction_constraints  | 2025-10-03 | eΔxT λ最適化 + xPass成功率導入 + fold整備          | CV mean: 0.2459 (std 0.0067) / OOF: 0.2460                                  | ✅ 大幅改善                                                                  | eΔxT調整で攻撃リスク評価を改善し、過去の劣化を巻き返しつつベスト更新。fold列生成順の修正で再現性向上。                                                                                                   | `experiments/exp0014/logs/host_baseline_002_metrics.json`, `experiments/exp0014/training.ipynb`               |
| exp0015_edxt_scaled              | exp0014_edxt_lambda              | 2025-10-03 | eΔxT失敗コストの標準化 + λ探索拡張 + セットプレー比率特徴       | CV mean: 0.2459 (std 0.0068) / OOF: 0.2460                                  | ➖ 微差                                                                    | 失敗コストを標準化した上でセットプレー/オープンプレー集約を追加。スコアは横ばいだがλ分布が広がり分析余地を確保。                                                                                                 | `experiments/exp0015/logs/host_baseline_002_metrics.json`, `experiments/exp0015/training.ipynb`               |
| exp0016_edxt_group_norm          | exp0015_edxt_scaled              | 2025-10-03 | eΔxT失敗コストの行為タイプ別標準化 + λ統計出力              | CV mean: 0.2461 (std 0.0068) / OOF: 0.2462                                  | ➖ 微差                                                                    | 行為タイプごとに失敗成分を標準化しλ探索を拡張。スコアは横ばいだが負のλが分散し、パス系コストが顕在化。                                                                                                      | `experiments/exp0016/logs/host_baseline_002_metrics.json`, `experiments/exp0016/training.ipynb`               |
| exp0017_team_context_gate        | exp0016_edxt_group_norm          | 2025-10-03 | eΔxT成功確率+チーム文脈比率+リーグゲート交互作用              | CV mean: 0.2462 (std 0.0066) / OOF: 0.2463                                  | ➖ 微差                                                                    | チーム合計/LOPO比率とCompetition×xTクロスで解釈性向上。fold2悪化で平均はわずかに増だが他foldは安定、重要度にチーム比率系が浮上。                                                                            | `experiments/exp0017/logs/host_baseline_002_metrics.json`, `experiments/exp0017/training.ipynb`               |
| exp0018_league_gate_residual     | exp0017_team_context_gate        | 2025-10-03 | リーグゲート残差補正 + eΔxT再学習 + LOPO差分強化          | CV mean: 0.2464 (std 0.0068) / OOF: 0.2465                                  | ➖ 微差                                                                    | fold2悪化が継続し平均は+0.0002。リーグ別残差を補正しつつLOPO比率を拡張したが改善は限定的、重要度ではリーグ×xT crossが維持。                                                                                | `experiments/exp0018/logs/host_baseline_002_metrics.json`, `experiments/exp0018/training.ipynb`               |
| exp0019_weighted_sample          | exp0018_league_gate_residual     | 2025-10-03 | wRMSE重みをLightGBMの`sample_weight`全工程へ統一適用 | CV mean: 0.2461 (std 0.0066) / OOF: 0.2462                                  | ✅ 改善                                                                    | 指標重みと最適化を揃えfold1が0.2347まで低下。fold間ばらつきが縮み、リーグ残差補正と組み合わせた検証が安定。                                                                                             | `experiments/exp0019/logs/host_baseline_002_metrics.json`, `experiments/exp0019/training.ipynb`               |
| exp0020_catboost_blend           | exp0019_weighted_sample          | 2025-10-03 | CatBoost追加 + LightGBMブレンド比率のグリッド最適化      | CV mean: 0.2461 (std 0.0066) / OOF: 0.2462 / Optuna best 0.2447             | ➖ 微差                                                                    | CatBoostのOrdered TSと緩い単調性制約でモデル多様性を確保しLGBMとブレンドしたが、平均スコアはexp0019と同等。重み刻みの細分化やCatBoost側の最適化余地を検討。                                                          | `experiments/exp0020/logs/host_baseline_002_metrics.json`, `experiments/exp0020/training_with_catboost.ipynb` |
| exp0021_soft_moe                 | exp0020_catboost_blend           | 2025-10-04 | Soft MoE (Mixture of Experts) アーキテクチャ導入  | CV mean: 0.2476 (std 0.0071) / OOF: 0.2477 / Gating AUC: 0.898 / AP: 0.812  | ➖ 微悪化                                                                   | xAG threshold=0.1でLow/High expertを分離。Gating精度は高いが平均スコア+0.0015悪化。Low expertが基礎パターンのみ、High expertがチーム×戦術特徴を活用する明確な役割分担を確認。threshold最適化とexpert間バランス調整が次の改善点。 | `experiments/exp0021/logs/host_soft_moe_metrics.json`, `experiments/exp0021/training_with_catboost.ipynb`     |
| exp0023_soft_moe_threshold_028   | exp0021_soft_moe                 | 2025-10-04 | Soft MoE threshold 0.1→0.28へ変更           | CV mean: 0.2293 (std 0.0074) / OOF: 0.2294 / Gating AUC: 0.876 / AP: 0.464  | リーダーボードの乖離が大きいため、softmoe戦略はcloseとする。exp0020に特徴量追加のアップデートを次回から開始することにする。 | threshold=0.28でLow/High expertの分離点を調整。exp0021比で-0.0183の大幅改善を達成し過去最良を更新。Gating APは低下したが、expertの専門性が明確化され予測精度が向上。                                           | `experiments/exp0023/logs/host_soft_moe_metrics.json`, `experiments/exp0023/training_with_soft_moe.ipynb`     |
| exp0024_data_leakage_fix         | exp0020_catboost_blend           | 2025-10-04 | xPassモデルのPlattスケーリング較正リーケージを修正           | CV mean: 0.2309 (std 0.0057) / OOF: 0.2310 / Optuna best: 0.2303            | ✅ 大幅改善                                                                  | Platt較正のfold外データリーケージを修正（全OOF予測+全ラベルで較正していた問題を修正）。exp0020比で-0.0152の大幅改善を達成。データリーケージ修正と再最適化によりCV/テストの乖離低減とスコア向上を両立。過去最良に近いベンチマーク。                          | `experiments/exp0024/logs/host_baseline_002_metrics.json`, `experiments/exp0024/DATA_LEAKAGE_AUDIT_REPORT.md` |

| exp0025_host_baseline_002        | exp0024_data_leakage_fix         | 2025-10-04 | （結果のみ反映）                              | CV mean: 0.2307 (std 0.0058) / OOF: 0.2308 / Optuna best: 0.2297 (fold1: 0.2221) | ➖ 微差   | 詳細は `experiments/exp0025/` のレポート参照                                                                                      | `experiments/exp0025/logs/host_baseline_002_metrics.json` |

> **How to use**
> 1. 実験ごとに1行追加し、`experiments/expXXXX` での変更内容・仮説を簡潔にまとめる。
> 2. 精度指標は CV/OOF/LB など比較できる数値を前後で記録する。
> 3. 再現性を高めるため、関連ノートブック・PR・スクリーンショットなどのパスを記載する（上記は記入例）。
> 4. 追加情報が多い場合は `experiments/expXXXX/notes.md` に詳細を書き、本表からリンクする。

### exp0002_host_baseline_002 追加要素

- 選手の年齢を`Date`と`birth_date`から算出し、基本特徴量に追加。
- アクションデータを試合×選手に集約し、アクション総数・平均座標・ゴール数・アクションタイプ別カウントを結合。
- パスやシュートなど主要アクションの成功率、フィールドゾーン別アクション比率、出場時間あたりの指標を作成。
- 攻守アクションの比率や`pass → shot`の連続発生回数を特徴量化して攻撃寄りの振る舞いを捉える。
- `player_id` / `Squad` / `Opponent`に対してターゲットエンコーディングを実施し、CVリークを避けるためfold単位の平均で平滑化。

### exp0004_two_stage_hurdle 所感

- xAG>0の発生率は約31%で、分類ステージの確率が0.2〜0.3程度に収束するケースが多く、回帰出力との積によって高xAG試合を過度に縮小する挙動が発生した。
- 2段階化により軽微な外れ値は抑制できた一方、単段LightGBM（exp0003）と比較してCV meanが約+0.023悪化し、ゼロインフレ対策としては現状のままでは有効性が確認できなかった。
- 改善余地としては、分類確率のキャリブレーション（Platt/Isotonic）やゼロ除外時のリサンプル、回帰ステージでのメトリクス最適化（Quantile目標やタスク専用メトリック）を併用するアブレーションが必要。

### exp0005_squad_opponent_te 追加要素

- **Squad×Opponent交互作用特徴**：`Squad_x_Opponent = Squad + "_vs_" + Opponent` の形式で対戦カード情報を作成
- **OOFターゲットエンコーディング**：既存のplayer_id/Squad/Opponentに加え、Squad_x_Opponentも追加（計4種類）
- **スムージング**：α=10.0でベイズ的平滑化を実施し、少数サンプルの過学習を抑制
- **漏洩防止**：GroupKFold(match_id)で分割したfold外データでTEを算出し、fold内に適用
- exp0003（CV: 0.2662）から**−0.0003改善**でベストスコアを更新。対戦カード特有のxAG傾向（攻撃的vs守備的、強豪vs下位など）を効果的に捕捉した。

### exp0006_monotone_constraints 追加要素

- **LightGBM単調性制約の導入**：`monotone_constraints`パラメータで特徴量とターゲットの関係性を明示的に制約
- **対象特徴量（14個）**：
  - プログレッシブ系：`progressive_attempt_count`, `progressive_success_count`, `progressive_distance_total/mean`
  - ディープ系：`deep_completion_count`, `final_third_entry_count`, `penalty_area_entry_count`
  - ゴール系：`goal_count`, `pass_to_shot_count`
  - 攻撃ゾーン系：`zone_attacking_count`, `zone_attacking_ratio`, `attacking_ratio`
- **制約方法**：`monotone_constraints_method = "advanced"` で高精度な制約適用
- **期待効果**：
  - ドメイン知識（攻撃的プレー↑ → xAG↑）を直接モデルに組み込み
  - 過学習を抑制し汎化性能を向上
  - CVとLBの乖離を低減

### exp0007_xt_features 追加要素

- **結果サマリー**：CV mean 0.2653 (std 0.0063) / OOF 0.2654。exp0006_monotone_constraints比で−0.0004改善し、新ベスト。fold2 (0.2764) が突出する一方で他foldは0.264前後に収束し安定。
- **xT (Expected Threat) 特徴量の導入**：サッカー分析の標準指標をxAG予測に活用。
- **xTグリッド**：ピッチを16×12グリッドに分割し、各位置の得点脅威度を定義。
  - ゴールに近いほど、中央に近いほど高い脅威値。
  - Karun Singh の手法ベースの簡易実装（経験則）。
- **ΔxT計算**：アクションの開始位置と終了位置の脅威差分を算出。
  - 成功アクション：実際のxT増分。
  - 失敗アクション：開始地点の価値を30%失う（ペナルティ）。
- **Optuna最適化結果**：trial 29が最良。`num_leaves=27`, `learning_rate≈0.0148`, `min_child_samples=39`でΔxTとの相性が良好。
- **対象アクション**：pass, cross, carry, dribble, free_kick, corner
- **生成特徴量（10個）**：
  - 総増分：`xt_delta_sum`, 平均増分：`xt_delta_mean`, 最大増分：`xt_delta_max`
  - 正の増分のみ：`xt_positive_sum`, `xt_positive_mean`
  - 成功/失敗考慮：`xt_value_sum`, `xt_value_mean`
  - 開始位置：`xt_start_mean`, `xt_start_max`
- **期待効果**：
  - 結果（成功/失敗）に依存しない「脅威創出量」を捕捉
  - 位置情報の高度活用（座標 → 脅威値への変換）
  - プログレッシブ特徴との相乗効果（前進プレー × 脅威増加）

## 🐳 Docker クイックスタート（推奨）

### 1. Docker環境のセットアップ

```bash
# リポジトリクローン
git clone https://github.com/YOUR_USERNAME/DSDOJO-3.git
cd DSDOJO-3

# Dockerイメージのビルドと起動
docker-compose up -d

# Jupyter Labへアクセス
# ブラウザで http://localhost:8888 を開く
```

### 2. ノートブック実行

```
# Jupyter Labで experiments/exp0001/training.ipynb を開いて実行
# 全セルを順番に実行: Cell → Run All Cells
```

### 3. Docker環境の管理

```bash
# コンテナ停止
docker-compose down

# コンテナ再起動
docker-compose restart

# ログ確認
docker-compose logs -f
```

## 💻 ローカル環境セットアップ（Docker未使用の場合）

### 1. Python環境準備

```bash
# Python 3.11推奨
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係インストール
pip install -r requirements.txt
```

### 2. Jupyter Lab起動

```bash
jupyter lab

# ブラウザで experiments/exp0001/training.ipynb を開いて実行
```

### 3. ワンコマンド実験実行（Jupyter不要）

```bash
# config.yaml と生データ (match_train/test, action_data など) が paths.data_dir に揃っている前提
python -m scripts.run_local_experiment \
  --config experiments/exp0001/config.yaml \
  --output-dir experiments/exp0001/artifacts

# 実行後: artifacts/ 内に metrics.json, oof_predictions.csv, feature_importance.csv, submission_exp0001.csv が生成されます
# また `experiments/exp0001/logs/` にも最新のCV結果 (`*_metrics.json`, `*_training.log`) が記録されます
```

## 📊 コンペティション固有の特徴

### データの時間分解能ギャップ

このコンペティションの最大の特徴は、**入力と出力の時間分解能が異なる**点です：

- **入力データ**: プレー単位のアクションデータ (`action_data.csv`)
- **出力データ**: 試合単位のxAG (`match_train_data.csv`, `match_test_data.csv`)

プレーレベルの情報をどのように集約して試合レベルの予測に繋げるかが鍵となります。

### 評価指標: 重み付きRMSE

```python
def calculate_pw_rmse(labels, preds, w_pos=5.0, thresh=0.1):
    """Position-Weighted RMSE

    xAG >= 0.1 の場合に誤差を5倍に重み付け
    チャンス創出があった試合の予測精度を重視
    """
    weights = np.where(labels >= thresh, w_pos, 1.0)
    squared_errors = (labels - preds) ** 2
    weighted_squared_errors = weights * squared_errors
    pw_rmse = np.sqrt(np.mean(weighted_squared_errors) + 1e-9)
    return float(pw_rmse)
```

### xAG (Expected Assists) とは

- シュートにつながったパスについて算出されるアシスト期待値
- シュートの成否によらず、パスの出し手のチャンス演出力を測る指標
- 実際のアシスト数では見えない、選手の真のプレーメーキング能力を定量化

## 📊 実験管理の仕組み

### 設定の階層

1. **基底設定** (`configs/`): プロジェクト共通の設定
2. **実験スナップショット** (`experiments/exp0001/config.yaml`): 実行時の固定設定

### CV分割の固定

```python
# cv_folds.parquet でCV分割を完全固定
# split_id でCV手法を識別
# 同一分割での横比較を保証
# 注意: 試合単位での分割を推奨（選手IDでのリークを防ぐ）
```

### 成果物の追跡

- **OOF**: `oof.parquet` (index, fold, y_true, y_pred)  
- **メトリクス**: `metrics.json` (CV mean±std, per-fold)
- **モデル**: `model/fold*.lgb` (LightGBM binary)
- **提出**: `submissions/submission.csv` + manifest.json

### 実験台帳

`experiments.csv` に全実験の記録を自動追記：

| exp_id | cv_mean | cv_std | lb_public | git_sha | wandb_url | notes |
|--------|---------|--------|-----------|---------|-----------|-------|
| exp0001 | 0.8732 | 0.0061 | TBD | abcd1234 | wandb.ai/... | baseline |

## 🔧 主要コマンド

### データ管理（DVC）

```bash
# データパイプライン実行
dvc repro

# データ復元
dvc pull

# 新データ追加
dvc add data/external/new_data.csv
dvc push
```

### 実験実行（CLI版）

```bash
# データ確認
ls -lh data/*.csv

# 前処理（必要に応じて）
python -m scripts.preprocess --config configs/data.yaml --input data --output data

# CV分割作成（試合単位での分割を推奨）
python -m scripts.make_folds --config configs/cv.yaml --data data/match_train_data.csv --output cv_folds.parquet
```

### コード品質

```bash
# フォーマット・リント
black .
ruff . --fix

# pre-commit インストール
pre-commit install

# テスト実行
pytest tests/
```

## 📈 特徴量エンジニアリングのアイデア

### アクションデータの集約

プレー単位のデータを試合単位に集約する際の特徴量例：

- **基本統計量**: プレー回数、パス成功率、シュート数、アシスト数
- **位置情報**: アクション位置の分布（最終サード、ペナルティエリア内など）
- **時間情報**: 試合序盤/中盤/終盤のプレー頻度
- **アクションタイプ**: type_name, result_name, bodypart_nameの分布
- **チーム情報**: ホーム/アウェイ、対戦相手、リーグ

### 選手情報の活用

- **年齢**: 生年月日から算出した年齢・年齢区分
- **経験**: プレイ分数、先発/途中出場
- **ポジション**: 背番号・プレー位置からの推定

### カテゴリ特徴量

LightGBMのネイティブcategorical機能を活用：

```python
categorical_feature = [
    'competition', 'team_name_short', 'Venue',
    'type_name', 'result_name', 'bodypart_name'
]
```

## ⚙️ LightGBM設定

### 決定性の確保

```yaml
params:
  deterministic: true
  force_row_wise: true  # 数値安定性
  seed: 42
```

### 重み付きL2最小化

- 評価指標のwRMSEと整合させるため、xAGが0.1以上の試合には重み5.0、それ未満には1.0を割り当てて`sample_weight`に渡す
- Optuna探索・本学習・OOF評価のすべてで同じ重みを適用し、早期終了条件やモデル比較が指標と一致するようにする

### GPU対応

```yaml
# Linux + NVIDIA GPU
device_type: cuda

# OpenCL（互換性重視）
device_type: gpu  

# CPU（Colab GPUなし時）
device_type: cpu
```

## 📋 実験チェックリスト

### 学習前

- [ ] config.yamlで設定固定
- [ ] cv_folds.parquetでCV分割固定
- [ ] W&B初期化
- [ ] Git SHA記録

### 学習中

- [ ] foldごとのスコア監視
- [ ] early_stopping活用
- [ ] feature_importance記録

### 学習後

- [ ] OOF分析（evaluation.ipynb）
- [ ] CV品質チェック（リーク監査）
- [ ] 推論・提出（inference.ipynb）
- [ ] 実験台帳更新
- [ ] notes.md更新

## ⚠️ データリークへの注意

### 選手IDによるリーク

同じ選手が訓練データとテストデータの両方に登場します。試合単位でCV分割を行い、選手IDによる情報リークを防ぐことが重要です。

### 時間によるリーク

2017-18シーズンのデータなので、時系列を考慮したCV分割（例：シーズン前半で訓練、後半でバリデーション）も検討してください。

## ⚠️ 開発時の注意事項

### .ipynb と .py の同期

**重要**: このプロジェクトでは `.ipynb` ノートブックと `.py` スクリプトの同期に細心の注意が必要です。

- ノートブックで実験を行う際は、再現性のため必ず対応する `.py` ファイルも更新してください
- `jupytext` による自動同期を推奨（`jupytext --sync`）
- 同期忘れは実験の再現性を損なう原因となります

### String型エラーの注意

**カテゴリカル特徴量の処理に注意**:

```python
# ❌ 悪い例：String型のままLightGBMに渡すとエラー
categorical_features = ['competition', 'Squad', 'Opponent']

# ✅ 良い例：明示的にcategory型へ変換
for col in categorical_features:
    df[col] = df[col].astype('category')
```

**よくあるエラー**:
- `ValueError: Cannot use string features with LightGBM`
- 解決策：`categorical_feature`パラメータに列名を渡す **または** 事前に`category`型へ変換

## 🔍 トラブルシューティング

### よくある問題

1. **GPU未対応エラー**
   ```yaml
   # config.yaml で切り替え
   device_type: cpu
   ```

2. **Kaggle API認証エラー**
   ```bash
   # ~/.kaggle/kaggle.json 確認
   # または環境変数設定
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_key
   ```

3. **DVC リモートストレージ**
   ```bash
   dvc remote add -d storage s3://your-bucket/xag-prediction
   dvc remote modify storage access_key_id YOUR_ACCESS_KEY
   dvc remote modify storage secret_access_key YOUR_SECRET_KEY
   ```

## 📚 参考資料

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [DVC User Guide](https://dvc.org/doc/user-guide)
- [Weights & Biases Guides](https://docs.wandb.ai/)
- [Kaggle API](https://github.com/Kaggle/kaggle-api)
- [xG/xAG Explained - FBref](https://fbref.com/en/expected-goals-model-explained/)

## 🎯 次のステップ

1. **時間分解能ギャップの解決**: プレーレベル→試合レベルへの効果的な集約方法
2. **ハイパーパラメータ最適化**: Optuna統合（重み付きRMSEを目的関数に）
3. **アンサンブル**: 複数モデル・CV手法の組み合わせ
4. **特徴量追加**: シュート位置・パス位置の空間特徴、選手の過去xAG統計
5. **CV戦略**: 試合単位・時系列考慮の分割でリーク防止

---

**\"Trust Your CV\"** - 重み付きRMSEでCVを信頼し、LBとの乖離を監視しながら改善を重ねましょう⚽🚀
