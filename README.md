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

| 実験ID / ブランチ | 実施日 | 試したこと | 精度への影響 (CV / LB etc.) | 根拠・スクリーンショット |
|:------------------|:-------|:-----------|:----------------------------|:---------------------------|
| exp0001_baseline  | 2025-10-02 | LightGBMベースライン構築 | CV: 0.246 → 0.231 (−6.1%) | `/Users/aritakohei/Library/CloudStorage/Dropbox/スクリーンショットスクリーンショット 2025-10-02 16.08.39.png` |

> **How to use**
> 1. 実験ごとに1行追加し、`experiments/expXXXX` での変更内容・仮説を簡潔にまとめる。
> 2. 精度指標は CV/OOF/LB など比較できる数値を前後で記録する。
> 3. 再現性を高めるため、関連ノートブック・PR・スクリーンショットなどのパスを記載する（上記は記入例）。
> 4. 追加情報が多い場合は `experiments/expXXXX/notes.md` に詳細を書き、本表からリンクする。

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
