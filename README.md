# Titanic Competition - Kaggle Grandmaster Template

Kaggleグランドマスター流の実験管理構成を使用したTitanicコンペテンプレートです。

## 🎯 競技概要

- **競技名**: Titanic - Machine Learning from Disaster
- **評価指標**: Accuracy
- **提出フォーマット**: PassengerId, Survived (0 or 1)
- **データ**: 学習データ 891件、テストデータ 418件

## 🏗️ プロジェクト構成

```
titanic/
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
├── data/                 # DVC管理データ
│   ├── raw/             # 生データ（Kaggle API取得）
│   ├── processed/       # 前処理済みデータ
│   └── external/        # 外部データ
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

## 🚀 Google Colab クイックスタート

### 1. 環境準備

```python
# 1) GPU ランタイムを選択
# 2) リポジトリクローン
!git clone https://github.com/YOUR_USERNAME/LIGHTBGM-TEM.git
%cd LIGHTBGM-TEM/kaggle-projects/titanic/experiments/exp0001

# 3) 依存関係インストール
!pip install -r env/requirements.lock

# 4) APIキー設定（Colab Secrets推奨）
import os
from google.colab import userdata
os.environ[\"WANDB_API_KEY\"] = userdata.get('WANDB_API_KEY')  # 任意
os.environ[\"KAGGLE_USERNAME\"] = userdata.get('KAGGLE_USERNAME')
os.environ[\"KAGGLE_KEY\"] = userdata.get('KAGGLE_KEY')
```

### 2. データ準備

```bash
# Kaggleデータダウンロード
cd ../..  # titanic/
kaggle competitions download -c titanic -p data/raw --unzip

# 前処理実行
python -m scripts.preprocess --config configs/data.yaml --input data/raw --output data/processed
```

### 3. 実験実行

```python
# experiments/exp0001/ で実行
# 1) training.ipynb: 学習・OOF・モデル保存・W&B
# 2) evaluation.ipynb: OOF分析・CV品質チェック
# 3) inference.ipynb: 推論・提出・台帳更新
```

## 📊 実験管理の仕組み

### 設定の階層

1. **基底設定** (`configs/`): プロジェクト共通の設定
2. **実験スナップショット** (`experiments/exp0001/config.yaml`): 実行時の固定設定

### CV分割の固定

```python
# cv_folds.parquet でCV分割を完全固定
# split_id でCV手法を識別
# 同一分割での横比較を保証
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
# データダウンロード
python -m scripts.download_data --competition titanic --output data/raw

# 前処理
python -m scripts.preprocess --config configs/data.yaml --input data/raw --output data/processed

# CV分割作成
python -m scripts.make_folds --config configs/cv.yaml --data data/processed/train_processed.parquet --output cv_folds.parquet
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

## 📈 特徴量エンジニアリング

### 実装済み特徴量

- **Title**: 敬称抽出・グルーピング (Mr/Mrs/Miss/Master/Rare)
- **FamilySize**: SibSp + Parch + 1
- **IsAlone**: FamilySize == 1
- **AgeBand**: 年齢区間 (Child/Teen/Adult/Middle/Senior)
- **FareBand**: 運賃区間 (4分位)

### カテゴリ特徴量

LightGBMのネイティブcategorical機能を活用：

```python
categorical_feature = ['Sex', 'Embarked', 'Title', 'AgeBand', 'FareBand']
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
   dvc remote add -d storage s3://your-bucket/titanic
   dvc remote modify storage access_key_id YOUR_ACCESS_KEY
   dvc remote modify storage secret_access_key YOUR_SECRET_KEY
   ```

## 📚 参考資料

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [DVC User Guide](https://dvc.org/doc/user-guide)
- [Weights & Biases Guides](https://docs.wandb.ai/)
- [Kaggle API](https://github.com/Kaggle/kaggle-api)

## 🎯 次のステップ

1. **ハイパーパラメータ最適化**: Optuna統合
2. **アンサンブル**: 複数モデル・CV手法の組み合わせ
3. **特徴量追加**: 外部データ・高次特徴量
4. **AutoML**: 自動特徴選択・ハイパーパラメータ探索

---

**\"Trust Your CV\"** - CVを信頼し、LBとの乖離を監視しながら改善を重ねましょう🚀