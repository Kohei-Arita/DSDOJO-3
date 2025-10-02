# Google Colab セットアップガイド

## 🚀 基本セットアップ

### 1. ランタイムの設定

1. **ランタイム** → **ランタイムのタイプを変更**
2. **ハードウェアアクセラレータ**: GPU
3. **GPU の種類**: T4 (推奨)
4. **保存**をクリック

### 2. リポジトリのクローン

```python
# リポジトリクローン（初回のみ）
!git clone https://github.com/YOUR_USERNAME/LIGHTBGM-TEM.git
%cd LIGHTBGM-TEM/kaggle-projects/titanic/experiments/exp0001
```

### 3. 依存関係のインストール

```python
# 固定環境のインストール
!pip install -r env/requirements.lock

# GPUサポートの確認
import lightgbm as lgb
print(f"LightGBM GPU support: {lgb.GPUError}")
```

## 🔐 APIキーの設定

### Colab Secrets（推奨）

1. 🔑 **Secrets** パネルを開く
2. 以下のシークレットを追加：
   - `WANDB_API_KEY`: W&B API Key（任意）
   - `KAGGLE_USERNAME`: koheiari  
   - `KAGGLE_KEY`: 2f78e74cc916ba697e7d9c3853f68922

```python
from google.colab import userdata

# APIキー設定
import os
os.environ["WANDB_API_KEY"] = userdata.get('WANDB_API_KEY')
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME') 
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
```

### 直接設定（非推奨）

```python
# セキュリティリスクあり - 本番では使用しない
import os
os.environ["KAGGLE_USERNAME"] = "YOUR_KAGGLE_USERNAME"
os.environ["KAGGLE_KEY"] = "YOUR_KAGGLE_KEY"
```

## 📁 ファイル構造の確認

```python
# プロジェクト構造確認
!ls -la
!ls experiments/exp0001/
```

## ⚠️ 注意点とトラブルシューティング

### GPU関連

```python
# GPU使用可能チェック
!nvidia-smi

# LightGBM GPU設定
# config.yaml で device_type を切り替え
# device_type: cuda  # GPU使用時
# device_type: cpu   # CPU使用時
```

### メモリ不足

```python
# メモリ使用量確認
!free -h

# 必要に応じてランタイムをリセット
# ランタイム → ランタイムを再起動
```

### ファイル永続化

```python
# Google Driveマウント（大容量ファイル保存用）
from google.colab import drive
drive.mount('/content/drive')

# 実験成果物をDriveに保存
!cp -r experiments/exp0001/model /content/drive/MyDrive/titanic_models/
```

## 🔄 実験実行フロー

### 完全版（推奨）

```python
# 1. セットアップ
!git clone https://github.com/YOUR_USERNAME/LIGHTBGM-TEM.git
%cd LIGHTBGM-TEM/kaggle-projects/titanic

# 2. データ準備
!kaggle competitions download -c titanic -p data/raw --unzip
!python -m scripts.preprocess --config configs/data.yaml --input data/raw --output data/processed

# 3. 実験実行
%cd experiments/exp0001
# training.ipynb → evaluation.ipynb → inference.ipynb 順に実行
```

### 高速版（データ準備済み）

```python
# セットアップのみ
!git clone https://github.com/YOUR_USERNAME/LIGHTBGM-TEM.git
%cd LIGHTBGM-TEM/kaggle-projects/titanic/experiments/exp0001
!pip install -r env/requirements.lock

# 実験ノートブック実行
# training.ipynb から開始
```

## 🎯 パフォーマンス最適化

### LightGBM GPU設定

```yaml
# config.yaml
lgbm:
  params:
    device_type: cuda  # GPU使用
    gpu_platform_id: 0
    gpu_device_id: 0
```

### 並列処理

```python
# num_threads設定（CPU使用時）
params:
  num_threads: -1  # 全CPU利用
```

### メモリ効率

```python
# データ型最適化
df = df.astype({
    'Pclass': 'int8',
    'SibSp': 'int8', 
    'Parch': 'int8'
})
```

## 📊 モニタリング

### リソース使用量

```python
# GPU使用量監視
!watch -n 1 nvidia-smi

# CPU・メモリ監視
!htop
```

### 学習進捗

```python
import wandb

# W&B dashboard でリアルタイム監視
# https://wandb.ai/your-entity/titanic-lgbm
```

## 💾 結果の保存

### ローカル保存

```python
# ZIP圧縮して保存
!zip -r exp0001_results.zip experiments/exp0001/
```

### Google Drive保存

```python
# 重要ファイルのみ保存
!cp experiments/exp0001/metrics.json /content/drive/MyDrive/
!cp experiments/exp0001/oof.parquet /content/drive/MyDrive/
!cp experiments/exp0001/submissions/submission.csv /content/drive/MyDrive/
```

## 🔧 デバッグ

### 一般的なエラー

1. **ModuleNotFoundError**
   ```python
   !pip install missing_package
   ```

2. **CUDA out of memory**
   ```python
   # CPU使用に切り替え
   # config.yaml: device_type: cpu
   ```

3. **Kaggle API認証エラー**
   ```python
   # Secrets設定を確認
   print(os.environ.get("KAGGLE_USERNAME"))
   print(os.environ.get("KAGGLE_KEY"))
   ```

### ログファイル確認

```python
# エラーログ表示
!tail -100 /var/log/messages
!dmesg | tail -50
```