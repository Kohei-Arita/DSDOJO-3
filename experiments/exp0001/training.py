# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
# ---

# %% [markdown]
# # Titanic Training Notebook - exp0001
#
# **実験概要**: Baseline model with title and family features
#
# **実行環境**: 
# - Google Colab (GPU)
# - LightGBM with categorical features

# %%
# Colab環境セットアップ
import os
import sys

# ランタイムタイプの確認
!nvidia-smi

# リポジトリクローン（初回のみ）
if not os.path.exists('/content/LIGHTBGM-TEM'):
    !git clone https://github.com/YOUR_USERNAME/LIGHTBGM-TEM.git

%cd /content/LIGHTBGM-TEM/kaggle-projects/titanic/experiments/exp0001

# %%
# 依存関係のインストール
!pip install -r env/requirements.lock

# %%
# APIキーの設定（Secrets経由）
from google.colab import userdata

os.environ["WANDB_API_KEY"] = userdata.get('WANDB_API_KEY')
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')

# %%
import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
import json
import wandb
import hashlib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 設定読み込み
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

print(f"実験ID: {cfg['experiment']['id']}")
print(f"実験説明: {cfg['experiment']['description']}")

# %%
# W&B初期化
run = wandb.init(
    project=cfg['wandb']['project'],
    name=cfg['experiment']['id'],
    config=cfg,
    tags=cfg['wandb']['tags'],
    job_type="train"
)

# W&B Run URLを保存
with open('wandb_run.txt', 'w') as f:
    f.write(f"{run.url}\n{run.id}")

# %%
# Git SHA取得
import subprocess

try:
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
except:
    git_sha = 'unknown'

with open('git_sha.txt', 'w') as f:
    f.write(git_sha)

print(f"Git SHA: {git_sha}")

# %%
# データ読み込み関数
def load_and_preprocess_data(cfg):
    """データの読み込みと前処理"""
    
    # 生データ読み込み
    train_df = pd.read_csv(f"{cfg['paths']['raw_dir']}/train.csv")
    test_df = pd.read_csv(f"{cfg['paths']['raw_dir']}/test.csv")
    
    # データ結合
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    
    # 欠損値処理
    all_data['Age'].fillna(all_data['Age'].median(), inplace=True)
    all_data['Fare'].fillna(all_data['Fare'].median(), inplace=True)
    all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace=True)
    
    # 特徴量エンジニアリング
    # Title抽出
    all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.')
    title_mapping = {
        'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Mrs', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    all_data['Title'] = all_data['Title'].map(title_mapping).fillna('Rare')
    
    # Family features
    all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
    all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)
    
    # Age bands
    all_data['AgeBand'] = pd.cut(all_data['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # Fare bands
    all_data['FareBand'] = pd.qcut(all_data['Fare'], q=4, 
                                   labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # 学習・テストに分離
    train_data = all_data[:len(train_df)].copy()
    test_data = all_data[len(train_df):].copy()
    
    # 特徴量とターゲットを分離
    feature_cols = cfg['features']['use']
    
    X = train_data[feature_cols].copy()
    y = train_data[cfg['data']['target']]
    X_test = test_data[feature_cols].copy()
    
    # カテゴリカル変数をLightGBM用にカテゴリ型に変換
    categorical_cols = cfg['data']['categorical']
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
            X_test[col] = X_test[col].astype('category')
    
    return X, y, X_test, train_data, test_data

# データ読み込み
X, y, X_test, train_data, test_data = load_and_preprocess_data(cfg)

print(f"学習データ形状: {X.shape}")
print(f"テストデータ形状: {X_test.shape}")
print(f"特徴量: {list(X.columns)}")

# デバッグ: データ型を確認
print(f"\nデータ型:")
print(X.dtypes)
print(f"\nカテゴリカル変数: {cfg['data']['categorical']}")
for col in cfg['data']['categorical']:
    if col in X.columns:
        print(f"{col}: {X[col].dtype}")

# %%
# CV分割を作成・保存
skf = StratifiedKFold(
    n_splits=cfg['cv']['n_splits'], 
    shuffle=cfg['cv']['shuffle'], 
    random_state=cfg['cv']['seed']
)

# split_id（分割の一意識別子）を生成
split_config = f"stratified_{cfg['cv']['n_splits']}_{cfg['cv']['seed']}"
split_id = hashlib.md5(split_config.encode()).hexdigest()[:8]

# CV分割をDataFrameとして保存
cv_folds = []
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    for idx in train_idx:
        cv_folds.append({'index': idx, 'fold': -1})  # train folds are marked as -1
    for idx in valid_idx:
        cv_folds.append({'index': idx, 'fold': fold})

cv_folds_df = pd.DataFrame(cv_folds)
cv_folds_df['split_id'] = split_id
cv_folds_df.to_parquet('cv_folds.parquet', index=False)

print(f"CV分割保存完了: {split_id}")
print(cv_folds_df['fold'].value_counts().sort_index())

# %%
# 学習ループ
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))
fold_scores = []
fold_models = []
feature_importance = pd.DataFrame()

start_time = datetime.now()

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\n=== Fold {fold + 1}/{cfg['cv']['n_splits']} ===")
    
    # データ分割
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    # LightGBM Dataset作成
    # カテゴリカル特徴量のインデックスを取得
    categorical_features = [col for col in cfg['data']['categorical'] if col in X.columns]
    
    train_dataset = lgb.Dataset(
        X_train, 
        label=y_train,
        categorical_feature=categorical_features
    )
    
    valid_dataset = lgb.Dataset(
        X_valid, 
        label=y_valid,
        categorical_feature=categorical_features,
        reference=train_dataset
    )
    
    # モデル学習
    model = lgb.train(
        cfg['lgbm']['params'],
        train_dataset,
        valid_sets=[train_dataset, valid_dataset],
        valid_names=['train', 'valid'],
        num_boost_round=cfg['lgbm']['train']['num_boost_round'],
        callbacks=[
            lgb.early_stopping(cfg['lgbm']['train']['early_stopping_rounds'], verbose=False),
            lgb.log_evaluation(cfg['lgbm']['train']['verbose_eval'])
        ]
    )
    
    # 予測
    valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # OOF予測を保存
    oof_predictions[valid_idx] = valid_pred
    test_predictions += test_pred / cfg['cv']['n_splits']
    
    # スコア計算
    fold_score = roc_auc_score(y_valid, valid_pred)
    fold_scores.append(fold_score)
    
    print(f"Fold {fold + 1} AUC: {fold_score:.6f}")
    print(f"Best iteration: {model.best_iteration}")
    
    # モデル保存
    model_path = f"model/fold{fold}.lgb"
    model.save_model(model_path)
    fold_models.append(model_path)
    
    # Feature importance
    fold_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance(importance_type='gain'),
        'fold': fold
    })
    feature_importance = pd.concat([feature_importance, fold_importance])
    
    # W&B logging
    wandb.log({
        f'fold_{fold}_auc': fold_score,
        f'fold_{fold}_best_iter': model.best_iteration
    })

train_time = (datetime.now() - start_time).total_seconds()
print(f"\n=== 学習完了 ===")
print(f"OOF AUC: {roc_auc_score(y, oof_predictions):.6f}")
print(f"CV AUC: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
print(f"学習時間: {train_time:.1f}秒")

# %%
# OOF予測を保存
oof_df = pd.DataFrame({
    'index': range(len(X)),
    'fold': -1,  # 後でcv_foldsから正しいfoldを割り当て
    'y_true': y,
    'y_pred': oof_predictions
})

# 正しいfold番号を割り当て
for fold, (_, valid_idx) in enumerate(skf.split(X, y)):
    oof_df.loc[valid_idx, 'fold'] = fold

oof_df.to_parquet('oof.parquet', index=False)

# メトリクス保存
metrics = {
    'cv': {
        'metric': 'auc',
        'mean': float(np.mean(fold_scores)),
        'std': float(np.std(fold_scores)),
        'per_fold': [float(score) for score in fold_scores]
    },
    'train_time_sec': train_time,
    'best_iteration_per_fold': [model.best_iteration for model in lgb.Booster(model_file=path) for path in fold_models]
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# 特徴量リスト保存
with open('feature_list.txt', 'w') as f:
    for feature in X.columns:
        f.write(f"{feature}\n")

print("実験成果物保存完了")

# %%
# W&B Artifacts保存
artifact = wandb.Artifact(f"{cfg['experiment']['id']}-artifacts", type="experiment")

# ファイルを追加
files_to_log = [
    'config.yaml', 'cv_folds.parquet', 'oof.parquet', 'metrics.json', 
    'feature_list.txt', 'git_sha.txt'
]

for file_path in files_to_log:
    if Path(file_path).exists():
        artifact.add_file(file_path)

# モデルファイルを追加
for model_path in fold_models:
    artifact.add_file(model_path)

# アーティファクトをログ
run.log_artifact(artifact)

# 最終メトリクスをログ
wandb.log({
    'cv_auc_mean': metrics['cv']['mean'],
    'cv_auc_std': metrics['cv']['std'],
    'train_time_sec': train_time
})

# Feature importance plot
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
feature_importance_mean = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
sns.barplot(x=feature_importance_mean.values, y=feature_importance_mean.index)
plt.title('Feature Importance (Mean across folds)')
plt.tight_layout()
wandb.log({"feature_importance": wandb.Image(plt)})
plt.show()

wandb.finish()
print("W&B logging完了")

# %%
# 提出用予測作成（テストセット）
submission = pd.DataFrame({
    cfg['data']['id']: test_data[cfg['data']['id']],
    cfg['data']['target']: (test_predictions > 0.5).astype(int)
})

submission_path = f"submissions/submission.csv"
submission.to_csv(submission_path, index=False)

print(f"提出ファイル作成完了: {submission_path}")
print(f"予測分布: {submission[cfg['data']['target']].value_counts().sort_index()}")
submission.head()

# %%
# 実験ノート
notes = f"""
# 実験ノート - {cfg['experiment']['id']}

## 実験設定
- 日付: {cfg['experiment']['date']}
- 説明: {cfg['experiment']['description']}
- Git SHA: {git_sha}

## 結果
- CV AUC: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}
- 学習時間: {train_time:.1f}秒

## 使用特徴量
{chr(10).join([f'- {feat}' for feat in X.columns])}

## 所感
- ベースラインとして良好なスコア
- TitleとFamilySize特徴量が有効
- 次回: ハイパーパラメータ調整

## 次のアクション
- [ ] Optunaでハイパーパラメータ最適化
- [ ] 他のCVスキーム（GroupKFold）を試す
- [ ] 新しい特徴量を追加
"""

with open('notes.md', 'w', encoding='utf-8') as f:
    f.write(notes)

print("実験ノート保存完了")