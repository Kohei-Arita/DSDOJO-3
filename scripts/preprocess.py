#!/usr/bin/env python3
"""
データ前処理・特徴量エンジニアリング

Usage:
    python -m scripts.preprocess --config configs/data.yaml --input data/raw --output data/processed
"""

import argparse
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
import logging
from typing import Dict, Any


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """設定ファイルを読み込み"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量エンジニアリング"""
    logger = logging.getLogger(__name__)
    df = df.copy()

    # Title extraction
    logger.info("Extracting Title feature")
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.")
    title_mapping = {
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Dr": "Rare",
        "Rev": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Mlle": "Miss",
        "Countess": "Rare",
        "Ms": "Mrs",
        "Lady": "Rare",
        "Jonkheer": "Rare",
        "Don": "Rare",
        "Dona": "Rare",
        "Mme": "Mrs",
        "Capt": "Rare",
        "Sir": "Rare",
    }
    df["Title"] = df["Title"].map(title_mapping).fillna("Rare")

    # Family features
    logger.info("Creating family features")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Age bands
    logger.info("Creating age bands")
    df["AgeBand"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100], labels=["Child", "Teen", "Adult", "Middle", "Senior"])

    # Fare bands
    logger.info("Creating fare bands")
    df["FareBand"] = pd.qcut(df["Fare"], q=4, labels=["Low", "Medium", "High", "VeryHigh"], duplicates="drop")

    return df


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame, config: Dict[str, Any]):
    """データ前処理"""
    logger = logging.getLogger(__name__)

    # データ結合（前処理の一貫性のため）
    logger.info("Combining train and test data for preprocessing")
    all_data = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    n_train = len(train_df)

    # 欠損値処理
    logger.info("Handling missing values")
    preprocessing = config.get("preprocessing", {})
    fill_missing = preprocessing.get("fill_missing", {})

    for col, method in fill_missing.items():
        if col in all_data.columns:
            if method == "median":
                all_data[col].fillna(all_data[col].median(), inplace=True)
            elif method == "mode":
                all_data[col].fillna(all_data[col].mode()[0], inplace=True)
            elif method == "mean":
                all_data[col].fillna(all_data[col].mean(), inplace=True)
            else:
                all_data[col].fillna(method, inplace=True)

    # 特徴量エンジニアリング
    if preprocessing.get("feature_engineering", {}).get("Title", False):
        all_data = engineer_features(all_data)

    # 元のデータに分離
    train_processed = all_data[:n_train].copy()
    test_processed = all_data[n_train:].copy()

    return train_processed, test_processed


def save_feature_info(train_df: pd.DataFrame, output_dir: str):
    """特徴量情報を保存"""
    feature_info = {
        "n_features": len(train_df.columns),
        "features": list(train_df.columns),
        "dtypes": train_df.dtypes.astype(str).to_dict(),
        "missing_counts": train_df.isnull().sum().to_dict(),
        "shape": train_df.shape,
    }

    with open(Path(output_dir) / "feature_info.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Preprocess Titanic data")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")

    args = parser.parse_args()
    logger = setup_logging()

    # 設定読み込み
    config = load_config(args.config)

    # 入出力パス設定
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    logger.info(f"Loading data from {input_dir}")
    train_df = pd.read_csv(input_dir / "train.csv")
    test_df = pd.read_csv(input_dir / "test.csv")

    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # 前処理実行
    logger.info("Starting preprocessing")
    train_processed, test_processed = preprocess_data(train_df, test_df, config)

    # 保存
    logger.info(f"Saving processed data to {output_dir}")
    train_processed.to_parquet(output_dir / "train_processed.parquet", index=False)
    test_processed.to_parquet(output_dir / "test_processed.parquet", index=False)

    # 特徴量情報保存
    save_feature_info(train_processed, output_dir)

    logger.info(f"Preprocessing completed")
    logger.info(f"Train processed shape: {train_processed.shape}")
    logger.info(f"Test processed shape: {test_processed.shape}")
    logger.info(f"Features: {list(train_processed.columns)}")


if __name__ == "__main__":
    main()
