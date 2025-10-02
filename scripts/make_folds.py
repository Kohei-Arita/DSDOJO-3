#!/usr/bin/env python3
"""
CV分割を作成・保存

Usage:
    python -m scripts.make_folds --config configs/cv.yaml --data data/processed/train_processed.parquet --output cv_folds.parquet
"""

import argparse
import pandas as pd
import numpy as np
import yaml
import hashlib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GroupKFold, TimeSeriesSplit, train_test_split
import logging


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def create_cv_splits(df: pd.DataFrame, config: dict, target_col: str = "Survived"):
    """CV分割を作成"""
    logger = logging.getLogger(__name__)

    method = config["method"]
    n_splits = config["n_splits"]
    seed = config["seed"]

    logger.info(f"Creating {method} splits with {n_splits} folds, seed={seed}")

    cv_folds = []
    split_config = f"{method}_{n_splits}_{seed}"
    split_id = hashlib.md5(split_config.encode()).hexdigest()[:8]

    if method == "stratified_kfold":
        skf = StratifiedKFold(n_splits=n_splits, shuffle=config.get("shuffle", True), random_state=seed)

        for fold, (train_idx, valid_idx) in enumerate(skf.split(df, df[target_col])):
            # Train indices (marked as -1)
            for idx in train_idx:
                cv_folds.append(
                    {
                        "index": idx,
                        "fold": -1,  # train folds are -1
                        "split_id": split_id,
                    }
                )
            # Valid indices
            for idx in valid_idx:
                cv_folds.append({"index": idx, "fold": fold, "split_id": split_id})

    elif method == "group_kfold":
        group_col = config.get("group_col")
        if not group_col or group_col not in df.columns:
            raise ValueError(f"GroupKFold requires group_col, got: {group_col}")

        gkf = GroupKFold(n_splits=n_splits)
        groups = df[group_col]

        for fold, (train_idx, valid_idx) in enumerate(gkf.split(df, df[target_col], groups)):
            for idx in train_idx:
                cv_folds.append({"index": idx, "fold": -1, "split_id": split_id})
            for idx in valid_idx:
                cv_folds.append({"index": idx, "fold": fold, "split_id": split_id})

    elif method == "time_series_split":
        tss = TimeSeriesSplit(n_splits=n_splits)

        for fold, (train_idx, valid_idx) in enumerate(tss.split(df)):
            for idx in train_idx:
                cv_folds.append({"index": idx, "fold": -1, "split_id": split_id})
            for idx in valid_idx:
                cv_folds.append({"index": idx, "fold": fold, "split_id": split_id})
    else:
        raise ValueError(f"Unknown CV method: {method}")

    cv_folds_df = pd.DataFrame(cv_folds)

    # データフィンガープリント追加（オプション）
    data_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()[:8]
    cv_folds_df["data_fingerprint"] = data_hash

    logger.info(f"Created CV splits: split_id={split_id}")
    logger.info(f"Fold distribution: {cv_folds_df['fold'].value_counts().sort_index()}")

    return cv_folds_df


def validate_cv_splits(cv_folds_df: pd.DataFrame, df: pd.DataFrame, target_col: str, config: dict):
    """CV分割の品質をチェック"""
    logger = logging.getLogger(__name__)

    validation = config.get("validation", {})
    min_samples = validation.get("min_samples_per_fold", 100)
    target_threshold = validation.get("target_distribution_threshold", 0.05)

    logger.info("Validating CV splits quality")

    # 各Foldのサンプル数チェック
    fold_counts = cv_folds_df[cv_folds_df["fold"] >= 0]["fold"].value_counts()
    min_fold_size = fold_counts.min()

    if min_fold_size < min_samples:
        logger.warning(f"Small fold detected: {min_fold_size} < {min_samples}")

    # Target分布の均一性チェック
    fold_target_rates = []
    for fold in fold_counts.index:
        fold_indices = cv_folds_df[cv_folds_df["fold"] == fold]["index"].values
        fold_target_rate = df.iloc[fold_indices][target_col].mean()
        fold_target_rates.append(fold_target_rate)

    target_std = np.std(fold_target_rates)
    logger.info(f"Target distribution std across folds: {target_std:.6f}")

    if target_std > target_threshold:
        logger.warning(f"Uneven target distribution: {target_std:.6f} > {target_threshold}")

    # サマリー出力
    logger.info(f"CV validation summary:")
    logger.info(f"- Fold sizes: min={min_fold_size}, max={fold_counts.max()}")
    logger.info(f"- Target rates: {[f'{rate:.3f}' for rate in fold_target_rates]}")
    logger.info(f"- Target std: {target_std:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Create CV splits for training")
    parser.add_argument("--config", required=True, help="CV config file")
    parser.add_argument("--data", required=True, help="Training data file")
    parser.add_argument("--output", required=True, help="Output CV folds file")
    parser.add_argument("--target", default="Survived", help="Target column name")

    args = parser.parse_args()
    logger = setup_logging()

    # 設定とデータ読み込み
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df = pd.read_parquet(args.data)
    logger.info(f"Loaded data: {df.shape}")

    # CV分割作成
    cv_folds_df = create_cv_splits(df, config, args.target)

    # 品質チェック
    validate_cv_splits(cv_folds_df, df, args.target, config)

    # 保存
    cv_folds_df.to_parquet(args.output, index=False)
    logger.info(f"Saved CV folds to {args.output}")


if __name__ == "__main__":
    main()
