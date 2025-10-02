#!/usr/bin/env python3
"""Run the xAG baseline experiment without Jupyter.

This script reproduces the core steps from ``experiments/exp0001/training.ipynb``
so the experiment can be executed with a single command on a local machine.

Example
-------
    python -m scripts.run_local_experiment \
        --config experiments/exp0001/config.yaml \
        --output-dir experiments/exp0001/artifacts

The script expects the raw competition files listed in the config to be
available under ``paths.data_dir`` (same assumption as the notebook).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold


# ---------------------------------------------------------------------------
# Utility dataclasses and helpers


@dataclass
class ExperimentConfig:
    """Minimal subset of the YAML config needed for this script."""

    exp_id: str
    data_dir: Path
    files: Dict[str, str]
    target_col: str
    match_id_col: str
    player_id_col: str
    categorical_candidates: List[str]
    cv_seed: int
    cv_splits: int
    cv_group_col: str
    lgbm_params: Dict[str, object]
    lgbm_train: Dict[str, object]
    eval_w_pos: float
    eval_thresh: float


def setup_logger(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("run_local_experiment")


def load_yaml_config(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    data_cfg = raw_cfg["data"]
    cv_cfg = raw_cfg.get("cv", {})
    lgbm_cfg = raw_cfg.get("lgbm", {})
    eval_cfg = raw_cfg.get("evaluation", {})
    paths_cfg = raw_cfg.get("paths", {})
    exp_cfg = raw_cfg.get("experiment", {})

    return ExperimentConfig(
        exp_id=exp_cfg.get("id", "unknown_exp"),
        data_dir=(path.parent / paths_cfg.get("data_dir", "../../data")).resolve(),
        files={
            "match_train": data_cfg.get("match_train", "match_train_data.csv"),
            "match_test": data_cfg.get("match_test", "match_test_data.csv"),
            "action_data": data_cfg.get("action_data", "action_data.csv"),
            "sample_submission": data_cfg.get("sample_submission", "sample_submission.csv"),
        },
        target_col=data_cfg.get("target", "xAG"),
        match_id_col=data_cfg.get("match_id", "match_id"),
        player_id_col=data_cfg.get("player_id", "player_id"),
        categorical_candidates=list(data_cfg.get("categorical", [])),
        cv_seed=cv_cfg.get("seed", 42),
        cv_splits=cv_cfg.get("n_splits", 5),
        cv_group_col=cv_cfg.get("group_col", "match_id"),
        lgbm_params=lgbm_cfg.get("params", {}),
        lgbm_train=lgbm_cfg.get("train", {}),
        eval_w_pos=eval_cfg.get("w_pos", 5.0),
        eval_thresh=eval_cfg.get("thresh", 0.1),
    )


def ensure_files_exist(cfg: ExperimentConfig) -> None:
    missing = []
    for key, rel in cfg.files.items():
        candidate = cfg.data_dir / rel
        if not candidate.exists():
            missing.append(str(candidate))
    if missing:
        raise FileNotFoundError(
            "Required data files are missing:\n- " + "\n- ".join(missing)
        )


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, w_pos: float, thresh: float) -> float:
    weights = np.where(y_true >= thresh, w_pos, 1.0)
    squared_errors = (y_true - y_pred) ** 2
    weighted_squared_errors = weights * squared_errors
    return float(np.sqrt(np.mean(weighted_squared_errors) + 1e-9))


def make_feval(w_pos: float, thresh: float):
    def _feval(y_pred: np.ndarray, train_data: lgb.Dataset):
        y_true = train_data.get_label()
        score = weighted_rmse(y_true, y_pred, w_pos=w_pos, thresh=thresh)
        return "weighted_rmse", score, False

    return _feval


# ---------------------------------------------------------------------------
# Feature engineering (mirrors the notebook logic)


def compute_age_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    df["age"] = (df["Date"] - df["birth_date"]).dt.days / 365.25
    return df


def slice_relevant_actions(
    actions: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    match_col: str,
    player_col: str,
) -> pd.DataFrame:
    target_pairs = pd.concat(
        [
            train_df[[match_col, player_col]],
            test_df[[match_col, player_col]],
        ],
        axis=0,
    ).drop_duplicates()

    subset = actions.merge(target_pairs, on=[match_col, player_col], how="inner")
    if subset.empty:
        raise ValueError(
            "Filtered action data is empty. Check that action_data.csv contains matching "
            "match_id / player_id combinations."
        )
    return subset


def standardize_coordinates(actions: pd.DataFrame) -> pd.DataFrame:
    if "is_home" not in actions.columns:
        return actions

    actions = actions.copy()
    away_mask = actions["is_home"] == False
    if away_mask.any():
        actions.loc[away_mask, "start_x"] = 105 - actions.loc[away_mask, "start_x"]
        actions.loc[away_mask, "end_x"] = 105 - actions.loc[away_mask, "end_x"]
        actions.loc[away_mask, "start_y"] = 68 - actions.loc[away_mask, "start_y"]
        actions.loc[away_mask, "end_y"] = 68 - actions.loc[away_mask, "end_y"]
    return actions


def build_basic_stats(actions: pd.DataFrame, match_col: str, player_col: str) -> pd.DataFrame:
    required = {"type_name", "start_x", "start_y", "minutes_played"}
    missing = required - set(actions.columns)
    if missing:
        raise KeyError(f"Action data is missing required columns: {sorted(missing)}")

    return (
        actions.groupby([match_col, player_col])
        .agg(
            action_count=("type_name", "size"),
            avg_x=("start_x", "mean"),
            avg_y=("start_y", "mean"),
            minutes_played=("minutes_played", "first"),
        )
        .reset_index()
    )


def build_goal_counts(actions: pd.DataFrame, match_col: str, player_col: str) -> pd.DataFrame:
    required = {"type_name", "result_name"}
    missing = required - set(actions.columns)
    if missing:
        raise KeyError(f"Action data is missing required columns: {sorted(missing)}")

    shot_types = {"shot", "shot_freekick", "shot_penalty"}
    is_shot = actions["type_name"].isin(shot_types)
    is_success = actions["result_name"].eq("success")
    actions = actions.assign(is_goal=(is_shot & is_success).astype(int))

    return (
        actions.groupby([match_col, player_col], as_index=False)["is_goal"]
        .sum()
        .rename(columns={"is_goal": "goal_count"})
    )


def build_action_type_counts(actions: pd.DataFrame, match_col: str, player_col: str) -> pd.DataFrame:
    if "type_name" not in actions.columns:
        raise KeyError("Action data is missing the 'type_name' column.")

    pivot = (
        actions.groupby([match_col, player_col, "type_name"])
        .size()
        .unstack(fill_value=0)
        .add_prefix("type_")
        .add_suffix("_count")
        .reset_index()
    )
    return pivot


def build_success_rates(actions: pd.DataFrame, match_col: str, player_col: str) -> pd.DataFrame:
    required = {"type_name", "result_name"}
    missing = required - set(actions.columns)
    if missing:
        raise KeyError(f"Action data is missing required columns: {sorted(missing)}")

    action_types = ["pass", "shot", "take_on", "cross", "corner_crossed", "freekick_crossed"]
    rows = []

    grouped = actions.groupby([match_col, player_col])
    for (match_id, player_id), group in grouped:
        row = {match_col: match_id, player_col: player_id}
        for action in action_types:
            subset = group[group["type_name"] == action]
            if subset.empty:
                row[f"{action}_success_rate"] = 0.0
            else:
                success = (subset["result_name"] == "success").mean()
                row[f"{action}_success_rate"] = float(success)
        rows.append(row)

    return pd.DataFrame(rows)


def build_zone_features(actions: pd.DataFrame, match_col: str, player_col: str) -> pd.DataFrame:
    if "start_x" not in actions.columns:
        raise KeyError("Action data is missing the 'start_x' column.")

    def categorize_position(x: float) -> str:
        if pd.isna(x):
            return "unknown"
        if x < 35:
            return "defensive"
        if x < 70:
            return "midfield"
        return "attacking"

    temp = actions.copy()
    temp["start_zone"] = temp["start_x"].apply(categorize_position)

    zone_actions = (
        temp.pivot_table(
            index=[match_col, player_col],
            columns="start_zone",
            values="period_id",
            aggfunc="count",
            fill_value=0,
        )
        .add_prefix("zone_")
        .add_suffix("_actions")
        .reset_index()
    )

    totals = (
        zone_actions.get("zone_defensive_actions", 0)
        + zone_actions.get("zone_midfield_actions", 0)
        + zone_actions.get("zone_attacking_actions", 0)
    )

    def safe_ratio(numer: Iterable[float]) -> np.ndarray:
        numer = np.asarray(numer)
        return np.where(totals > 0, numer / totals, 0.0)

    zone_actions["zone_attacking_actions_ratio"] = safe_ratio(
        zone_actions.get("zone_attacking_actions", 0)
    )
    zone_actions["zone_midfield_actions_ratio"] = safe_ratio(
        zone_actions.get("zone_midfield_actions", 0)
    )
    zone_actions["zone_defensive_actions_ratio"] = safe_ratio(
        zone_actions.get("zone_defensive_actions", 0)
    )

    zone_actions = zone_actions.drop(columns=[col for col in zone_actions.columns if col.endswith("total_actions")])
    return zone_actions


def build_per_minute_features(
    match_player_stats: pd.DataFrame,
    goal_counts: pd.DataFrame,
    action_type_counts: pd.DataFrame,
    match_col: str,
    player_col: str,
) -> pd.DataFrame:
    df = match_player_stats.merge(goal_counts, on=[match_col, player_col], how="left")
    df["goal_count"] = df["goal_count"].fillna(0)

    df = df.merge(action_type_counts, on=[match_col, player_col], how="left")

    per_minute = df[[match_col, player_col, "minutes_played", "action_count", "goal_count"]].copy()
    per_minute["minutes_played"] = per_minute["minutes_played"].replace({0: np.nan})

    per_minute["action_count_per_minute"] = (
        per_minute["action_count"] / per_minute["minutes_played"]
    ).fillna(0.0)
    per_minute["goal_count_per_minute"] = (
        per_minute["goal_count"] / per_minute["minutes_played"]
    ).fillna(0.0)

    action_cols = [col for col in df.columns if col.startswith("type_") and col.endswith("_count")]
    result = per_minute[[match_col, player_col, "action_count_per_minute", "goal_count_per_minute"]].copy()

    for col in action_cols:
        values = df[col].fillna(0)
        ratio = (values / per_minute["minutes_played"]).fillna(0.0)
        result[col.replace("_count", "_count_per_minute")] = ratio

    return result


def build_offense_defense_balance(actions: pd.DataFrame, match_col: str, player_col: str) -> pd.DataFrame:
    mapping = {
        "shot": "offensive",
        "pass": "offensive",
        "cross": "offensive",
        "take_on": "offensive",
        "dribble": "offensive",
        "tackle": "defensive",
        "interception": "defensive",
        "clearance": "defensive",
    }

    temp = actions.copy()
    temp["action_type"] = temp["type_name"].map(mapping)

    balance = (
        temp.pivot_table(
            index=[match_col, player_col],
            columns="action_type",
            values="period_id",
            aggfunc="count",
            fill_value=0,
        )
        .add_prefix("type_")
        .add_suffix("_actions")
        .reset_index()
    )

    total = (
        balance.get("type_offensive_actions", 0)
        + balance.get("type_defensive_actions", 0)
    )
    balance["type_offensive_action_ratio"] = np.where(
        total > 0,
        balance.get("type_offensive_actions", 0) / total,
        0.0,
    )
    return balance


def build_pass_leads_to_shot(actions: pd.DataFrame, match_col: str, player_col: str) -> pd.DataFrame:
    required = {"type_name", "period_id", "time_seconds"}
    missing = required - set(actions.columns)
    if missing:
        raise KeyError(f"Action data is missing required columns: {sorted(missing)}")

    temp = actions.sort_values([match_col, "period_id", "time_seconds"])
    temp["next_type"] = temp.groupby(match_col)["type_name"].shift(-1)

    mask = (temp["type_name"] == "pass") & (temp["next_type"] == "shot")
    pass_to_shot = temp.loc[mask]

    return (
        pass_to_shot.groupby([match_col, player_col])
        .size()
        .reset_index(name="pass_leads_to_shot")
    )


def merge_feature_blocks(
    base: pd.DataFrame,
    blocks: Iterable[pd.DataFrame],
    match_col: str,
    player_col: str,
) -> pd.DataFrame:
    df = base.copy()
    for block in blocks:
        df = df.merge(block, on=[match_col, player_col], how="left")
    return df


def engineer_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    cfg: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    match_col, player_col = cfg.match_id_col, cfg.player_id_col

    train_df = compute_age_features(train_df)
    test_df = compute_age_features(test_df)

    relevant_actions = slice_relevant_actions(actions_df, train_df, test_df, match_col, player_col)
    relevant_actions = standardize_coordinates(relevant_actions)

    match_player_stats = build_basic_stats(relevant_actions, match_col, player_col)
    goal_counts = build_goal_counts(relevant_actions, match_col, player_col)
    action_type_counts = build_action_type_counts(relevant_actions, match_col, player_col)
    success_rates = build_success_rates(relevant_actions, match_col, player_col)
    zone_features = build_zone_features(relevant_actions, match_col, player_col)
    per_minute = build_per_minute_features(match_player_stats, goal_counts, action_type_counts, match_col, player_col)
    ad_balance = build_offense_defense_balance(relevant_actions, match_col, player_col)
    pass_leads = build_pass_leads_to_shot(relevant_actions, match_col, player_col)

    train_aug = merge_feature_blocks(
        train_df,
        [
            match_player_stats,
            goal_counts,
            action_type_counts,
            success_rates,
            zone_features,
            per_minute,
            ad_balance,
            pass_leads,
        ],
        match_col,
        player_col,
    )

    test_aug = merge_feature_blocks(
        test_df,
        [
            match_player_stats,
            goal_counts,
            action_type_counts,
            success_rates,
            zone_features,
            per_minute,
            ad_balance,
            pass_leads,
        ],
        match_col,
        player_col,
    )

    zero_fill_cols = [col for col in train_aug.columns if col.startswith("type_")]
    zero_fill_cols += ["action_count", "minutes_played", "goal_count", "pass_leads_to_shot"]
    zero_fill_cols = sorted({col for col in zero_fill_cols if col in train_aug.columns})

    for col in zero_fill_cols:
        train_aug[col] = train_aug[col].fillna(0)
        test_aug[col] = test_aug[col].fillna(0)

    return train_aug, test_aug, zero_fill_cols


# ---------------------------------------------------------------------------
# Training loop


def prepare_feature_lists(train_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    base_features = ["age", "action_count", "avg_x", "avg_y", "minutes_played", "goal_count"]
    categorical_features = [col for col in ("Comp", "Squad", "Venue") if col in train_df.columns]
    action_type_features = [col for col in train_df.columns if col.startswith("type_") and col.endswith("_count")]
    success_rate_features = [col for col in train_df.columns if col.endswith("_success_rate")]
    zone_features = [col for col in train_df.columns if col.startswith("zone_")]
    per_minute_features = [col for col in train_df.columns if col.endswith("_per_minute")]
    ad_balance_features = [
        col for col in ("type_offensive_actions", "type_defensive_actions", "type_offensive_action_ratio")
        if col in train_df.columns
    ]
    sequential_features = [col for col in ("pass_leads_to_shot",) if col in train_df.columns]

    feature_list = (
        base_features
        + categorical_features
        + action_type_features
        + success_rate_features
        + zone_features
        + per_minute_features
        + ad_balance_features
        + sequential_features
    )

    feature_list = [col for col in feature_list if col in train_df.columns]
    cat_features = categorical_features
    return feature_list, cat_features


def assign_folds(train_df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=cfg.cv_splits)
    folds = np.zeros(len(train_df), dtype=int)

    groups = train_df[cfg.cv_group_col]
    for fold_idx, (_, val_idx) in enumerate(gkf.split(train_df, groups=groups), start=1):
        folds[val_idx] = fold_idx

    result = train_df.copy()
    result["fold"] = folds
    return result


def run_training(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    categorical_features: List[str],
    cfg: ExperimentConfig,
    logger: logging.Logger,
):
    lgbm_params = cfg.lgbm_params.copy()
    train_params = cfg.lgbm_train
    num_boost_round = int(train_params.get("num_boost_round", 1000))
    early_stopping_rounds = train_params.get("early_stopping_rounds", 100)
    verbose_eval = train_params.get("verbose_eval", 100)

    feval = make_feval(cfg.eval_w_pos, cfg.eval_thresh)

    oof_preds = np.zeros(len(train_df))
    cv_scores: List[float] = []
    models: List[lgb.Booster] = []
    feature_importance = []

    for fold in range(1, cfg.cv_splits + 1):
        logger.info("Training fold %s", fold)
        trn_mask = train_df["fold"] != fold
        val_mask = train_df["fold"] == fold

        X_train = train_df.loc[trn_mask, features]
        y_train = train_df.loc[trn_mask, cfg.target_col]
        X_val = train_df.loc[val_mask, features]
        y_val = train_df.loc[val_mask, cfg.target_col]

        train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features or None)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, categorical_feature=categorical_features or None)

        callbacks = []
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(int(early_stopping_rounds)))
        if verbose_eval is not None:
            callbacks.append(lgb.log_evaluation(int(verbose_eval)))

        model = lgb.train(
            params=lgbm_params,
            train_set=train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "val"],
            feval=feval,
            num_boost_round=num_boost_round,
            callbacks=callbacks,
        )

        preds_val = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_mask] = preds_val
        score = weighted_rmse(y_val.values, preds_val, cfg.eval_w_pos, cfg.eval_thresh)
        cv_scores.append(score)
        models.append(model)

        importance = model.feature_importance(importance_type="gain")
        feature_importance.append(pd.DataFrame({
            "feature": features,
            "importance": importance,
            "fold": fold,
        }))

        logger.info("Fold %s weighted RMSE: %.5f", fold, score)

    oof_score = weighted_rmse(train_df[cfg.target_col].values, oof_preds, cfg.eval_w_pos, cfg.eval_thresh)
    logger.info("OOF weighted RMSE: %.5f (mean %.5f, std %.5f)", oof_score, np.mean(cv_scores), np.std(cv_scores))

    test_preds = np.zeros(len(test_df))
    for model in models:
        test_preds += model.predict(test_df[features], num_iteration=model.best_iteration)
    test_preds /= len(models)

    importance_df = pd.concat(feature_importance, axis=0, ignore_index=True)

    return {
        "oof_preds": oof_preds,
        "cv_scores": cv_scores,
        "oof_score": oof_score,
        "test_preds": test_preds,
        "feature_importance": importance_df,
    }


# ---------------------------------------------------------------------------
# CLI entry point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the xAG experiment locally.")
    parser.add_argument("--config", required=True, help="Path to the experiment config YAML.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where predictions/metrics will be stored.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(verbose=args.verbose)

    cfg = load_yaml_config(Path(args.config))
    ensure_files_exist(cfg)

    logger.info("Experiment ID: %s", cfg.exp_id)
    logger.info("Loading data from %s", cfg.data_dir)

    dtype_overrides = {
        cfg.match_id_col: "string",
        cfg.player_id_col: "string",
    }

    train_df = pd.read_csv(cfg.data_dir / cfg.files["match_train"], dtype=dtype_overrides)
    test_df = pd.read_csv(cfg.data_dir / cfg.files["match_test"], dtype=dtype_overrides)
    actions_df = pd.read_csv(cfg.data_dir / cfg.files["action_data"], dtype=dtype_overrides)
    submission_df = pd.read_csv(cfg.data_dir / cfg.files["sample_submission"], dtype={cfg.match_id_col: "string"})

    logger.info("Train shape: %s, Test shape: %s, Actions shape: %s", train_df.shape, test_df.shape, actions_df.shape)

    train_aug, test_aug, _ = engineer_features(train_df, test_df, actions_df, cfg)
    features, categorical_features = prepare_feature_lists(train_aug)

    logger.info("Using %d features (%d categorical)", len(features), len(categorical_features))

    train_with_folds = assign_folds(train_aug, cfg)

    results = run_training(train_with_folds, test_aug, features, categorical_features, cfg, logger)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics = {
        "exp_id": cfg.exp_id,
        "cv_scores": results["cv_scores"],
        "cv_mean": float(np.mean(results["cv_scores"])),
        "cv_std": float(np.std(results["cv_scores"])),
        "oof_score": float(results["oof_score"]),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save OOF predictions and feature importance
    oof_df = train_with_folds[[cfg.match_id_col, cfg.player_id_col, cfg.target_col, "fold"]].copy()
    oof_df["oof_pred"] = results["oof_preds"]
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    results["feature_importance"].to_csv(output_dir / "feature_importance.csv", index=False)

    # Save submission-style predictions
    submission = submission_df.copy()
    submission[cfg.target_col] = results["test_preds"]
    submission.to_csv(output_dir / f"submission_{cfg.exp_id}.csv", index=False)

    logger.info("Artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()

