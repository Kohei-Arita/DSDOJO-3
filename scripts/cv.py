"""
Cross-validation utilities with stratified group folding.

This module provides enhanced CV splitting that maintains target distribution
across folds while preventing data leakage at the group (match_id) level.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_stratified_group_folds(
    df: pd.DataFrame,
    y_col: str = "xAG",
    group_col: str = "match_id",
    threshold: float = 0.1,
    n_splits: int = 5,
    n_bins: int = 5,
    seed: int = 42,
) -> pd.Series:
    """
    Create stratified group-based CV folds.
    
    Stratifies by match-level positive rate bins while keeping match_id intact
    within folds to prevent leakage. This helps stabilize weighted RMSE across folds.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe with target and group columns
    y_col : str
        Target column name (default: 'xAG')
    group_col : str
        Group column name (default: 'match_id')
    threshold : float
        Threshold for positive class (default: 0.1 for wRMSE weight boundary)
    n_splits : int
        Number of CV folds (default: 5)
    n_bins : int
        Number of stratification bins for positive rate (default: 5)
        Will auto-reduce if fewer unique bins exist
    seed : int
        Random state for reproducibility (default: 42)
    
    Returns
    -------
    pd.Series
        Fold assignments (0-indexed) aligned with input df
    
    Notes
    -----
    - Computes match-level positive rate: mean(y >= threshold) per match_id
    - Bins these rates into n_bins quantiles
    - Uses StratifiedGroupKFold with bins as stratification target
    - If binning fails (too few unique matches), falls back to fewer bins
    """
    # Compute match-level positive rate
    match_pos_rate = (
        df.groupby(group_col)[y_col]
        .apply(lambda v: (v >= threshold).mean())
    )
    
    # Map back to all rows
    row_rates = df[group_col].map(match_pos_rate).values
    
    # Create bins (handle edge cases where unique rates < n_bins)
    unique_rates = np.unique(row_rates)
    actual_n_bins = min(n_bins, len(unique_rates))
    
    if actual_n_bins < 2:
        # Fallback: no stratification possible, treat all as same stratum
        print(f"警告: ユニークな正例率が{len(unique_rates)}種しかないため、層化なしでグループ分割します")
        bins = np.zeros(len(df), dtype=int)
    else:
        # Use quantiles to create bins
        quantiles = np.linspace(0, 1, actual_n_bins + 1)
        bin_edges = np.quantile(row_rates, quantiles)
        # digitize is 1-indexed, shift to 0-indexed
        bins = np.digitize(row_rates, bin_edges[1:-1], right=False)
    
    # Apply StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )
    
    fold_array = np.zeros(len(df), dtype=int)
    groups = df[group_col].values
    
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df, y=bins, groups=groups)):
        fold_array[val_idx] = fold_idx
    
    return pd.Series(fold_array, index=df.index, name="fold")


def evaluate_fold_balance(
    df: pd.DataFrame,
    fold_col: str = "fold",
    y_col: str = "xAG",
    threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Evaluate balance of positive examples across folds.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe with fold assignments
    fold_col : str
        Fold column name
    y_col : str
        Target column name
    threshold : float
        Threshold for positive class
    
    Returns
    -------
    pd.DataFrame
        Summary statistics per fold:
        - fold: fold index
        - n_samples: total samples
        - n_positive: count of y >= threshold
        - pos_rate: ratio of positive samples
        - mean_y: average target value
        - std_y: target standard deviation
    """
    stats = []
    for fold in sorted(df[fold_col].unique()):
        fold_df = df[df[fold_col] == fold]
        y_vals = fold_df[y_col].values
        n_pos = (y_vals >= threshold).sum()
        
        stats.append({
            "fold": int(fold),
            "n_samples": len(fold_df),
            "n_positive": int(n_pos),
            "pos_rate": n_pos / len(fold_df),
            "mean_y": float(y_vals.mean()),
            "std_y": float(y_vals.std()),
        })
    
    return pd.DataFrame(stats)

