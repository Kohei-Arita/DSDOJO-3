"""
Non-negative stacking utilities for model blending.

Uses non-negative least squares to prevent negative predictions and
reduce over-cancellation in zero-abundant regions.
"""

import numpy as np
from scipy.optimize import nnls


def fit_nnls_stack(
    features_oof: np.ndarray,
    y_true: np.ndarray,
    sample_weight: np.ndarray = None,
) -> np.ndarray:
    """
    Fit non-negative stacking coefficients via NNLS.
    
    Parameters
    ----------
    features_oof : np.ndarray
        OOF predictions matrix, shape (n_samples, n_models)
        Each column is one base model's OOF predictions
    y_true : np.ndarray
        True target values, shape (n_samples,)
    sample_weight : np.ndarray, optional
        Sample weights for weighted MSE objective
        If provided, shape (n_samples,)
    
    Returns
    -------
    np.ndarray
        Non-negative coefficients, shape (n_models,)
        No normalization applied - coefficients are raw NNLS solution
    
    Notes
    -----
    - Solves: min ||sqrt(w) * (y - X @ coef)||^2  s.t. coef >= 0
    - No bias term (intercept) to preserve zero predictions
    - Coefficients are NOT normalized (sum may != 1)
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    
    # Apply sqrt(weight) to both sides for weighted least squares
    sqrt_w = np.sqrt(sample_weight).reshape(-1, 1)
    X_weighted = features_oof * sqrt_w
    y_weighted = y_true * sqrt_w.ravel()
    
    # Solve NNLS
    coefs, residual = nnls(X_weighted, y_weighted)
    
    return coefs


def predict_nnls_stack(
    features: np.ndarray,
    coefs: np.ndarray,
) -> np.ndarray:
    """
    Apply NNLS stacking coefficients to features.
    
    Parameters
    ----------
    features : np.ndarray
        Prediction matrix, shape (n_samples, n_models)
    coefs : np.ndarray
        NNLS coefficients from fit_nnls_stack()
    
    Returns
    -------
    np.ndarray
        Blended predictions, shape (n_samples,)
    """
    return features @ coefs


def create_stack_features(
    base_preds: dict,
    threshold: float = 0.1,
    include_positive_excess: bool = True,
) -> tuple:
    """
    Create stacking feature matrix from base model predictions.
    
    Parameters
    ----------
    base_preds : dict
        Dictionary of {model_name: predictions_array}
        Each array has shape (n_samples,)
    threshold : float
        Threshold for positive region features (default: 0.1)
    include_positive_excess : bool
        If True, add ReLU(pred - threshold) features for each model
    
    Returns
    -------
    tuple of (feature_matrix, feature_names)
        - feature_matrix: np.ndarray, shape (n_samples, n_features)
        - feature_names: list of str, feature names for logging
    
    Examples
    --------
    >>> base_preds = {'lgbm': oof_lgbm, 'catboost': oof_cb}
    >>> X_stack, names = create_stack_features(base_preds)
    >>> # X_stack columns: [lgbm, catboost, lgbm_pos, catboost_pos]
    """
    feature_list = []
    feature_names = []
    
    # Base predictions
    for name, preds in sorted(base_preds.items()):
        feature_list.append(preds.reshape(-1, 1))
        feature_names.append(name)
    
    # Positive excess features: ReLU(pred - threshold)
    if include_positive_excess:
        for name, preds in sorted(base_preds.items()):
            pos_excess = np.maximum(preds - threshold, 0)
            feature_list.append(pos_excess.reshape(-1, 1))
            feature_names.append(f"{name}_pos")
    
    feature_matrix = np.hstack(feature_list)
    
    return feature_matrix, feature_names

