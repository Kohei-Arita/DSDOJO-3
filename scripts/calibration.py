"""
Post-hoc calibration utilities for positive region only.

Applies isotonic regression only to predictions above threshold to avoid
distorting the negative region (where zeros are abundant).
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic_positive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.1,
    pos_weight: float = 5.0,
) -> IsotonicRegression:
    """
    Fit isotonic regression on positive region only.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values (OOF)
    y_pred : np.ndarray
        Predicted values (OOF)
    threshold : float
        Boundary for positive region (default: 0.1)
    pos_weight : float
        Weight multiplier for true positives (default: 5.0)
        Sample weight = (y_true >= threshold) * (pos_weight - 1) + 1
    
    Returns
    -------
    IsotonicRegression
        Fitted isotonic model (call .predict() to calibrate)
    
    Notes
    -----
    - Only samples where y_pred >= threshold are used for fitting
    - Sample weights align with wRMSE: 5.0 for y_true >= threshold, else 1.0
    - out_of_bounds='clip' ensures predictions stay within training range
    """
    # Select positive region samples
    mask = (y_pred >= threshold)
    
    if mask.sum() == 0:
        raise ValueError(f"No predictions >= {threshold} found for calibration")
    
    y_true_pos = y_true[mask]
    y_pred_pos = y_pred[mask]
    
    # Compute sample weights (align with wRMSE)
    weights = np.where(y_true_pos >= threshold, pos_weight, 1.0)
    
    # Fit isotonic regression
    iso = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip"
    )
    iso.fit(y_pred_pos, y_true_pos, sample_weight=weights)
    
    return iso


def apply_isotonic_positive(
    y_pred: np.ndarray,
    iso_model: IsotonicRegression,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Apply isotonic calibration to positive region only.
    
    Parameters
    ----------
    y_pred : np.ndarray
        Raw predictions to calibrate
    iso_model : IsotonicRegression
        Fitted isotonic model from fit_isotonic_positive()
    threshold : float
        Boundary for positive region (default: 0.1)
    
    Returns
    -------
    np.ndarray
        Calibrated predictions:
        - y_pred < threshold: unchanged
        - y_pred >= threshold: iso_model.predict(y_pred)
    """
    y_calib = y_pred.copy()
    mask = (y_pred >= threshold)
    
    if mask.sum() > 0:
        y_calib[mask] = iso_model.predict(y_pred[mask])
    
    return y_calib


def optimize_isotonic_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_metric,
    threshold_candidates: np.ndarray = None,
    pos_weight: float = 5.0,
) -> tuple:
    """
    Find optimal threshold for isotonic calibration by grid search.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values (OOF)
    y_pred : np.ndarray
        Predicted values (OOF)
    eval_metric : callable
        Evaluation function: eval_metric(y_true, y_pred) -> float (lower is better)
    threshold_candidates : np.ndarray, optional
        Threshold values to try (default: np.arange(0.05, 0.20, 0.01))
    pos_weight : float
        Weight multiplier for true positives (default: 5.0)
    
    Returns
    -------
    tuple of (best_threshold, best_score, iso_model)
        - best_threshold: optimal threshold
        - best_score: evaluation metric at best threshold
        - iso_model: fitted isotonic model at best threshold
    """
    if threshold_candidates is None:
        threshold_candidates = np.arange(0.05, 0.20, 0.01)
    
    best_score = float("inf")
    best_threshold = None
    best_model = None
    
    for thresh in threshold_candidates:
        try:
            iso = fit_isotonic_positive(y_true, y_pred, threshold=thresh, pos_weight=pos_weight)
            y_calib = apply_isotonic_positive(y_pred, iso, threshold=thresh)
            score = eval_metric(y_true, y_calib)
            
            if score < best_score:
                best_score = score
                best_threshold = thresh
                best_model = iso
        except ValueError:
            # Skip if no samples in positive region
            continue
    
    if best_model is None:
        raise ValueError("No valid threshold found for isotonic calibration")
    
    return best_threshold, best_score, best_model

