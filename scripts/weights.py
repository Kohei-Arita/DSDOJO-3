"""
Sample weighting schemes for training stability.

Provides both step-function and smooth weighting for wRMSE-aligned objectives.
"""

import numpy as np


def make_sample_weight(
    y: np.ndarray,
    scheme: str = "step",
    threshold: float = 0.1,
    pos_weight: float = 5.0,
    tau: float = 0.03,
) -> np.ndarray:
    """
    Generate sample weights for training.
    
    Parameters
    ----------
    y : np.ndarray
        Target values
    scheme : str
        Weighting scheme:
        - 'step': step function (default, matches eval metric)
          w = pos_weight if y >= threshold else 1.0
        - 'smooth': smooth sigmoid transition
          w = 1 + (pos_weight - 1) * sigmoid((y - threshold) / tau)
    threshold : float
        Threshold for positive class (default: 0.1)
    pos_weight : float
        Weight for positive class (default: 5.0)
    tau : float
        Temperature for smooth scheme (default: 0.03)
        Smaller tau → sharper transition (closer to step)
    
    Returns
    -------
    np.ndarray
        Sample weights, shape same as y
    
    Notes
    -----
    - 'step' scheme: exact match to evaluation metric
    - 'smooth' scheme: reduces boundary instability, monotonic with step
    - Both schemes have same weight ratio at extremes (1.0 vs pos_weight)
    """
    y = np.asarray(y, dtype=float)
    
    if scheme == "step":
        return np.where(y >= threshold, pos_weight, 1.0)
    
    elif scheme == "smooth":
        # Sigmoid smooth transition
        # At y=threshold: weight ≈ (1 + pos_weight) / 2
        # Far below: → 1.0, far above: → pos_weight
        sigmoid_val = 1.0 / (1.0 + np.exp(-(y - threshold) / tau))
        return 1.0 + (pos_weight - 1.0) * sigmoid_val
    
    else:
        raise ValueError(f"Unknown scheme: {scheme}. Use 'step' or 'smooth'.")


def evaluate_weight_statistics(
    y: np.ndarray,
    scheme: str = "step",
    threshold: float = 0.1,
    pos_weight: float = 5.0,
    tau: float = 0.03,
) -> dict:
    """
    Compute weight distribution statistics for a given scheme.
    
    Parameters
    ----------
    Same as make_sample_weight()
    
    Returns
    -------
    dict
        Statistics including:
        - mean_weight: average weight
        - std_weight: weight standard deviation
        - min_weight: minimum weight
        - max_weight: maximum weight
        - n_positive: count of y >= threshold
        - pos_rate: ratio of positive samples
        - effective_n: effective sample size (sum(w)^2 / sum(w^2))
    """
    weights = make_sample_weight(y, scheme, threshold, pos_weight, tau)
    n_pos = (y >= threshold).sum()
    
    # Effective sample size (Kish's formula)
    sum_w = weights.sum()
    sum_w2 = (weights ** 2).sum()
    effective_n = (sum_w ** 2) / sum_w2 if sum_w2 > 0 else 0
    
    return {
        "mean_weight": float(weights.mean()),
        "std_weight": float(weights.std()),
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
        "n_positive": int(n_pos),
        "pos_rate": float(n_pos / len(y)),
        "effective_n": float(effective_n),
        "total_samples": len(y),
    }


def compare_weight_schemes(
    y: np.ndarray,
    threshold: float = 0.1,
    pos_weight: float = 5.0,
    tau_values: list = None,
) -> dict:
    """
    Compare step vs smooth weighting schemes.
    
    Parameters
    ----------
    y : np.ndarray
        Target values
    threshold : float
        Threshold for positive class (default: 0.1)
    pos_weight : float
        Weight for positive class (default: 5.0)
    tau_values : list, optional
        List of tau values to try for smooth scheme
        Default: [0.02, 0.03, 0.05]
    
    Returns
    -------
    dict
        Nested dict: {scheme_name: statistics_dict}
    """
    if tau_values is None:
        tau_values = [0.02, 0.03, 0.05]
    
    results = {}
    
    # Step scheme
    results["step"] = evaluate_weight_statistics(
        y, scheme="step", threshold=threshold, pos_weight=pos_weight
    )
    
    # Smooth schemes with different tau
    for tau in tau_values:
        scheme_name = f"smooth_tau{tau:.3f}"
        results[scheme_name] = evaluate_weight_statistics(
            y, scheme="smooth", threshold=threshold, pos_weight=pos_weight, tau=tau
        )
    
    return results

