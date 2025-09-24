# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np, pandas as pd

def _as_df(X):
    # Ensure we keep column names after imblearn returns numpy arrays
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError("X must be a pandas DataFrame")

def _as_sr(y):
    if isinstance(y, pd.Series):
        return y
    raise TypeError("y must be a pandas Series")

def simple_random_oversample(X: pd.DataFrame, y: pd.Series, target_min_count: int, seed: int=1337):
    """Randomly oversample the current minority class up to target_min_count."""
    rng = np.random.default_rng(seed)
    X = _as_df(X); y = _as_sr(y)

    # Identify classes
    vc = y.value_counts()
    minority_label = vc.idxmin()
    minority_idx = np.where(y.values == minority_label)[0]

    need = int(target_min_count) - len(minority_idx)
    if need <= 0:
        return X, y

    sampled = rng.choice(minority_idx, size=need, replace=True)
    X_aug = pd.concat([X, X.iloc[sampled]], ignore_index=True)
    y_aug = pd.concat([y, y.iloc[sampled]], ignore_index=True)
    return X_aug, y_aug

def smote_or_oversample(
    X: pd.DataFrame,
    y: pd.Series,
    multiplier: float = 2.0,
    seed: int = 1337,
    *,
    balance_to_max: bool = False,
    target_ratio: float | None = None,
):
    """
    Oversample the minority class using SMOTE when available; else simple random oversampling.

    Parameters
    ----------
    multiplier : float
        If neither balance_to_max nor target_ratio is set, grow the minority count by this factor.
        e.g., 2.0 doubles the minority count.
    balance_to_max : bool
        If True, make the minority count equal to the majority count (full balance).
    target_ratio : float | None
        Desired minority/majority ratio (e.g., 0.7 -> minority will be 70% of majority size).

    Notes
    -----
    - Minority class is detected automatically from y’s value_counts().
    - Works when the *positive* class is the majority as well (we just balance the minority).
    """
    X = _as_df(X); y = _as_sr(y)

    vc = y.value_counts()
    if len(vc) < 2:
        # Nothing to resample if only one class
        return X, y

    # Identify minority/majority
    minority_label = vc.idxmin()
    majority_label = vc.idxmax()
    n_min = int(vc.loc[minority_label])
    n_maj = int(vc.loc[majority_label])

    # Decide target minority count
    if balance_to_max:
        target_min = n_maj
    elif target_ratio is not None:
        target_min = int(round(target_ratio * n_maj))
    else:
        target_min = int(round(multiplier * n_min))

    # No need to upsample if target not larger than current minority
    if target_min <= n_min:
        return X, y

    # Try SMOTE first
    try:
        from imblearn.over_sampling import SMOTE
        # sampling_strategy expects the *final* minority count
        sampling_strategy = {minority_label: target_min}
        k = min(5, max(1, n_min - 1))
        sm = SMOTE(random_state=seed, k_neighbors=k, sampling_strategy=sampling_strategy)
        X_res, y_res = sm.fit_resample(X, y)
        # Keep as DataFrame/Series with original column names
        X_res = pd.DataFrame(X_res, columns=X.columns)
        y_res = pd.Series(y_res, name=y.name)
        return X_res, y_res
    except Exception:
        # Fallback: random oversample to target_min
        return simple_random_oversample(X, y, target_min_count=target_min, seed=seed)

