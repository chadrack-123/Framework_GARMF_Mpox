# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np, pandas as pd

def _counts(y: pd.Series):
    # returns dict {label: count}
    vals, cnts = np.unique(y, return_counts=True)
    return dict(zip(vals, cnts))

def simple_random_oversample_any_minority(
    X: pd.DataFrame, y: pd.Series, target_ratio: float = 1.0, seed: int = 1337
):
    """
    Randomly oversample the true minority class(es) until:
      minority_count ≈ target_ratio * majority_count
    target_ratio=1.0 -> full balance (equal counts).
    """
    rng = np.random.default_rng(seed)
    counts = _counts(y)
    if len(counts) <= 1:
        return X, y  # nothing to balance

    # identify majority and minority labels
    maj_label = max(counts, key=counts.get)
    maj_n = counts[maj_label]
    target_min_n = int(np.ceil(target_ratio * maj_n))

    X_aug, y_aug = X.copy(), y.copy()
    for lbl, n in counts.items():
        if lbl == maj_label:
            continue
        if n >= target_min_n:
            continue  # already at/above target
        need = target_min_n - n
        idx = np.where(y == lbl)[0]
        sampled = rng.choice(idx, size=need, replace=True)
        X_aug = pd.concat([X_aug, X.iloc[sampled]], ignore_index=True)
        y_aug = pd.concat([y_aug, y.iloc[sampled]], ignore_index=True)

    return X_aug, y_aug

def smote_or_oversample(
    X: pd.DataFrame,
    y: pd.Series,
    multiplier: float = 2.0,        # kept for backward-compat
    seed: int = 1337,
    balance_to_max: bool = False,   # NEW: if True, ignore multiplier and fully balance
    target_ratio: float | None = None  # NEW: minority/majority ratio; 1.0 == full balance
):
    """
    If balance_to_max=True -> fully balance (minority up to majority).
    Else if target_ratio is given -> minority up to target_ratio * majority.
    Else fallback to legacy 'multiplier' for *positive-class* only (y==1).
    """
    # Decide strategy
    if balance_to_max or (target_ratio is not None):
        ratio = 1.0 if balance_to_max else float(target_ratio)
        try:
            from imblearn.over_sampling import SMOTE
            counts = _counts(y)
            if len(counts) <= 1:
                return X, y

            # Build sampling_strategy dict: for every minority class, desired count
            maj_n = max(counts.values())
            desired = {lbl: max(n, int(np.ceil(ratio * maj_n)))
                       for lbl, n in counts.items()}
            # imblearn expects only minority targets in dict; remove majority if equal
            maj_label = max(counts, key=counts.get)
            if desired.get(maj_label, maj_n) == maj_n:
                desired.pop(maj_label, None)

            if not desired:
                return X, y  # already balanced to desired ratio

            # k_neighbors must be < minority count; pick safely
            min_minority_n = min(n for lbl, n in counts.items() if lbl != maj_label)
            k = max(1, min(5, min_minority_n - 1))
            sm = SMOTE(random_state=seed, k_neighbors=k, sampling_strategy=desired)
            X_res, y_res = sm.fit_resample(X, y)
            print(f"[DEBUG] SMOTE balance → before={counts} after={_counts(y_res)} target_ratio={ratio}")
            # Return as pandas
            return (pd.DataFrame(X_res, columns=X.columns)
                    if not isinstance(X, pd.DataFrame) else X_res, 
                    pd.Series(y_res) if not isinstance(y, pd.Series) else y_res)
        except Exception as e:
            print(f"[DEBUG] SMOTE unavailable ({e}); falling back to random oversample")
            X2, y2 = simple_random_oversample_any_minority(X, y, target_ratio=ratio, seed=seed)
            print(f"[DEBUG] Random balance → before={_counts(y)} after={_counts(y2)} target_ratio={ratio}")
            return X2, y2

    # -------- Legacy path (kept so your old profiles still run) --------
    # Only oversamples the positive class (1) by 'multiplier' — not class-aware.
    # Prefer the balanced modes above for real class balancing.
    try:
        from imblearn.over_sampling import SMOTE
        pos_n = int((y == 1).sum()); neg_n = int((y == 0).sum())
        k = max(1, min(5, pos_n - 1))
        sm = SMOTE(random_state=seed, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X, y)
        target_pos = int(multiplier * pos_n)
        cur_pos = int((y_res == 1).sum())
        if target_pos > cur_pos:
            more = target_pos - cur_pos
            pos_idx = np.where(y_res == 1)[0]
            rng = np.random.default_rng(seed)
            sampled = rng.choice(pos_idx, size=more, replace=True)
            X_res = pd.concat(
                [pd.DataFrame(X_res, columns=X.columns),
                 pd.DataFrame(X_res[sampled], columns=X.columns)],
                ignore_index=True
            )
            y_res = pd.concat([pd.Series(y_res), pd.Series(y_res[sampled])], ignore_index=True)
        print(f"[DEBUG] Legacy synth → before=[{neg_n} {pos_n}] after={_counts(y_res)} multiplier={multiplier}")
        return X_res, y_res
    except Exception:
        # simple positive-only oversample
        rng = np.random.default_rng(seed)
        pos_idx = np.where(y == 1)[0]
        if len(pos_idx) == 0 or multiplier <= 1.0:  # nothing to do
            return X, y
        n_new = int((multiplier - 1.0) * len(pos_idx))
        sampled = rng.choice(pos_idx, size=n_new, replace=True)
        X_aug = pd.concat([X, X.iloc[sampled]], ignore_index=True)
        y_aug = pd.concat([y, y.iloc[sampled]], ignore_index=True)
        print(f"[DEBUG] Legacy random → before={_counts(y)} after={_counts(y_aug)} multiplier={multiplier}")
        return X_aug, y_aug

