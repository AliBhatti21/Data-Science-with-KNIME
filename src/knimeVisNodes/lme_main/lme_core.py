from __future__ import annotations
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Column-naming scheme (agreed naming: target-prefixed, subject-level cols
# unprefixed)
# --------------------------------------------------------------------------

def target_col(target: str, suffix: str) -> str:
    """e.g. target_col('FS383', 'ra_int') -> 'FS383_ra_int'"""
    return f"{target}_{suffix}"


PER_TARGET_SUFFIXES = ("ra_int", "ra_slope", "dev_bl", "dev_last", "max_change")
SUBJECT_LEVEL_COLUMNS = ("age_mean", "age_max")


# --------------------------------------------------------------------------
# Validation — runs before any model fitting. Every failure raises with a
# message naming the offending subject IDs, per the "error, don't silently
# drop" decision.
# --------------------------------------------------------------------------

class LMEDataError(ValueError):
    """Raised for any data-quality problem that must stop the node."""


def validate_and_clean(df, id_col: str, time_col: str, target_cols: list[str]):
    """
    Applies, in order:
      1. Missing-value check on time_col and every target_col -> hard error.
      2. Duplicate-timepoint resolution per subject -> keep first row.
      3. Single-visit-subject check -> hard error.

    Returns a cleaned copy of df. Never called with quadratic terms in v1.
    """
    import pandas as pd  # deferred import

    df = df.copy()

    # --- 1. Missing values --------------------------------------------
    check_cols = [time_col] + list(target_cols)
    missing_mask = df[check_cols].isna().any(axis=1)
    if missing_mask.any():
        bad_ids = sorted(df.loc[missing_mask, id_col].astype(str).unique().tolist())
        raise LMEDataError(
            f"Missing values found in columns {check_cols} for subject(s): "
            f"{bad_ids[:20]}{' ...' if len(bad_ids) > 20 else ''}. "
            f"Remove or impute these values upstream before running this node "
            f"— missing values are not silently dropped."
        )

    # --- 2. Duplicate timepoints: keep first row per (subject, time) ---
    df = df.sort_index(kind="stable")
    dup_mask = df.duplicated(subset=[id_col, time_col], keep="first")
    if dup_mask.any():
        df = df.loc[~dup_mask].copy()

    # --- 3. Single-visit subjects --------------------------------------
    visit_counts = df.groupby(id_col)[time_col].count()
    single_visit_ids = visit_counts[visit_counts < 2].index.astype(str).tolist()
    if single_visit_ids:
        raise LMEDataError(
            f"Subject(s) with only a single visit cannot be fit into a "
            f"mixed-effects model (no within-subject slope information): "
            f"{single_visit_ids[:20]}{' ...' if len(single_visit_ids) > 20 else ''}. "
            f"Remove these subjects upstream, or exclude them via a Row Filter "
            f"before this node."
        )

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Fitted-model container — this is what gets serialized into the Learner's
# output PortObject. One instance per target column.
# --------------------------------------------------------------------------

@dataclass
class FittedTargetModel:
    target: str
    id_col: str
    time_col: str
    fe_intercept: float
    fe_slope: float
    cov_re: list  # 2x2 covariance matrix of random effects, as nested lists
    scale: float  # residual variance
    training_ids: list = field(default_factory=list)
    converged: bool = True


def fit_linear_lme(df, id_col: str, time_col: str, target: str) -> "tuple[FittedTargetModel, object]":
    """
    Fits a single linear LME (random intercept + random slope on time_col)
    for one target column. Returns (FittedTargetModel, statsmodels result)
    so the Learner can also pull per-subject random effects for the
    training-features table without refitting.
    """
    import statsmodels.formula.api as smf  # deferred import

    work = df[[id_col, time_col, target]].rename(
        columns={time_col: "_time", target: "_target"}
    )
    work[id_col] = work[id_col].astype(str)

    model = smf.mixedlm(
        "_target ~ _time",
        work,
        groups=work[id_col],
        re_formula="~_time",
    )
    result = model.fit(method="lbfgs", reml=False, maxiter=5000)

    fitted = FittedTargetModel(
        target=target,
        id_col=id_col,
        time_col=time_col,
        fe_intercept=float(result.fe_params["Intercept"]),
        fe_slope=float(result.fe_params["_time"]),
        cov_re=result.cov_re.values.tolist(),
        scale=float(result.scale),
        training_ids=sorted(work[id_col].unique().tolist()),
        converged=result.converged if hasattr(result, "converged") else True,
    )
    return fitted, result


# --------------------------------------------------------------------------
# Per-subject feature extraction — shared by Learner (measured random
# effects) and Apply (empirical-Bayes estimated random effects). Everything
# except the random-effect pair is computed identically in both nodes.
# --------------------------------------------------------------------------

def _fixed_effect_at(fitted: FittedTargetModel, t: float) -> float:
    return fitted.fe_intercept + fitted.fe_slope * t


def extract_deterministic_features(df, id_col: str, time_col: str, target: str, fitted: FittedTargetModel) -> dict:
    """
    Computes, for every subject in df, the features that do NOT depend on
    random effects: dev_bl, dev_last, max_change (exactly as originally
    scripted — magnitude only, no direction), age_mean, age_max.

    Returns {subject_id: {feature_name: value}}.
    """
    out = {}
    for sid, sub in df.groupby(id_col):
        sub = sub.sort_values(time_col)
        t_min, t_max = sub[time_col].min(), sub[time_col].max()

        fe_min = _fixed_effect_at(fitted, t_min)
        fe_max = _fixed_effect_at(fitted, t_max)

        val_bl = sub.loc[sub[time_col] == t_min, target].iloc[0]
        val_last = sub.loc[sub[time_col] == t_max, target].iloc[0]

        dev_bl = val_bl - fe_min
        dev_last = val_last - fe_max

        # max_change: kept EXACTLY as originally scripted — magnitude only.
        if t_max != t_min:
            max_change = (sub[target].max() - sub[target].min()) / (t_max - t_min)
        else:
            max_change = 0.0

        out[str(sid)] = {
            "dev_bl": float(dev_bl),
            "dev_last": float(dev_last),
            "max_change": float(max_change),
            "age_mean": float(sub[time_col].mean()),
            "age_max": float(t_max),
        }
    return out


def measured_random_effects(id_col: str, result) -> dict:
    """
    Learner-side only: pulls the actually-fitted random intercept/slope
    for each training subject straight from the statsmodels result —
    no estimation involved, these were measured during fitting.
    Returns {subject_id: {"ra_int": ..., "ra_slope": ...}}.
    """
    re = result.random_effects
    out = {}
    for sid, vals in re.items():
        out[str(sid)] = {
            "ra_int": float(vals.get("Group", vals.iloc[0])),
            "ra_slope": float(vals.get("_time", vals.iloc[1] if len(vals) > 1 else 0.0)),
        }
    return out


def empirical_bayes_random_effects(df, id_col: str, time_col: str, target: str, fitted: FittedTargetModel) -> dict:
    """
    Apply-node only: estimates a shrinkage (empirical-Bayes / BLUP) random
    intercept + slope for subjects the training model never saw, using only
    (a) the fixed effects, (b) the random-effect covariance and residual
    variance saved in the Learner's model, and (c) this subject's own visits.
    No information from other test subjects, no refitting, no leakage.

    Returns {subject_id: {"ra_int": ..., "ra_slope": ...}}.
    """
    import numpy as np  # deferred import

    D = np.array(fitted.cov_re)          # 2x2 random-effects covariance
    sigma2 = fitted.scale                # residual variance

    out = {}
    for sid, sub in df.groupby(id_col):
        sub = sub.sort_values(time_col)
        t = sub[time_col].to_numpy(dtype=float)
        y = sub[target].to_numpy(dtype=float)
        n = len(t)

        Z = np.column_stack([np.ones(n), t])              # design for random effects
        resid = y - (fitted.fe_intercept + fitted.fe_slope * t)

        # Standard BLUP: b_hat = D Z' (Z D Z' + sigma2 I)^-1 * resid
        V = Z @ D @ Z.T + sigma2 * np.eye(n)
        try:
            b_hat = D @ Z.T @ np.linalg.solve(V, resid)
        except np.linalg.LinAlgError:
            b_hat = np.zeros(2)

        out[str(sid)] = {"ra_int": float(b_hat[0]), "ra_slope": float(b_hat[1])}

    return out


def check_age_extrapolation(df, id_col: str, time_col: str, train_min: float, train_max: float) -> list[str]:
    """
    Apply-node only: flags (does not fail) subjects whose visits fall
    outside the training time range. Returns a list of human-readable
    warning strings, one per affected subject.
    """
    warnings = []
    for sid, sub in df.groupby(id_col):
        lo, hi = sub[time_col].min(), sub[time_col].max()
        if lo < train_min or hi > train_max:
            warnings.append(
                f"Subject {sid}: time range [{lo:.2f}, {hi:.2f}] falls outside "
                f"the training range [{train_min:.2f}, {train_max:.2f}] — "
                f"features for this subject involve extrapolation."
            )
    return warnings
