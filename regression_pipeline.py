#!/usr/bin/env python
"""
Nested cross-validated Ridge regression of subcortical shape features on
ADHD symptom scores (ADHD Index, Inattentive, Hyper/Impulsive).

Design
------
Input: raw 5040 shape-mode coefficients from shape_features_matrix.csv.
  We intentionally do NOT start from shape_features_pca.csv: that file's PCA
  was fit on all 365 subjects and would leak test-fold information into every
  outer CV split.  PCA is refit from scratch inside each fold via the Pipeline.

Per-fold pipeline (applied to training data only; transform applied to test):
  1. CovariateResidualizer  — OLS-project out Age, Gender, Site from features.
  2. StandardScaler         — unit-variance scaling before PCA.
  3. PCA(n_components=0.95) — keeps enough components for 95% variance (fitted
                              on this fold's training residuals; count varies).
  4. RidgeCV(cv=5)          — Ridge with 5-fold inner CV for alpha selection.
                              This is the inner loop of the true nested CV.

Cross-validation
----------------
Outer loop : 5-fold, stratified by DX label so ADHD/control ratio is balanced
             across folds.  A custom splitter (StratifiedByGroupCV) is used
             because sklearn's StratifiedKFold errors on continuous y.
Inner loop : embedded inside RidgeCV — alpha selected via 5-fold CV on the
             outer training split only; test fold is never seen during fitting.
Permutation: 1000 shuffles of y with the same nested CV; p-value = fraction of
             null R² values that meet or exceed the observed R².

Output: regression_results.csv  (one row per target variable).

Usage
-----
    conda activate adhd200
    python regression_pipeline.py [--n-permutations 1000] [--n-jobs -1]
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Custom CV splitter: stratify by a pre-fixed label array, not by y
# ---------------------------------------------------------------------------

class StratifiedByGroupCV:
    """K-fold CV stratified by an externally supplied discrete label array.

    sklearn's StratifiedKFold validates that y is discrete and raises an error
    when y is a continuous regression target.  This wrapper captures the
    stratification labels (e.g. DX) at construction time and uses them in
    split(), letting cross_val_score / permutation_test_score receive
    continuous y without modification.
    """

    def __init__(self, strat_labels, n_splits=5, shuffle=True, random_state=42):
        self._strat  = np.asarray(strat_labels)
        self._inner  = StratifiedKFold(n_splits=n_splits, shuffle=shuffle,
                                       random_state=random_state)

    def split(self, X, y=None, groups=None):
        return self._inner.split(X, self._strat)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._inner.get_n_splits()


# ---------------------------------------------------------------------------
# Custom transformer: OLS-residualize features against covariates
# ---------------------------------------------------------------------------

class CovariateResidualizer(BaseEstimator, TransformerMixin):
    """Project out linear covariate effects from a feature matrix.

    Expects the input matrix X = [shape_features | covariates], where the last
    ``n_covariates`` columns are the confounds (Age, Gender, Site).

    fit()      : estimates OLS coefficients  Xfeatures ~ intercept + covariates
                 on the training data only.
    transform(): subtracts predicted covariate contribution from features and
                 returns the residuals (covariate columns are dropped).
    """

    def __init__(self, n_covariates=3):
        self.n_covariates = n_covariates

    def fit(self, X, y=None):
        Xf    = X[:, :-self.n_covariates]
        C     = X[:,  -self.n_covariates:]
        C_aug = np.column_stack([np.ones(len(C)), C])   # add intercept column
        self.coef_, _, _, _ = np.linalg.lstsq(C_aug, Xf, rcond=None)
        return self

    def transform(self, X):
        Xf    = X[:, :-self.n_covariates]
        C     = X[:,  -self.n_covariates:]
        C_aug = np.column_stack([np.ones(len(C)), C])
        return Xf - C_aug @ self.coef_


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MISSING_SENTINEL = -999.0
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0]
TARGETS = ["ADHD Index", "Inattentive", "Hyper/Impulsive"]
N_OUTER_FOLDS = 5


def build_pipeline():
    return Pipeline([
        ("residualizer", CovariateResidualizer(n_covariates=3)),
        ("scaler",       StandardScaler()),
        ("pca",          PCA(n_components=0.95, svd_solver="full")),
        ("ridge",        RidgeCV(alphas=ALPHAS, cv=5)),
    ])


# ---------------------------------------------------------------------------
# Per-target regression runner
# ---------------------------------------------------------------------------

def run_target(name, X_full, y_raw, dx_raw, n_permutations, n_jobs):
    """Run nested CV + permutation test for one target variable.

    Parameters
    ----------
    name           : display name of the target column
    X_full         : (N, n_features+3) array — raw shape features with
                     [Age, Gender, site_binary] appended as the last 3 cols
    y_raw          : pd.Series of raw target values (may contain NaN / -999)
    dx_raw         : pd.Series of DX labels for CV stratification
    n_permutations : number of label permutations for the null distribution
    n_jobs         : parallel workers (-1 = all cores)

    Returns
    -------
    dict  with keys for the results CSV
    """
    print(f"\n{'='*60}")
    print(f"Target: {name}")

    # Keep only subjects with a valid (non-missing, non-sentinel) score.
    y_num = pd.to_numeric(y_raw, errors="coerce")
    valid = y_num.notna() & (y_num != MISSING_SENTINEL)
    X  = X_full[valid.values]
    y  = y_num[valid].values.astype(np.float64)
    dx = dx_raw[valid].astype(str).values
    n  = len(y)
    print(f"  N={n}  y∈[{y.min():.1f}, {y.max():.1f}]")

    # Outer CV stratified by DX (built here so its label array matches the
    # subject subset that passed the validity filter above).
    outer_cv = StratifiedByGroupCV(
        strat_labels=dx,
        n_splits=N_OUTER_FOLDS,
        shuffle=True,
        random_state=42,
    )

    pipe = build_pipeline()

    # --- Outer CV: per-fold R² ---
    t0 = time.time()
    cv_scores = cross_val_score(
        pipe, X, y,
        cv=outer_cv,
        scoring="r2",
        n_jobs=n_jobs,
    )
    print(f"  CV R² per fold : {np.round(cv_scores, 4)}")
    print(f"  Mean R²={cv_scores.mean():.4f}  SD={cv_scores.std():.4f}  "
          f"[{time.time()-t0:.1f}s]")

    # --- Permutation test: null distribution + p-value ---
    print(f"  Running {n_permutations} permutations (n_jobs={n_jobs}) …")
    t0 = time.time()
    actual_r2, perm_r2, pvalue = permutation_test_score(
        pipe, X, y,
        cv=outer_cv,
        n_permutations=n_permutations,
        scoring="r2",
        n_jobs=n_jobs,
        random_state=42,
    )
    print(f"  p={pvalue:.4f}  "
          f"(observed R²={actual_r2:.4f}, "
          f"null median={np.median(perm_r2):.4f}, "
          f"null 95th={np.percentile(perm_r2, 95):.4f})  "
          f"[{time.time()-t0:.1f}s]")

    return {
        "target":           name,
        "n_subjects":       n,
        "cv_r2_mean":       round(float(cv_scores.mean()), 6),
        "cv_r2_std":        round(float(cv_scores.std()),  6),
        "cv_r2_fold1":      round(float(cv_scores[0]), 6),
        "cv_r2_fold2":      round(float(cv_scores[1]), 6),
        "cv_r2_fold3":      round(float(cv_scores[2]), 6),
        "cv_r2_fold4":      round(float(cv_scores[3]), 6),
        "cv_r2_fold5":      round(float(cv_scores[4]), 6),
        "perm_pvalue":      round(float(pvalue), 6),
        "null_r2_median":   round(float(np.median(perm_r2)), 6),
        "null_r2_95pct":    round(float(np.percentile(perm_r2, 95)), 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    home    = os.path.expanduser("~")
    project = os.path.join(home, "adhd200_project")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--input",
        default=os.path.join(project, "shape_features_matrix.csv"),
        help="raw shape feature matrix (5040 features) — see note in docstring "
             "about why we don't use shape_features_pca.csv here",
    )
    ap.add_argument(
        "--output",
        default=os.path.join(project, "regression_results.csv"),
    )
    ap.add_argument(
        "--n-permutations", type=int, default=1000,
        help="permutations for p-value estimation (default 1000)",
    )
    ap.add_argument(
        "--n-jobs", type=int, default=-1,
        help="parallel workers for permutation test (-1 = all cores)",
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  {df.shape[0]} subjects × {df.shape[1]} columns")

    feat_cols = [c for c in df.columns
                 if "_m" in c and c.split("_m")[-1].isdigit()]
    print(f"  {len(feat_cols)} raw shape-mode features")

    for t in TARGETS:
        assert t in df.columns, f"Target column '{t}' not in CSV"

    # ------------------------------------------------------------------
    # Build feature + covariate matrix
    # Covariates appended as the last 3 columns so CovariateResidualizer
    # can split them off by index without needing column names.
    # Site is recoded to binary (NYU=1, Peking=0) before appending.
    # ------------------------------------------------------------------
    site_binary = (df["Site"].astype(int) == 5).astype(np.float64).values

    X_full = np.hstack([
        df[feat_cols].values.astype(np.float64),          # 5040 shape features
        df["Age"].values.astype(np.float64).reshape(-1, 1),
        df["Gender"].values.astype(np.float64).reshape(-1, 1),
        site_binary.reshape(-1, 1),                        # NYU=1, Peking=0
    ])
    print(f"  Combined matrix: {X_full.shape}  "
          f"(last 3 cols = Age, Gender, site_binary)")

    # ------------------------------------------------------------------
    # Run pipeline for each target
    # ------------------------------------------------------------------
    results = []
    for target in TARGETS:
        rec = run_target(
            name=target,
            X_full=X_full,
            y_raw=df[target],
            dx_raw=df["DX"],
            n_permutations=args.n_permutations,
            n_jobs=args.n_jobs,
        )
        results.append(rec)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print(f"Saved: {args.output}\n")
    print(out_df[["target", "n_subjects",
                  "cv_r2_mean", "cv_r2_std", "perm_pvalue"]].to_string(index=False))


if __name__ == "__main__":
    main()
