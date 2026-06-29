#!/Users/devansh/opt/anaconda3/envs/adhd200/bin/python
"""
Two improved whole-brain classification analyses, compared to the baseline
(whole-brain LinearSVC: BA=0.596, AUC=0.607, p=0.005).

  Analysis 1 — "No Hippocampus"
    Same pipeline as baseline (CovariateResidualizer -> StandardScaler ->
    PCA(95%) -> LinearSVC C=0.01) but DROP L_Hipp + R_Hipp shape columns,
    since both scored below chance in the structure-by-structure analysis.

  Analysis 2 — "RBF-SVM (tuned)"
    Full feature set (all 15 structures).  Replace LinearSVC with an RBF SVC
    whose C and gamma are tuned by an INNER 3-fold GridSearchCV.  The
    GridSearchCV is the final pipeline step, so PCA is still refit once per
    OUTER fold (no leakage) and tuning happens only on PCA-reduced data.
      C     in [0.1, 1, 10]
      gamma in ['scale', 'auto']

Both: 5-fold stratified outer CV, balanced accuracy + ROC-AUC, and a
200-permutation p-value.  n_jobs=1 throughout (macOS loky overhead).

Saves -> improved_classification_results.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score, permutation_test_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

# ---------------------------------------------------------------------------
# Covariate residualizer (last n_covariates columns of X are confounds)
# ---------------------------------------------------------------------------
class CovariateResidualizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_covariates=3):
        self.n_covariates = n_covariates
    def fit(self, X, y=None):
        Xf    = X[:, :-self.n_covariates]
        C_aug = np.column_stack([np.ones(len(X)), X[:, -self.n_covariates:]])
        self.coef_, _, _, _ = np.linalg.lstsq(C_aug, Xf, rcond=None)
        return self
    def transform(self, X):
        Xf    = X[:, :-self.n_covariates]
        C_aug = np.column_stack([np.ones(len(X)), X[:, -self.n_covariates:]])
        return Xf - C_aug @ self.coef_

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_PERMS  = 200
CV       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
BASELINE = {"name": "Whole-brain LinearSVC (baseline)",
            "bal_acc_mean": 0.596, "roc_auc_mean": 0.607, "perm_pvalue": 0.005}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading shape_features_matrix.csv ...")
df = pd.read_csv("/Users/devansh/adhd200_project/shape_features_matrix.csv")
print(f"  {df.shape[0]} subjects x {df.shape[1]} columns")

all_feat_cols = [c for c in df.columns if "_m" in c and c.split("_m")[-1].isdigit()]
y = (df["DX"].astype(int) > 0).astype(int).values
print(f"  ADHD={y.sum()}  Controls={(y==0).sum()}")

site_binary = (df["Site"].astype(int) == 5).astype(np.float64).values  # NYU=1, Peking=0
covars = np.column_stack([
    df["Age"].values.astype(np.float64),
    df["Gender"].values.astype(np.float64),
    site_binary,
])

def build_X(feat_cols):
    return np.hstack([df[feat_cols].values.astype(np.float64), covars])

def run_analysis(name, X, final_estimator):
    """Run BA, ROC-AUC, and a permutation test for one configuration."""
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    print(f"  feature+covariate matrix: {X.shape}")

    def make_pipe():
        return Pipeline([
            ("residualizer", CovariateResidualizer(n_covariates=3)),
            ("scaler",       StandardScaler()),
            ("pca",          PCA(n_components=0.95, svd_solver="full")),
            ("clf",          final_estimator),
        ])

    print("  5-fold CV (balanced accuracy) ...", flush=True)
    ba = cross_val_score(make_pipe(), X, y, cv=CV,
                         scoring="balanced_accuracy", n_jobs=1)
    print(f"    per fold: {np.round(ba, 4)}   mean={ba.mean():.4f}  SD={ba.std():.4f}")

    print("  5-fold CV (ROC-AUC) ...", flush=True)
    auc = cross_val_score(make_pipe(), X, y, cv=CV,
                          scoring="roc_auc", n_jobs=1)
    print(f"    per fold: {np.round(auc, 4)}   mean={auc.mean():.4f}  SD={auc.std():.4f}")

    print(f"  {N_PERMS} permutations (n_jobs=1) ...", flush=True)
    obs, null, pval = permutation_test_score(
        make_pipe(), X, y, cv=CV,
        scoring="balanced_accuracy",
        n_permutations=N_PERMS, n_jobs=1, random_state=42,
    )
    print(f"    observed BA={obs:.4f}  null median={np.median(null):.4f}"
          f"  95th pct={np.percentile(null, 95):.4f}  p={pval:.4f}")

    return {
        "analysis":       name,
        "n_features":     X.shape[1] - 3,
        "bal_acc_mean":   round(float(ba.mean()),  6),
        "bal_acc_std":    round(float(ba.std()),   6),
        "bal_acc_fold1":  round(float(ba[0]), 6),
        "bal_acc_fold2":  round(float(ba[1]), 6),
        "bal_acc_fold3":  round(float(ba[2]), 6),
        "bal_acc_fold4":  round(float(ba[3]), 6),
        "bal_acc_fold5":  round(float(ba[4]), 6),
        "roc_auc_mean":   round(float(auc.mean()), 6),
        "roc_auc_std":    round(float(auc.std()),  6),
        "perm_pvalue":    round(float(pval),  6),
        "null_ba_median": round(float(np.median(null)), 6),
        "null_ba_95pct":  round(float(np.percentile(null, 95)), 6),
    }

# ---------------------------------------------------------------------------
# Analysis 1 — drop hippocampus
# ---------------------------------------------------------------------------
nohip_cols = [c for c in all_feat_cols
              if not (c.startswith("L_Hipp") or c.startswith("R_Hipp"))]
print(f"\nAnalysis 1 features: {len(nohip_cols)} "
      f"(dropped {len(all_feat_cols) - len(nohip_cols)} hippocampus modes)")
res1 = run_analysis(
    "No-Hippocampus LinearSVC (C=0.01)",
    build_X(nohip_cols),
    LinearSVC(C=0.01, dual="auto", class_weight="balanced", max_iter=5000),
)

# ---------------------------------------------------------------------------
# Analysis 2 — RBF SVM with inner GridSearchCV
# ---------------------------------------------------------------------------
inner_svc = GridSearchCV(
    SVC(kernel="rbf", class_weight="balanced"),
    param_grid={"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    cv=3, scoring="balanced_accuracy", n_jobs=1,
)
res2 = run_analysis(
    "Whole-brain RBF-SVM (inner 3-fold GridSearchCV)",
    build_X(all_feat_cols),
    inner_svc,
)

# ---------------------------------------------------------------------------
# Save + compare
# ---------------------------------------------------------------------------
out = pd.DataFrame([res1, res2])
out.to_csv("/Users/devansh/adhd200_project/improved_classification_results.csv", index=False)

print("\n" + "="*70)
print("Saved: improved_classification_results.csv\n")
print(f"{'Analysis':45s} {'Bal Acc':>9s} {'ROC-AUC':>9s} {'p':>8s}")
print("-"*74)
print(f"{BASELINE['name']:45s} {BASELINE['bal_acc_mean']:>9.4f} "
      f"{BASELINE['roc_auc_mean']:>9.4f} {BASELINE['perm_pvalue']:>8.4f}")
for r in (res1, res2):
    dba = r["bal_acc_mean"] - BASELINE["bal_acc_mean"]
    print(f"{r['analysis']:45s} {r['bal_acc_mean']:>9.4f} "
          f"{r['roc_auc_mean']:>9.4f} {r['perm_pvalue']:>8.4f}   "
          f"(ΔBA={dba:+.4f} vs baseline)")
