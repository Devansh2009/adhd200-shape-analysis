#!/Users/devansh/opt/anaconda3/envs/adhd200/bin/python
"""
Structure-by-structure ADHD classification.
Runs the same pipeline independently for each of the 15 subcortical structures:
  CovariateResidualizer -> StandardScaler -> PCA(95%) -> LinearSVC(C=0.01)
5-fold stratified CV, 200 permutations, n_jobs=1.
Saves one row per structure to structure_results.csv.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class CovariateResidualizer(BaseEstimator, TransformerMixin):
    """OLS-project Age, Gender, site_binary out of feature matrix.
    Last n_covariates columns of X are treated as confounds and dropped."""
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

def make_pipe():
    return Pipeline([
        ("residualizer", CovariateResidualizer(n_covariates=3)),
        ("scaler",       StandardScaler()),
        ("pca",          PCA(n_components=0.95, svd_solver="full")),
        ("svc",          LinearSVC(C=0.01, dual="auto",
                                   class_weight="balanced", max_iter=5000)),
    ])

# ---------------------------------------------------------------------------
# Structures (canonical order)
# ---------------------------------------------------------------------------
STRUCTURES = [
    "L_Accu", "R_Accu",
    "L_Amyg", "R_Amyg",
    "L_Caud", "R_Caud",
    "L_Hipp", "R_Hipp",
    "L_Pall", "R_Pall",
    "L_Puta", "R_Puta",
    "L_Thal", "R_Thal",
    "BrStem",
]

N_PERMS = 200
CV      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading shape_features_matrix.csv ...")
df = pd.read_csv("/Users/devansh/adhd200_project/shape_features_matrix.csv")
print(f"  {df.shape[0]} subjects x {df.shape[1]} columns")

y = (df["DX"].astype(int) > 0).astype(int).values
print(f"  ADHD={y.sum()}  Controls={(y==0).sum()}\n")

site_binary = (df["Site"].astype(int) == 5).astype(np.float64).values
covars = np.column_stack([
    df["Age"].values.astype(np.float64),
    df["Gender"].values.astype(np.float64),
    site_binary,                              # NYU=1, Peking=0
])

# ---------------------------------------------------------------------------
# Run pipeline for each structure
# ---------------------------------------------------------------------------
results = []

for i, struct in enumerate(STRUCTURES, 1):
    feat_cols = [c for c in df.columns if c.startswith(struct + "_m")]
    X = np.hstack([df[feat_cols].values.astype(np.float64), covars])

    print(f"[{i:2d}/15] {struct:8s}  ({len(feat_cols)} modes) ", end="", flush=True)

    # Balanced accuracy
    ba = cross_val_score(make_pipe(), X, y, cv=CV,
                         scoring="balanced_accuracy", n_jobs=1)

    # ROC-AUC (via decision_function)
    auc = cross_val_score(make_pipe(), X, y, cv=CV,
                          scoring="roc_auc", n_jobs=1)

    # Permutation test (balanced accuracy)
    obs, null, pval = permutation_test_score(
        make_pipe(), X, y,
        cv=CV,
        scoring="balanced_accuracy",
        n_permutations=N_PERMS,
        n_jobs=1,
        random_state=42,
    )

    sig = "**" if pval <= 0.01 else "*" if pval <= 0.05 else "n.s."
    print(f"BA={ba.mean():.4f}±{ba.std():.4f}  "
          f"AUC={auc.mean():.4f}  "
          f"p={pval:.4f} {sig}")

    results.append({
        "structure":      struct,
        "n_modes":        len(feat_cols),
        "bal_acc_mean":   round(float(ba.mean()),  6),
        "bal_acc_std":    round(float(ba.std()),   6),
        "bal_acc_fold1":  round(float(ba[0]), 6),
        "bal_acc_fold2":  round(float(ba[1]), 6),
        "bal_acc_fold3":  round(float(ba[2]), 6),
        "bal_acc_fold4":  round(float(ba[3]), 6),
        "bal_acc_fold5":  round(float(ba[4]), 6),
        "roc_auc_mean":   round(float(auc.mean()), 6),
        "roc_auc_std":    round(float(auc.std()),  6),
        "roc_auc_fold1":  round(float(auc[0]), 6),
        "roc_auc_fold2":  round(float(auc[1]), 6),
        "roc_auc_fold3":  round(float(auc[2]), 6),
        "roc_auc_fold4":  round(float(auc[3]), 6),
        "roc_auc_fold5":  round(float(auc[4]), 6),
        "perm_pvalue":    round(float(pval),  6),
        "null_ba_median": round(float(np.median(null)), 6),
        "null_ba_95pct":  round(float(np.percentile(null, 95)), 6),
    })

# ---------------------------------------------------------------------------
# Save and summarise
# ---------------------------------------------------------------------------
out = pd.DataFrame(results)
out.to_csv("/Users/devansh/adhd200_project/structure_results.csv", index=False)

print("\n" + "="*65)
print("Saved: structure_results.csv\n")
print(f"{'Structure':10s}  {'Bal Acc':>10s}  {'ROC-AUC':>9s}  {'p-value':>9s}  Sig")
print("-"*55)
for _, r in out.iterrows():
    sig = "**" if r.perm_pvalue <= 0.01 else "*" if r.perm_pvalue <= 0.05 else ""
    print(f"{r.structure:10s}  {r.bal_acc_mean:>8.4f}    {r.roc_auc_mean:>7.4f}  "
          f"{r.perm_pvalue:>9.4f}  {sig}")
