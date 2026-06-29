#!/usr/bin/env python
"""
Case-control classification: ADHD (DX>0) vs. controls (DX=0).
Pipeline per fold: CovariateResidualizer -> StandardScaler -> PCA(95%) -> LinearSVC(C=0.01)
5-fold stratified CV, 200 permutations, n_jobs=1 throughout.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, permutation_test_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class CovariateResidualizer(BaseEstimator, TransformerMixin):
    """OLS-project Age, Gender, Site out of feature matrix.
    Expects X = [shape_features | Age | Gender | site_binary] — last 3 cols are covariates.
    """
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


# --- Load data ---
print("Loading shape_features_matrix.csv ...")
df = pd.read_csv("/Users/devansh/adhd200_project/shape_features_matrix.csv")
print(f"  {df.shape[0]} subjects x {df.shape[1]} columns")

feat_cols = [c for c in df.columns if "_m" in c and c.split("_m")[-1].isdigit()]
print(f"  {len(feat_cols)} shape features")

y = (df["DX"].astype(int) > 0).astype(int).values
print(f"  ADHD={y.sum()}  Controls={(y==0).sum()}")

site_binary = (df["Site"].astype(int) == 5).astype(np.float64).values  # NYU=1, Peking=0
X_full = np.hstack([
    df[feat_cols].values.astype(np.float64),
    df["Age"].values.reshape(-1, 1).astype(np.float64),
    df["Gender"].values.reshape(-1, 1).astype(np.float64),
    site_binary.reshape(-1, 1),
])
print(f"  Feature+covariate matrix: {X_full.shape}")

# --- Pipeline ---
pipe = Pipeline([
    ("residualizer", CovariateResidualizer(n_covariates=3)),
    ("scaler",       StandardScaler()),
    ("pca",          PCA(n_components=0.95, svd_solver="full")),
    ("svc",          LinearSVC(C=0.01, dual="auto",
                               class_weight="balanced", max_iter=5000)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Balanced accuracy ---
print("\nRunning 5-fold CV (balanced accuracy) ...")
ba = cross_val_score(pipe, X_full, y, cv=cv, scoring="balanced_accuracy", n_jobs=1)
print(f"  Per fold: {np.round(ba, 4)}")
print(f"  Mean={ba.mean():.4f}  SD={ba.std():.4f}")

# --- ROC-AUC ---
print("\nRunning 5-fold CV (ROC-AUC) ...")
auc = cross_val_score(pipe, X_full, y, cv=cv, scoring="roc_auc", n_jobs=1)
print(f"  Per fold: {np.round(auc, 4)}")
print(f"  Mean={auc.mean():.4f}  SD={auc.std():.4f}")

# --- Permutation test ---
print("\nRunning 200 permutations (n_jobs=1) ...")
obs, null, pval = permutation_test_score(
    pipe, X_full, y,
    cv=cv,
    scoring="balanced_accuracy",
    n_permutations=200,
    n_jobs=1,
    random_state=42,
)
print(f"  Observed BA={obs:.4f}  null median={np.median(null):.4f}"
      f"  95th pct={np.percentile(null, 95):.4f}  p={pval:.4f}")

# --- Save ---
result = pd.DataFrame([{
    "n_adhd":         int(y.sum()),
    "n_control":      int((y == 0).sum()),
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
}])
result.to_csv("/Users/devansh/adhd200_project/classification_results.csv", index=False)
print("\nSaved: classification_results.csv")
print(f"  Balanced accuracy : {ba.mean():.4f} ± {ba.std():.4f}")
print(f"  ROC-AUC           : {auc.mean():.4f} ± {auc.std():.4f}")
print(f"  Permutation p     : {pval:.4f}")
