#!/Users/devansh/opt/anaconda3/envs/adhd200/bin/python
"""
Produce three publication-ready figures from the ADHD-200 shape analysis:

  Figure 1  bar_chart.png            — Balanced accuracy / R² across all three
                                        analyses with error bars (SD across folds)
                                        and permutation p-value annotations.
  Figure 2  permutation_null.png     — Whole-brain classification null distribution
                                        (200 permutations) vs. observed score.
  Figure 3  confusion_matrix.png     — Confusion matrix for the best-performing
                                        fold of the caudate-only classification.

Figures saved to ~/adhd200_project/figures/.
Figure 2 re-runs the 200-permutation test (deterministic; ~3 min, n_jobs=1).
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT  = os.path.expanduser("~/adhd200_project")
FIG_DIR  = os.path.join(PROJECT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Pipeline (identical to classification_pipeline.py / caudate_analysis.py)
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

def make_pipe():
    return Pipeline([
        ("residualizer", CovariateResidualizer(n_covariates=3)),
        ("scaler",       StandardScaler()),
        ("pca",          PCA(n_components=0.95, svd_solver="full")),
        ("svc",          LinearSVC(C=0.01, dual="auto",
                                   class_weight="balanced", max_iter=5000)),
    ])

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------------------------------------------------------
# Load result CSVs
# ---------------------------------------------------------------------------
print("Loading result CSVs ...")
reg  = pd.read_csv(os.path.join(PROJECT, "regression_results.csv"))
clf  = pd.read_csv(os.path.join(PROJECT, "classification_results.csv")).iloc[0]
caud = pd.read_csv(os.path.join(PROJECT, "caudate_results.csv")).iloc[0]

# Fold-level scores for error bars
reg_adhd_folds  = [reg.loc[reg.target=="ADHD Index",  f"cv_r2_fold{i}"].values[0] for i in range(1,6)]
clf_ba_folds    = [clf[f"bal_acc_fold{i}"] for i in range(1,6)]
caud_ba_folds   = [caud[f"bal_acc_fold{i}"] for i in range(1,6)]

# ---------------------------------------------------------------------------
# Load shape data (needed for Figures 2 and 3)
# ---------------------------------------------------------------------------
print("Loading shape_features_matrix.csv ...")
df = pd.read_csv(os.path.join(PROJECT, "shape_features_matrix.csv"))

y = (df["DX"].astype(int) > 0).astype(int).values
site_binary = (df["Site"].astype(int) == 5).astype(np.float64).values
covars = np.column_stack([
    df["Age"].values.astype(np.float64),
    df["Gender"].values.astype(np.float64),
    site_binary,
])

all_feat_cols  = [c for c in df.columns if "_m" in c and c.split("_m")[-1].isdigit()]
caud_feat_cols = [c for c in df.columns if c.startswith("L_Caud") or c.startswith("R_Caud")]

X_wb   = np.hstack([df[all_feat_cols].values.astype(np.float64),  covars])
X_caud = np.hstack([df[caud_feat_cols].values.astype(np.float64), covars])

# ============================================================================
# FIGURE 1 — Bar chart: performance across analyses
# ============================================================================
print("\n[Figure 1] Building bar chart ...")

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F8F8")

labels = [
    "Whole-brain\nRegression\n(ADHD Index  R²)",
    "Whole-brain\nClassification\n(Balanced Acc.)",
    "Caudate-only\nClassification\n(Balanced Acc.)",
]
means   = [np.mean(reg_adhd_folds), np.mean(clf_ba_folds),  np.mean(caud_ba_folds)]
sds     = [np.std(reg_adhd_folds),  np.std(clf_ba_folds),   np.std(caud_ba_folds)]
pvals   = [reg.loc[reg.target=="ADHD Index", "perm_pvalue"].values[0],
           clf["perm_pvalue"],
           caud["perm_pvalue"]]
colors  = ["#C44E52", "#4C72B0", "#55A868"]

x = np.arange(len(labels))
bars = ax.bar(x, means, yerr=sds, capsize=7, width=0.55,
              color=colors, alpha=0.85, edgecolor="black", linewidth=0.8,
              error_kw=dict(elinewidth=1.5, ecolor="black", capthick=1.5))

# Reference lines
ax.axhline(0.50, color="#E87722", linewidth=1.4, linestyle="--", zorder=0,
           label="Balanced-acc. chance (0.50)")
ax.axhline(0.00, color="#888888", linewidth=1.2, linestyle=":",  zorder=0,
           label="R² chance (0.00)")

# Significance annotations above each bar
for i, (m, s, p) in enumerate(zip(means, sds, pvals)):
    top = m + s + 0.025
    if p <= 0.01:
        sig = f"p = {p:.3f} **"
    elif p <= 0.05:
        sig = f"p = {p:.3f} *"
    else:
        sig = f"p = {p:.3f}\n(n.s.)"
    ax.text(i, top, sig, ha="center", va="bottom", fontsize=9.5,
            color="black", fontweight="bold" if p < 0.05 else "normal")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Shape Classification Performance Across Analyses\n"
             "(error bars = ±1 SD across folds, 200-permutation test)",
             fontsize=12, pad=10)
ax.set_ylim(-0.18, 0.82)
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.tick_params(axis="y", labelsize=10)
ax.grid(axis="y", color="white", linewidth=1.2)
ax.legend(fontsize=9.5, loc="upper right", framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
path1 = os.path.join(FIG_DIR, "bar_chart.png")
fig.savefig(path1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path1}")

# ============================================================================
# FIGURE 2 — Permutation null distribution (whole-brain classification)
# ============================================================================
print("\n[Figure 2] Re-running whole-brain permutation test "
      "(200 perms, n_jobs=1) ...")
obs_ba, null_ba, pval_ba = permutation_test_score(
    make_pipe(), X_wb, y,
    cv=CV,
    scoring="balanced_accuracy",
    n_permutations=200,
    n_jobs=1,
    random_state=42,
)
print(f"  Observed BA={obs_ba:.4f}  null median={np.median(null_ba):.4f}"
      f"  95th pct={np.percentile(null_ba, 95):.4f}  p={pval_ba:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F8F8")

p95 = np.percentile(null_ba, 95)
ax.hist(null_ba, bins=30, color="#4C72B0", alpha=0.6, edgecolor="white",
        linewidth=0.6, label="Null distribution (200 permutations)")

# Shade the critical region (right tail beyond 95th pct)
bin_edges = np.histogram(null_ba, bins=30)[1]
ax.hist(null_ba[null_ba >= p95], bins=bin_edges, color="#C44E52",
        alpha=0.55, edgecolor="white", linewidth=0.6,
        label=f"Top 5% of null  (≥ {p95:.3f})")

ax.axvline(obs_ba, color="#C44E52", linewidth=2.5, zorder=5,
           label=f"Observed BA = {obs_ba:.3f}")
ax.axvline(p95, color="#E87722", linewidth=1.8, linestyle="--", zorder=4,
           label=f"Null 95th pct = {p95:.3f}")
ax.axvline(0.50, color="#888888", linewidth=1.2, linestyle=":", zorder=3,
           label="Chance (0.50)")

# p-value annotation
ymax = ax.get_ylim()[1]
ax.text(obs_ba + 0.003, ymax * 0.88,
        f"p = {pval_ba:.3f}", color="#C44E52", fontsize=12, fontweight="bold",
        va="top", ha="left")

ax.set_xlabel("Balanced Accuracy", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Whole-brain Classification: Permutation Null Distribution\n"
             "(5-fold CV, LinearSVC C=0.01, 200 permutations)",
             fontsize=12, pad=10)
ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=10)
ax.grid(axis="y", color="white", linewidth=1.2)

plt.tight_layout()
path2 = os.path.join(FIG_DIR, "permutation_null.png")
fig.savefig(path2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path2}")

# ============================================================================
# FIGURE 3 — Confusion matrix for the best-performing caudate fold
# ============================================================================
print("\n[Figure 3] Fitting caudate pipeline on best fold ...")

# Best fold = fold index 0 (fold 1; BA=0.712 from caudate_results.csv)
caud_ba_per_fold = np.array(caud_ba_folds)
best_fold_idx    = int(np.argmax(caud_ba_per_fold))
best_fold_ba     = caud_ba_per_fold[best_fold_idx]
print(f"  Best fold: index {best_fold_idx}  (fold {best_fold_idx+1})  "
      f"BA={best_fold_ba:.4f}")

splits = list(CV.split(X_caud, y))
train_idx, test_idx = splits[best_fold_idx]
pipe_caud = make_pipe()
pipe_caud.fit(X_caud[train_idx], y[train_idx])
y_pred = pipe_caud.predict(X_caud[test_idx])
y_true = y[test_idx]

cm = confusion_matrix(y_true, y_pred)
# cm[i,j] = true class i predicted as class j
tn, fp, fn, tp = cm.ravel()
n_test = len(y_true)

fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("white")

im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Count", fontsize=10)

class_labels = ["Control\n(DX=0)", "ADHD\n(DX>0)"]
tick_marks = [0, 1]
ax.set_xticks(tick_marks); ax.set_xticklabels(class_labels, fontsize=11)
ax.set_yticks(tick_marks); ax.set_yticklabels(class_labels, fontsize=11)

# Cell text: count and row-normalised percentage
thresh = cm.max() / 2.0
for i in range(2):
    row_total = cm[i].sum()
    for j in range(2):
        pct = cm[i, j] / row_total * 100
        color = "white" if cm[i, j] > thresh else "black"
        ax.text(j, i, f"{cm[i,j]}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=13,
                color=color, fontweight="bold")

ax.set_xlabel("Predicted label", fontsize=12, labelpad=8)
ax.set_ylabel("True label",      fontsize=12, labelpad=8)
ax.set_title(
    f"Caudate Classification — Best Fold (Fold {best_fold_idx+1})\n"
    f"Balanced Accuracy = {best_fold_ba:.3f}  |  "
    f"n_test = {n_test}  "
    f"(ADHD={y_true.sum()}, Control={(y_true==0).sum()})",
    fontsize=10.5, pad=10,
)

# Derived stats in text box
sens = tp / (tp + fn) if (tp + fn) > 0 else 0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0
stats_text = f"Sensitivity (recall) = {sens:.3f}\nSpecificity          = {spec:.3f}"
ax.text(1.32, 0.5, stats_text, transform=ax.transAxes,
        fontsize=9.5, va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEF4FB",
                  edgecolor="#4C72B0", alpha=0.9))

plt.tight_layout()
path3 = os.path.join(FIG_DIR, "confusion_matrix.png")
fig.savefig(path3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path3}")

# ---------------------------------------------------------------------------
print(f"\nAll figures saved to {FIG_DIR}/")
print("  bar_chart.png")
print("  permutation_null.png")
print("  confusion_matrix.png")
