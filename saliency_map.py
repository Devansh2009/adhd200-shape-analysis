#!/Users/devansh/opt/anaconda3/envs/adhd200/bin/python
"""
Right caudate shape mode saliency map for ADHD vs. control classification.

Fits the full pipeline (CovariateResidualizer → StandardScaler → PCA(95%) →
LinearSVC C=0.01) on all 365 subjects using the 336 R_Caud shape modes, then
maps the SVM decision-function coefficients back to the original mode space.

Weight interpretation
---------------------
  w = coef_ @ pca.components_    (shape: 336)

  This is the weight of each shape mode in *standardized* (unit-variance) space
  — the natural space for comparing modes because each mode is scaled to equal
  variance before PCA.  Dividing by scaler.scale_ would re-expand to raw mode
  amplitudes, but 250/336 high-frequency modes have scale ≈ 0 (near-zero
  natural variance), making that division numerically meaningless.

  Positive w_i : when the caudate deforms along mode i in the positive
                 direction → model votes ADHD.
  Negative w_i : same deformation direction → model votes Control.

Output
------
  figures/caudate_saliency.png — two-panel figure:
    Top    : modes sorted by |weight|  (saliency ranking)
    Bottom : modes in natural order 0–335  (frequency-domain view)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT = os.path.expanduser("~/adhd200_project")
FIG_DIR = os.path.join(PROJECT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_MODES  = 336
N_LABEL  = 12    # label this many top modes in each panel

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading shape_features_matrix.csv ...")
df = pd.read_csv(os.path.join(PROJECT, "shape_features_matrix.csv"))
print(f"  {df.shape[0]} subjects")

feat_cols = [c for c in df.columns if c.startswith("R_Caud_m")]
assert len(feat_cols) == N_MODES, f"Expected {N_MODES} R_Caud columns, got {len(feat_cols)}"

y = (df["DX"].astype(int) > 0).astype(int).values
print(f"  ADHD={y.sum()}  Controls={(y==0).sum()}")

site_binary = (df["Site"].astype(int) == 5).astype(np.float64).values
X = np.hstack([
    df[feat_cols].values.astype(np.float64),
    df["Age"].values.reshape(-1, 1).astype(np.float64),
    df["Gender"].values.reshape(-1, 1).astype(np.float64),
    site_binary.reshape(-1, 1),
])

# ---------------------------------------------------------------------------
# Pipeline
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

pipe = Pipeline([
    ("residualizer", CovariateResidualizer(n_covariates=3)),
    ("scaler",       StandardScaler()),
    ("pca",          PCA(n_components=0.95, svd_solver="full")),
    ("svc",          LinearSVC(C=0.01, dual="auto",
                               class_weight="balanced", max_iter=5000)),
])

print("\nFitting pipeline on all 365 subjects ...")
pipe.fit(X, y)

n_pca = pipe["pca"].n_components_
print(f"  PCA kept {n_pca} components (95% variance)")
print(f"  SVM coef_ shape: {pipe['svc'].coef_.shape}")

# ---------------------------------------------------------------------------
# Map coefficients back to mode space
# ---------------------------------------------------------------------------
# w[i] = contribution of standardised mode i to the SVM decision function.
# Positive → ADHD; Negative → Control.
svm_coef = pipe["svc"].coef_.flatten()         # (n_pca,)
pca_comp = pipe["pca"].components_             # (n_pca, 336)
w        = svm_coef @ pca_comp                 # (336,)  — standardised-space weights

mode_idx = np.arange(N_MODES)                  # 0 … 335

# Sorted by |weight| for the saliency-rank panel
rank_order = np.argsort(np.abs(w))[::-1]       # most important first
w_sorted   = w[rank_order]
idx_sorted = mode_idx[rank_order]

print(f"\nWeight range: [{w.min():.6f}, {w.max():.6f}]")
print(f"Top {N_LABEL} modes by |weight|:")
for r in range(N_LABEL):
    i = idx_sorted[r]
    print(f"  #{r+1:2d}  R_Caud_m{i:03d}  w={w[i]:+.6f}  "
          f"({'ADHD ↑' if w[i] > 0 else 'Ctrl ↑'})")

# Variance explained by retained PCA components
var_explained = pipe["pca"].explained_variance_ratio_.cumsum()[-1]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 9))
fig.patch.set_facecolor("white")

ADHD_COL = "#C44E52"   # red  — ADHD-associated (positive weight)
CTRL_COL = "#4C72B0"   # blue — control-associated (negative weight)

# ── Panel A: sorted by |weight| ─────────────────────────────────────────────
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_facecolor("#F8F8F8")

colors_sorted = [ADHD_COL if v > 0 else CTRL_COL for v in w_sorted]
x_pos = np.arange(N_MODES)
ax1.bar(x_pos, w_sorted, width=1.0, color=colors_sorted,
        linewidth=0, alpha=0.85)
ax1.axhline(0, color="black", linewidth=0.8, zorder=3)

# Annotate the top N_LABEL modes
for r in range(N_LABEL):
    val   = w_sorted[r]
    midx  = idx_sorted[r]
    yoff  = 0.003 if val >= 0 else -0.003
    va    = "bottom" if val >= 0 else "top"
    ax1.text(r, val + yoff, f"m{midx:03d}",
             ha="center", va=va, fontsize=7.5, fontweight="bold",
             color="black", rotation=90)

# Legend patches
import matplotlib.patches as mpatches
adhd_patch = mpatches.Patch(color=ADHD_COL, alpha=0.85, label="Positive weight → ADHD-associated")
ctrl_patch = mpatches.Patch(color=CTRL_COL, alpha=0.85, label="Negative weight → Control-associated")
ax1.legend(handles=[adhd_patch, ctrl_patch], fontsize=9.5,
           loc="upper right", framealpha=0.9)

ax1.set_xlim(-1, N_MODES)
ax1.set_xlabel("Mode rank (1 = highest |weight|)", fontsize=11)
ax1.set_ylabel("Standardised weight\n(coef ᵀ · PCA components)", fontsize=10)
ax1.set_title(
    f"(A)  Right Caudate — Saliency Ranking  |  "
    f"LinearSVC C=0.01, N=365, PCA {n_pca} components ({var_explained*100:.1f}% variance)",
    fontsize=11, pad=8, loc="left",
)
ax1.spines[["top", "right"]].set_visible(False)
ax1.tick_params(axis="x", labelsize=9)
ax1.tick_params(axis="y", labelsize=9)
ax1.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax1.xaxis.set_minor_locator(mticker.MultipleLocator(10))
ax1.grid(axis="y", color="white", linewidth=1.0, zorder=0)

# ── Panel B: natural mode order (0 → 335) ───────────────────────────────────
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_facecolor("#F8F8F8")

colors_natural = [ADHD_COL if v > 0 else CTRL_COL for v in w]
ax2.bar(mode_idx, w, width=1.0, color=colors_natural,
        linewidth=0, alpha=0.85)
ax2.axhline(0, color="black", linewidth=0.8, zorder=3)

# Annotate top N_LABEL modes in their natural positions
annotated = set()
for r in range(N_LABEL):
    midx = idx_sorted[r]
    val  = w[midx]
    if midx in annotated:
        continue
    annotated.add(midx)
    yoff = 0.003 if val >= 0 else -0.003
    va   = "bottom" if val >= 0 else "top"
    ax2.text(midx, val + yoff, f"m{midx:03d}",
             ha="center", va=va, fontsize=7.5, fontweight="bold",
             color="black", rotation=90)

# Shade the low-frequency region (modes 0–49)
ax2.axvspan(-0.5, 49.5, alpha=0.07, color="gold", zorder=0,
            label="Low-frequency modes (0–49)")
ax2.legend(fontsize=9.5, loc="upper right", framealpha=0.9)

ax2.set_xlim(-1, N_MODES)
ax2.set_xlabel("Shape mode index (0 = lowest frequency)", fontsize=11)
ax2.set_ylabel("Standardised weight\n(coef ᵀ · PCA components)", fontsize=10)
ax2.set_title(
    "(B)  Same weights in natural mode order — "
    "discriminative signal concentrated in low-frequency modes",
    fontsize=11, pad=8, loc="left",
)
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(axis="x", labelsize=9)
ax2.tick_params(axis="y", labelsize=9)
ax2.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax2.xaxis.set_minor_locator(mticker.MultipleLocator(10))
ax2.grid(axis="y", color="white", linewidth=1.0, zorder=0)

plt.tight_layout(h_pad=2.5)

out_path = os.path.join(FIG_DIR, "caudate_saliency.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")
