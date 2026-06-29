#!/usr/bin/env python
"""
Per-structure PCA reduction of subcortical shape features.

Reads shape_features_matrix.csv (output of extract_shape_features.py), runs
PCA independently on each of the 15 subcortical structures retaining enough
components to explain 95% of variance, concatenates the reduced components,
and writes shape_features_pca.csv.

A numeric 'Site' covariate (Peking=1, NYU=5) is carried through from the
original phenotypics column.  All other phenotypic/metadata columns are
preserved in the output.

PCA is fit on all retained subjects (no held-out set).  If you later split
into train/test, fit PCA only on the training fold and transform the test
fold using the saved components (see --save-models).

Usage
-----
    conda activate adhd200
    python run_pca_reduction.py [--input ...] [--output ...] [--variance 0.95]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

STRUCTURES = [
    "L_Accu", "L_Amyg", "L_Caud", "L_Hipp", "L_Pall", "L_Puta", "L_Thal",
    "R_Accu", "R_Amyg", "R_Caud", "R_Hipp", "R_Pall", "R_Puta", "R_Thal",
    "BrStem",
]

# Phenotypic / metadata columns to carry through to the output CSV.
PHENO_COLS = [
    "SubjectID", "SiteName", "ID_int", "ScanDir ID",
    "Site",           # numeric site code: Peking=1, NYU=5 (scanner covariate)
    "Gender", "Age", "Handedness",
    "DX", "Secondary Dx",
    "ADHD Measure", "ADHD Index", "Inattentive", "Hyper/Impulsive",
    "IQ Measure", "Verbal IQ", "Performance IQ", "Full2 IQ", "Full4 IQ",
    "Med Status", "QC_Athena", "QC_NIAK",
]


def reduce_structure(X, struct, variance_threshold, verbose=True):
    """Fit PCA on X (n_subjects x 336) and return transformed array plus metadata."""
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    X_r = pca.fit_transform(X)
    n_components = X_r.shape[1]
    cum_var = pca.explained_variance_ratio_.cumsum()[-1]
    if verbose:
        print(f"  {struct:8s}: {X.shape[1]:3d} -> {n_components:3d} PCs "
              f"({cum_var*100:.1f}% variance explained)")
    cols = [f"{struct}_pc{i:03d}" for i in range(n_components)]
    return X_r, cols, pca


def main():
    home = os.path.expanduser("~")
    project = os.path.join(home, "adhd200_project")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.path.join(project, "shape_features_matrix.csv"),
                    help="input CSV from extract_shape_features.py")
    ap.add_argument("--output", default=os.path.join(project, "shape_features_pca.csv"),
                    help="output CSV with PCA-reduced features")
    ap.add_argument("--variance", type=float, default=0.95,
                    help="fraction of variance to retain per structure (default 0.95)")
    ap.add_argument("--save-models", action="store_true",
                    help="pickle per-structure PCA objects to <project>/pca_models/")
    args = ap.parse_args()

    print(f"Reading: {args.input}")
    df = pd.read_csv(args.input, dtype=str)
    print(f"  loaded {df.shape[0]} subjects x {df.shape[1]} columns")

    # Determine which phenotypics columns are actually present.
    pheno_cols_present = [c for c in PHENO_COLS if c in df.columns]
    missing_pheno = [c for c in PHENO_COLS if c not in df.columns]
    if missing_pheno:
        print(f"  NOTE: phenotypics columns not found and skipped: {missing_pheno}",
              file=sys.stderr)
    meta = df[pheno_cols_present].copy()

    # Validate that the Site column carries the expected Peking=1 / NYU=5 coding.
    if "Site" in meta.columns:
        site_counts = meta["Site"].value_counts().to_dict()
        print(f"  Site covariate distribution: {site_counts}")
        unexpected = set(meta["Site"].unique()) - {"1", "5", 1, 5}
        if unexpected:
            print(f"  WARNING: unexpected Site values: {unexpected}", file=sys.stderr)

    print(f"\nRunning per-structure PCA (retain {args.variance*100:.0f}% variance):")
    pc_blocks = []
    pc_cols_all = []
    total_raw = 0
    total_reduced = 0

    for struct in STRUCTURES:
        struct_cols = [c for c in df.columns if c.startswith(f"{struct}_m")]
        if not struct_cols:
            print(f"  WARNING: no mode columns found for {struct}", file=sys.stderr)
            continue
        total_raw += len(struct_cols)

        X = df[struct_cols].astype(np.float64).values
        X_r, pc_cols, pca_model = reduce_structure(
            X, struct, args.variance, verbose=True
        )
        pc_blocks.append(X_r)
        pc_cols_all.extend(pc_cols)
        total_reduced += len(pc_cols)

        if args.save_models:
            import pickle
            model_dir = os.path.join(os.path.dirname(args.output), "pca_models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"pca_{struct}.pkl")
            with open(model_path, "wb") as fh:
                pickle.dump(pca_model, fh)

    print(f"\n  Total: {total_raw} raw features -> {total_reduced} PCA features "
          f"({total_reduced/total_raw*100:.1f}% of original dimensionality)")

    feat_df = pd.DataFrame(np.hstack(pc_blocks), columns=pc_cols_all)

    out = pd.concat([meta.reset_index(drop=True), feat_df], axis=1)

    out.to_csv(args.output, index=False, float_format="%.6g")
    print(f"\nSaved: {args.output}")
    print(f"  shape: {out.shape[0]} subjects x {out.shape[1]} columns "
          f"({len(pc_cols_all)} PCA features + {len(pheno_cols_present)} metadata)")


if __name__ == "__main__":
    main()
