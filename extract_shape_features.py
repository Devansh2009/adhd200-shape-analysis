#!/usr/bin/env python
"""
Extract FSL FIRST subcortical shape coefficients (.bvars) for the ADHD-200
Peking + NYU samples, merge with phenotypics, and write a feature matrix CSV.

Pipeline
--------
1. Read every ``*_first.bvars`` file for all subjects across both sites.
2. Extract the shape-mode coefficients (the b-vars) as a per-subject feature
   vector: one block of ``nmodes`` coefficients per subcortical structure.
3. Merge with the phenotypics table, keeping only subjects with QC_NIAK == 1
   and a valid (non-missing) ADHD Index score.
4. Save the resulting subjects x (phenotypics + features) matrix as CSV.

.bvars format note
------------------
An FSL FIRST .bvars file has three ASCII header lines::

    this is a bvars file
    <model .bmv path>
    NumberOfSubjects <N>

followed, per subject, by::

    <image name> <nmodes> <binary float32 payload>

The payload holds ``nmodes`` shape-mode coefficients followed by a 4x4 pose /
registration matrix (16 float32, bottom row [0,0,0,1]). Only the first
``nmodes`` values are the shape coefficients; the pose matrix is discarded.
Empirically every structure/subject here has nmodes == 336 (the models_336
training set), i.e. 352 float32 total per file.

Usage
-----
    conda activate adhd200
    python extract_shape_features.py            # uses default paths below
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# Subcortical structures in a fixed canonical column order.
STRUCTURES = [
    "L_Accu", "L_Amyg", "L_Caud", "L_Hipp", "L_Pall", "L_Puta", "L_Thal",
    "R_Accu", "R_Amyg", "R_Caud", "R_Hipp", "R_Pall", "R_Puta", "R_Thal",
    "BrStem",
]

# site subdirectory -> (display name, phenotypics "Site" code)
SITES = [
    ("Peking", "peking", 1),
    ("NYU", "nyu", 5),
]

# Missing-data sentinel used by the ADHD-200 phenotypics for ADHD Index.
ADHD_INDEX_MISSING = -999

_ID_RE = re.compile(r"X_([0-9]+)-")


def read_bvars(path):
    """Return the shape-mode coefficients (b-vars) from a FIRST .bvars file.

    Reads the trailing binary payload as little-endian float32 and returns the
    first ``nmodes`` values (the shape coefficients), excluding the trailing
    4x4 pose matrix.
    """
    with open(path, "rb") as fh:
        raw = fh.read()

    # Skip the three ASCII header lines.
    nl1 = raw.index(b"\n")
    nl2 = raw.index(b"\n", nl1 + 1)
    nl3 = raw.index(b"\n", nl2 + 1)
    rest = raw[nl3 + 1:]

    # Per-subject record: "<image name> <nmodes> <binary float32 ...>"
    sp1 = rest.index(b" ")              # end of image name
    sp2 = rest.index(b" ", sp1 + 1)     # end of nmodes token
    nmodes = int(rest[sp1 + 1:sp2])
    payload = rest[sp2 + 1:]

    # Read exactly nmodes float32 from the start of the payload; this is robust
    # to the trailing pose matrix and the file's terminating newline.
    coeffs = np.frombuffer(payload[: nmodes * 4], dtype="<f4").astype(np.float64)
    if coeffs.size != nmodes:
        raise ValueError(
            f"{path}: expected {nmodes} coefficients, got {coeffs.size}"
        )
    return coeffs


def list_subject_ids(site_dir):
    """Return sorted unique subject IDs (as found in filenames) for a site dir."""
    ids = set()
    for f in glob.glob(os.path.join(site_dir, "*_first.bvars")):
        m = _ID_RE.search(os.path.basename(f))
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def build_feature_matrix(first_root):
    """Read all subjects/structures and return (feature_df, nmodes).

    feature_df columns: SubjectID, Site, SiteCode, ID_int, <feature columns>.
    Feature columns are named "<STRUCT>_m<NNN>".
    """
    # Determine nmodes from the first available file so column names are exact.
    nmodes = None
    meta_rows = []          # (file_id, site_name, site_code, int_id)
    feature_rows = []       # 1-D arrays of length n_struct * nmodes
    n_total = n_skipped = 0

    for site_name, subdir, site_code in SITES:
        site_dir = os.path.join(first_root, subdir)
        if not os.path.isdir(site_dir):
            print(f"WARNING: site directory not found: {site_dir}", file=sys.stderr)
            continue

        subj_ids = list_subject_ids(site_dir)
        print(f"  {site_name:6s}: {len(subj_ids)} subjects in {site_dir}")

        for sid in subj_ids:
            n_total += 1
            blocks = []
            complete = True
            for struct in STRUCTURES:
                path = os.path.join(site_dir, f"X_{sid}-{struct}_first.bvars")
                if not os.path.exists(path):
                    print(f"    WARNING: missing {os.path.basename(path)} "
                          f"-> skipping subject {sid}", file=sys.stderr)
                    complete = False
                    break
                coeffs = read_bvars(path)
                if nmodes is None:
                    nmodes = coeffs.size
                if coeffs.size != nmodes:
                    print(f"    WARNING: {os.path.basename(path)} has "
                          f"{coeffs.size} modes (expected {nmodes}) "
                          f"-> skipping subject {sid}", file=sys.stderr)
                    complete = False
                    break
                blocks.append(coeffs)

            if not complete:
                n_skipped += 1
                continue

            feature_rows.append(np.concatenate(blocks))
            meta_rows.append((sid, site_name, site_code, int(sid)))

    if not feature_rows:
        raise RuntimeError("No subjects with a complete set of structures found.")

    feat_cols = [f"{struct}_m{i:03d}" for struct in STRUCTURES for i in range(nmodes)]
    feat = pd.DataFrame(np.vstack(feature_rows), columns=feat_cols)
    meta = pd.DataFrame(meta_rows, columns=["SubjectID", "SiteName", "SiteCode", "ID_int"])
    out = pd.concat([meta, feat], axis=1)

    print(f"  -> {len(out)} subjects with complete data "
          f"({n_skipped} skipped of {n_total}); "
          f"{nmodes} modes x {len(STRUCTURES)} structures = {len(feat_cols)} features")
    return out, nmodes


def main():
    home = os.path.expanduser("~")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first-root", default=os.path.join(home, "adhd200_project", "first_output"),
                    help="root containing peking/ and nyu/ FIRST output")
    ap.add_argument("--phenotypics", default=os.path.join(home, "Downloads", "adhd200_preprocessed_phenotypics.tsv"),
                    help="ADHD-200 phenotypics .tsv")
    ap.add_argument("--output", default=os.path.join(home, "adhd200_project", "shape_features_matrix.csv"),
                    help="output CSV path")
    args = ap.parse_args()

    print("Reading .bvars shape coefficients...")
    feat, nmodes = build_feature_matrix(args.first_root)

    print(f"\nReading phenotypics: {args.phenotypics}")
    ph = pd.read_csv(args.phenotypics, sep="\t", dtype=str)
    ph["ID_int"] = pd.to_numeric(ph["ScanDir ID"], errors="coerce")
    ph = ph.dropna(subset=["ID_int"])
    ph["ID_int"] = ph["ID_int"].astype(int)
    dup = ph["ID_int"].duplicated().sum()
    if dup:
        print(f"  WARNING: {dup} duplicate ScanDir IDs in phenotypics; keeping first")
        ph = ph.drop_duplicates(subset="ID_int", keep="first")

    # Merge features <- phenotypics on integer subject ID.
    feat_cols = [c for c in feat.columns if c not in ("SubjectID", "SiteName", "SiteCode", "ID_int")]
    merged = feat.merge(ph, on="ID_int", how="inner")
    print(f"  merged: {len(merged)} of {len(feat)} subjects matched phenotypics")

    # Filter: QC_NIAK == 1 and a valid (non-missing) ADHD Index.
    qc = pd.to_numeric(merged["QC_NIAK"], errors="coerce")
    adhd = pd.to_numeric(merged["ADHD Index"], errors="coerce")
    keep = (qc == 1) & adhd.notna() & (adhd != ADHD_INDEX_MISSING)

    print(f"\nFiltering:")
    print(f"  failed QC_NIAK (!=1):        {int((qc != 1).sum())}")
    print(f"  invalid/missing ADHD Index:  {int((adhd.isna() | (adhd == ADHD_INDEX_MISSING)).sum())}")
    final = merged[keep].copy()
    print(f"  -> {len(final)} subjects retained")

    # Column order: phenotypics/metadata first, then shape features.
    pheno_cols = [c for c in ph.columns if c != "ID_int"]
    lead = ["SubjectID", "SiteName", "ID_int"] + pheno_cols
    lead = [c for c in lead if c in final.columns]
    final = final[lead + feat_cols]

    final.to_csv(args.output, index=False, float_format="%.6g")
    print(f"\nSaved: {args.output}")
    print(f"  shape: {final.shape[0]} subjects x {final.shape[1]} columns "
          f"({len(feat_cols)} shape features)")
    print("  by site:", final["SiteName"].value_counts().to_dict())
    if "DX" in final.columns:
        print("  by DX:  ", final["DX"].value_counts().to_dict())


if __name__ == "__main__":
    main()
