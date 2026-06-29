# Subcortical Brain Shape Analysis in ADHD

**UCLA Emerging Scientists Program, Summer 2026**
Research mentored by Dr. Shantanu Joshi, UCLA Department of Neurology

## Research Question
Does the 3D shape of subcortical brain structures distinguish ADHD from typical development, and does it encode continuous symptom severity?

## Key Findings
- Subcortical shape significantly classifies ADHD vs. controls (balanced accuracy=59.6%, p=0.005)
- Shape does not predict continuous symptom severity (ADHD Index R²=-0.046, p=0.299)
- The right caudate is the single most discriminative structure (BA=0.599, AUC=0.644, p=0.005)
- Low-frequency shape modes (global caudate geometry) drive classification signal

## Dataset
ADHD-200 (Peking and NYU sites, N=365 after QC filtering)
Downloaded from NITRC/FCP-INDI. Not included in this repository.

## Pipeline
1. FSL FIRST subcortical segmentation on raw T1 scans
2. Shape coefficient extraction from .bvars files
3. PCA dimensionality reduction (95% variance, per structure)
4. Nested cross-validated Ridge regression (symptom severity)
5. Nested cross-validated LinearSVC classification (ADHD vs. controls)
6. Structure-by-structure analysis across 15 subcortical structures
7. Caudate saliency mapping

## Requirements
- FSL 6.0+
- Python 3.11 (conda env)
- numpy, pandas, scikit-learn, matplotlib, nibabel, nilearn

## Scripts
- `extract_shape_features.py` — extracts shape coefficients from .bvars files
- `run_pca_reduction.py` — PCA dimensionality reduction per structure
- `regression_pipeline.py` — nested CV Ridge regression on symptom scores
- `classification_pipeline.py` — nested CV LinearSVC ADHD vs. control
- `caudate_analysis.py` — bilateral caudate classification
- `structure_by_structure.py` — per-structure classification across all 15 structures
- `saliency_map.py` — right caudate saliency mapping
- `results_visualization.py` — figures

## Acknowledgements
Data provided by the ADHD-200 Consortium and The Neuro Bureau.
