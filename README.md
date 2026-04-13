<div align="center">

# ⚗️ Oxidation State Assignment in MOFs via Soft-Voting Ensemble

**Reproducing and extending Jablonka et al. — from CSD-trained MOFs to a custom OQMD/ICSD dataset**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-100%25-3776AB?logo=python&logoColor=white)](.)
[![Proxy R²](https://img.shields.io/badge/Proxy_R²-≈_0.91-brightgreen)](.)
[![Dataset](https://img.shields.io/badge/Samples-~7k_OQMD-blue)](.)

</div>

---

## Motivation

Oxidation state assignment is a deceptively hard problem in materials informatics. For simple ionic solids, electronegativity rules suffice — but in metal-organic frameworks (MOFs) with mixed-valence metals, ambiguous coordination, and complex ligand fields, manual assignment becomes unreliable and heuristics break down. With ongoing efforts to innovate high-entropy alooys, a pipeline to assign oxidation state on MOF is of great value.

Jablonka et al. ([Nature Chemistry, 2021](https://doi.org/10.1038/s41557-021-00717-y)) demonstrated that a soft-voting ensemble trained on CSD-derived MOF data with "collective knowledge" labels can assign oxidation states with high accuracy. Crucially, the paper claimed the model transfers to other material types — binary ionic solids, simple metal complexes — beyond its training domain.

This project puts that transferability claim to the test. Rather than reusing the paper's CSD dataset, it builds an **entirely new dataset** from ~7,000 ICSD-tagged entries scraped from OQMD, with **manually assigned oxidation state labels**, a **replicated featurization pipeline**, and an **extended ensemble** that outperforms the paper's configuration by ~6%.

## What Makes This Non-Trivial

This is not a "download pretrained model and run inference" reproduction. The project required:

1. **Dataset construction from scratch** — Scraping ~7k ICSD-tagged entries from the OQMD API, filtering for multi-element compounds, identifying metals via Pymatgen, and **manually assigning oxidation state labels** using charge-balance constraints and common oxidation state tables
2. **Feature engineering replication** — Rebuilding the paper's featurization pipeline (metal center features, CrystalNN fingerprints, Gaussian symmetry functions, local property statistics) from their source code, then verifying feature matrix alignment column-by-column against the paper's published features
3. **Systematic model comparison** — First evaluating the paper's pretrained model on the new dataset, then the paper's hyperparameters on the new dataset, then fully re-tuned models, then an extended ensemble with additional model families

Each step uncovered decisions that the paper left implicit.

## Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                     DATA CURATION                         │
│  OQMD API → ~7k ICSD-tagged entries                      │
│  → filter multi-element compounds                        │
│  → identify metals (Pymatgen)                            │
│  → manual oxidation state labeling (charge balance)      │
│  → exclude clashing atoms / NaN labels                   │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                    FEATURIZATION                          │
│  Metal center: column, row, valence electrons,           │
│    diff-to-18, s/p/d unfilled                            │
│  Geometry: CrystalNN fingerprint, Gaussian symmetry      │
│  Chemistry: local property stats (Magpie)                │
│  → 116+ features per metal environment                   │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│              SHAP-GUIDED FEATURE SELECTION                │
│  SHAP on ExtraTrees → rank 116 features                  │
│  Sweep top-N (30→116) for both ET alone and ensemble     │
│  Optimum: 47 features (ensemble) / 44 features (ET)      │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│           INDIVIDUAL MODEL TUNING (Hyperopt)             │
│                                                          │
│  GradientBoosting ─── TPE, 400+ iterations               │
│  ExtraTrees ───────── TPE, 400 iterations                │
│  KNN ──────────────── TPE                                │
│  SGD (linear) ─────── TPE                                │
│  LightGBM ─────────── TPE (extension beyond paper)       │
│  RandomForest ─────── TPE (extension beyond paper)       │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│           WEIGHTED SOFT-VOTING ENSEMBLE                   │
│                                                          │
│  Hyperopt over voting weights (1:10 scale)               │
│  + voting type (soft vs hard)                            │
│  → optimized per-model contribution                      │
│  → ~6% over uniform voting + paper's ensemble            │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
                    Proxy R² ≈ 0.91
```

## Progression of Results

The notebook walks through a deliberate sequence of experiments, each building on the last:

| Step | Configuration | Purpose |
|---|---|---|
| **Phase 1** | Paper's pretrained model on new data | Does the published model transfer? |
| **Phase 2a** | Paper's model types + paper's hyperparameters | Do the hyperparameters transfer? |
| **Phase 2b** | Paper's model types + re-tuned hyperparameters | How much does tuning help on this dataset? |
| **Phase 2c** | SHAP feature selection (47 of 116) | Does pruning noise features improve accuracy? |
| **Phase 2d** | Optimized voting weights (1:10 scale) | Does asymmetric weighting beat uniform? |
| **Phase 2e** | Extended ensemble (+ LGBM, RF) | Do additional model families help? |
| **Final** | All extensions combined | **Proxy R² ≈ 0.91, ~6% over baseline** |

This progression is the scientific contribution — each row isolates one variable, making it clear *where* the improvement comes from.

## Hyperparameter Optimization Strategy

Each base learner is tuned independently using **Hyperopt** with **TPE (Tree-structured Parzen Estimator)** as the primary sampler. The notebook also experiments with random search and simulated annealing to compare convergence behavior. Learning curves (cumulative best accuracy vs. iteration) are plotted for each model to determine when additional iterations stop yielding improvement.

The ensemble-level weights are themselves optimized via Hyperopt — searching over a continuous weight space (scale 1:10) for each base model, jointly with voting type (soft vs. hard). The weight-vs-performance landscape is visualized to show that GB and ET dominate the optimal weighting while KNN and SGD receive near-zero weight.

## Feature Selection

SHAP values (TreeExplainer) are computed on the ExtraTrees base model and used to rank all 116 features by mean absolute importance. A sweep from 30 to 116 features is run for both the standalone ExtraTrees classifier and the full ensemble:

- **ExtraTrees optimum:** 44 features
- **Ensemble optimum:** 47 features

The 3-feature gap between the two optima shows that the ensemble benefits from slightly more features than any individual model — additional weak signals that are noise for one model become useful when combined across diverse learners.

## Exploratory Analysis

The notebook includes a **PCA visualization of Cu-containing compounds**, projecting the 47 selected features into 2D and coloring by oxidation state. This reveals whether the feature space naturally separates oxidation states for a specific metal family — a useful sanity check that the learned features encode chemically meaningful information.

## Repository Structure

```
Oxidation_State_MOF/
├── src/                  # Core pipeline implementation
├── featurize/            # Feature engineering (CrystalNN, Gaussian, Magpie, metal center)
├── optimization/         # Hyperopt tuning scripts for all base models + ensemble weights
├── data/                 # Processed features, labels, intermediate artifacts
├── notebooks/            # Full experiment notebook with all phases
├── utils/                # Shared utilities
├── env/                  # Environment configuration
├── learn_mof_ox_state    # Reference: paper's original featurization code
├── hyperopt-sklearn      # Reference: hyperopt-sklearn integration
└── README.md
```

## Tech Stack

| Component | Technology |
|---|---|
| **Base Learners** | ExtraTrees, GradientBoosting, KNN, SGD, LightGBM, RandomForest |
| **Ensemble** | scikit-learn VotingClassifier (soft voting, Hyperopt-optimized weights) |
| **Feature Engineering** | Pymatgen (structure parsing, metal identification), Matminer (CrystalNN, GaussianSymmFunc, MagpieData) |
| **Feature Selection** | SHAP (TreeExplainer) |
| **Hyperparameter Optimization** | Hyperopt (TPE, random search, simulated annealing) |
| **Data Source** | OQMD API (~7k ICSD-tagged entries) |
| **Scaling** | RobustScaler (fitted on training set only) |

## Reference

This project reproduces and extends:

> Jablonka et al., "Using collective knowledge to assign oxidation states of metal cations in metal–organic frameworks," *Nature Chemistry*, 2021.
> [DOI: 10.1038/s41557-021-00717-y](https://doi.org/10.1038/s41557-021-00717-y)

The extension validates the paper's transferability claim on a completely independent dataset (OQMD/ICSD vs. CSD), with manual label curation, replicated featurization, and an expanded ensemble that improves over the published configuration by ~6%.

---

<div align="center">

📬 sayeed.shahriar@gmail.com · [Portfolio](https://submerged-in-matrix.github.io/projects/oxidation-states/) · [GitHub](https://github.com/submerged-in-matrix)

</div>
