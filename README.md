# Soft-Voting Ensemble for Oxidation State Assignment in MOFs

## Overview

This project extends and reproduce a published (DOI: https://www.nature.com/articles/s41557-021-00717-y) soft-voting ensemble to assign oxidation states in MOF compounds.

It evaluates whether supervised ML can reproduce and improve upon literature heuristics.

---

## Dataset

- ~7,000 ICSD-tagged MOF entries  
- Derived from OQMD subset  
- Targets manually curated  

---

## Modeling Strategy

Four independently tuned base learners using:
- Random search  
- Simulated annealing  
- TPE optimization  

Weighted soft voting used for final inference.

Custom weighting scale: 1:10.

---

## Results

- Proxy R² ≈ **0.91**  
- ~6% improvement over uniform voting baseline  
- Demonstrated benefit of asymmetric ensemble weighting
