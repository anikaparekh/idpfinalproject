# Organized Data – Nutlin-3a Cancer Response Project

This folder contains cleaned and processed datasets for the Nutlin-3a Cancer Response project. The data here has either been derived from larger raw datasets or included directly if no preprocessing was required.

##  Included Datasets

### 1. anova.csv

Description: Results from ANOVA analysis evaluating the association between genetic features and drug sensitivity.

Key Columns:
- Genetic Feature
- feature_pval – p-value for the feature association
- ic50_effect_size – strength of feature effect on IC50
- feature_delta_mean_ic50 – mean IC50 difference between feature-positive and -negative groups
- msi_pval, fdr, log_ic50_mean_pos

### 2. drugdata.csv
Description: Core drug response dataset linking Nutlin-3a sensitivity metrics to cell lines and tissue types.

Key Columns:
- Drug Name, Cell Line Name, TCGA Classification, Tissue
- IC50, AUC, Z score - drug effectiveness measures

### 3. combined.csv
Description: Merged dataset combining drugdata.csv, anova.csv, and additional genetic features to enable integrated modeling and analysis.

Usage: Used as the master dataset for training machine learning models and generating final visualizations.
