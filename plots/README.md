# Nutlin-3a Cancer Response Project - Plot Information

This folder contains all data visualizations created for our project focused on analyzing the efficacy of Nutlin-3a across cancer cell lines and tissue types and modeling response prediction using machine learning approaches. These plots are central to our analysis and model evaluation.

##  Files & Descriptions

### 1. IC50 vs Cell Line – Lollipop Plot

Filename: lollipop_ic50_cell_line.png.

Description: Displays IC50 values for Nutlin-3a across various cancer cell lines, sorted by response magnitude. Lollipop format emphasizes differences in sensitivity and identifies resistant lines.

### 2. Efficacy by Cancer Type – Ridge Plot

Filename: efficacy_ridge_plot.png

Description: Ridge plot showing the distribution of Nutlin-3a efficacy across different cancer types. Useful for comparing sensitivity trends between tumor categories.

### 3. MSI Status – Yes/No Percentage Plot

Filename: msi_status_percentage_plot.png

Description: Visualizes the percentage of cell lines between certain categories of levels of Microsatellite Instability (MSI) and their response classification to Nutlin-3a. Highlights  relationships between genomic instability and drug effectiveness.

### 4. Random Forest Regressor – Predictions vs Actual

Filename: observed_vs_predicted.png

Description: Scatter plot comparing predicted vs actual Nutlin-3a response values using the Random Forest Regressor. Includes performance metrics (like R squared and MAE) and a diagonal reference line.

### 5. Random Forest Feature Importance

Filename: feature_importance_no_auc.png

Description: Bar plot ranking input features by their relative importance in the Random Forest model. Highlights predictive biomarkers or genomic features that play a role in drug effectiveness.

### 6. PyTorch Model Predictions

Filename: pytorch_predictions_plot.png

Description: Visualization of predicted vs actual values using the PyTorch neural network model. Designed to assess deep learning performance compared to classical ML models.

### 7. Model Comparison Plot (?)

Filename: model_comparison_plot.png

Description: An overlay comparison of PyTorch and Random Forest performance metrics or predictions. Demonstrates strengths and limitations of each approach.



