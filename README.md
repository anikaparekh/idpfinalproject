[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/AKP3XMJQ)

# IDP Final Project - Nutlin-3a Cancer Response
### Authors: Indu Lingala, Anika Parekh

**This project investigates the effectiveness of Nutlin-3a across various cancer types and gene mutations. Our goal is to identify patterns and predictive markers of drug sensitivity using data visualization and machine learning.**

Our preprocessing code involves:
- Filtering for Nutlin-3a only - Extracted all relevant information about the drug Nutlin-3a from large pan-cancer datasets to focus our analysis on a single experiment. Our data was gathered from the website [cancerrxgene.org](cancerrxgene.org) (specifically, these links: [ANOVA dataset](cancerrxgene.org/downloads/anova), [Drug Data](cancerrxgene.org/downloads/drug_data), and [Genetic Features Data](cancerrxgene.org/downloads/genetic_features)). 
- Cleaning and simplifying raw files - Selected only the important columns (e.g., IC50, AUC, feature p-values), making the data easier to work with.
- Merged three datasets (ANOVA, Genetic features, IC) into a new one based on common columns (such as 'Cell Line Name' and 'Genetic Feature'), allowing us to explore how different factors such as mutations and cell lines impact drug effectiveness. 
- Exported organized files - Created three cleaned .csv files (the final merged dataset or combined.csv, the ANOVA dataset with only necessary features or anova.csv, and the drug data dataset with necessary features only, or drugdata.csv). 

## Steps to Run the Code
All required data files are **present in the repository**. No additional setup or external API calls are needed; **simply run the script locally** to preprocess and organize the data.

Ensure your dataset file **combined.csv** is inside the **data_organized** folder.

Generate each plot using the methods in the **main.py** file:
- plot1 generates the IC50 vs Cell Line Lollipop plot
- plot2 generates the Effectiveness by Cancer Type Ridge plot
- plot3 generates the MSI Status plot
- plot4 generates the basic Machine Learning Observed vs Predicted IC50 and Feature Importance graphs
- plot5 generates the PyTorch plot and Model Comparison plot

Using the machine learning model(s), **IC50** can be predicted. The models are trained and fit in the **mach_learning.py** file, specifically in the **train_rf_model** and the ___ methods. Then they are called in the **main.py** file to generate plots. 

View results: all plots and model outputs will be saved in the **plots** folder, named after their respective plot contents. 

## Data Flow Explanation
We begin by loading, combining, and preprocessing the dataset, filtering relevant columns and handling missing values.

We then generate exploratory plots to uncover patterns between mutations, tissues, other determinants such as MSI, and IC50 values.

Finally, we train a regression model (RandomForestRegressor or PyTorch MLP) to predict IC50 using features like tissue type and MSI. 

Model results and performance metrics are saved and visualized in the plots folder (specifically __ and __ plots)

## Key Columns Used
- IC50: The drug concentration that stops 50% of cell activity (like growth). Low IC50: Very effective drug (needs little to work). High IC50: Less effective drug (needs a lot to work).
- Cell Line Name: Unique identifier for the tested cell line
- Genetic Feature: Name of the mutation or genetic alteration 
- ic50_effect_size: Effect size measuring the influence of the genetic feature on IC50
- msi_pval: P-value indicating association between MSI status and IC50
- fdr: False discovery rate correction for multiple testing
- log_ic50_mean_pos: Log-transformed mean IC50 for positive (mutated) samples
- Tissue: Tissue of origin for the cell line
- AUC: Area under the drug response curve for the cell line
- Z score: Shows how sensitive a cell line is compared to others. A positive z-score means the drug response is above average. A negative z-score means the drug response is below average. 

## Libraries Used
- **pandas** – Data manipulation
- **matplotlib**, seaborn – Plotting
- **scikit-learn** – Machine learning (Random Forest, train-test splitting)
- **PyTorch** – Neural network model (optional/advanced)
- **numpy** – Numerical operations
