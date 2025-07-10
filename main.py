"""
Create amazing plots in this file. You will read the data from `data_organized`
(unless your raw data required no reduction, in which case you can read your data from `raw_data`).
You will do plot-related work such as joins, column filtering, pivots,
small calculations and other simple organizational work.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mach_learning import train_rf_model  # Import model function
from mach_learning import setup


def plot1(df, n=25):
    """
    Generates a lollipop plot of IC50 values for N cell lines chosen at equal intervals after sorting by IC50.

    Args:
        df (pd.DataFrame): DataFrame containing 'Cell Line Name' and 'IC50' columns.
        n (int, optional): The number of cell lines to display, chosen at equal intervals. Defaults to 25.
    """
    # get unique cell lines and their IC50 values, sorted by IC50
    # cell lines are the cancer cell
    # ic50 is a measure of effectiveness
    cell_line_ic50 = df[['Cell Line Name', 'IC50']].drop_duplicates().sort_values(by='IC50')

    # determine the number of unique cell lines
    num_cell_lines = len(cell_line_ic50)
    # if 'n' is greater than or equal to the total number of cell lines, use all of them
    if n >= num_cell_lines:
        selected_ic50 = cell_line_ic50
    # otherwise, select 'n' cell lines at equal intervals
    else:
        # generate 'n' equally spaced indices AFTER sorting
        indices = np.linspace(0, num_cell_lines - 1, n, dtype=int)
        # select cell linesbased on these indices
        selected_ic50 = cell_line_ic50.iloc[indices]

    plt.figure(figsize=(12, 8))
    # draw horizontal lines for each cell line
    plt.hlines(y=selected_ic50['Cell Line Name'], xmin=0, xmax=selected_ic50['IC50'], color='darkred')
    # plot circles at the end of the lines to look like the head of a lollipop
    plt.plot(selected_ic50['IC50'], selected_ic50['Cell Line Name'], "o", color='cadetblue')

    # add labels and title
    plt.xlabel('IC50', fontsize=12)
    plt.ylabel('Cell Line Name', fontsize=12)
    plt.title(f'IC50 Values for {len(selected_ic50)} Cell Lines', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/lollipop_ic50_cell_line.png')


def plot2(df):
    # tissue is the type of cancer (breast, lung, etc.)
    df_cleaned = df.dropna(subset=["Tissue"]).copy()

    # set up the figure size (tall to accommodate multiple subplots)
    plt.figure(figsize=(8, 12))
    tissue_types = df_cleaned["Tissue"].unique()
    num_tissues = len(tissue_types)

    # create KDE plot for each tissue type in a vertical layout
    for i, tissue in enumerate(tissue_types):
        # ic50 is a measure of effectiveness
        data_subset = df_cleaned[df_cleaned["Tissue"] == tissue]["IC50"]
        ax = plt.subplot(num_tissues, 1, i + 1)
        # plot filled and outlined curves of KDE plot
        sns.kdeplot(data_subset, fill=True, alpha=0.8, ax=ax)
        sns.kdeplot(data_subset, color="black", linewidth=0.5, ax=ax)
        # remove y ticks and label tissue name
        ax.set_yticks([])
        ax.set_ylabel(tissue, rotation=0, ha="right")
        if i < num_tissues - 1:
            ax.set_xticks([])
            ax.set_xlabel("")
        else:
            ax.set_xlabel("IC50 values")
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # reduce space between subplots to make them appear like a ridge plot
    plt.subplots_adjust(hspace=-0.7)
    # add title
    plt.suptitle("Nutlin-3a Efficacy Across Cancers", y=0.98, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    plt.savefig("plots/efficacy_ridge_plot.png", bbox_inches="tight")
    print("done")


def plot3(df):
    # 1. Find Min and Max
    # msi pval is [i forgor]
    min_msi = df['msi_pval'].min()
    max_msi = df['msi_pval'].max()

    print(f"Minimum msi_pval: {min_msi}")
    print(f"Maximum msi_pval: {max_msi}")

    # 2. Define Bins
    num_bins = 5
    bins = [min_msi + (max_msi - min_msi) * i / num_bins for i in range(num_bins + 1)]
    print(f"Bins: {bins}")

    bin_labels = [f'{bins[i]:.6f} to {bins[i+1]:.6f}' for i in range(num_bins)]

    # 3. Assign Bins to Data (Create a new column)
    df['msi_bin'] = pd.cut(df['msi_pval'], bins=bins, include_lowest=True, labels=bin_labels)

    # Define bins for AUC (you can adjust these based on your data distribution)
    # auc is area under the curve, another measure of effectiveness
    num_auc_bins = 5
    auc_bins = np.linspace(df['AUC'].min(), df['AUC'].max(), num_auc_bins + 1)
    auc_labels = [f'{auc_bins[i]:.2f} to {auc_bins[i+1]:.2f}' for i in range(num_auc_bins)]
    df['auc_bin'] = pd.cut(df['AUC'], bins=auc_bins, include_lowest=True, labels=auc_labels)

    # Count occurrences of each (msi_bin, auc_bin) combination
    heatmap_data = df.groupby(['msi_bin', 'auc_bin']).size().unstack(fill_value=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap='YlGnBu')  # fmt="d" for integers
    plt.xlabel('AUC Bins')
    plt.ylabel('MSI P-value Bins')
    plt.title('Heatmap of Data Point Counts by MSI and AUC Bins')
    plt.tight_layout()
    plt.savefig('plots/msi_auc_heatmap_counts.png')


def plot4(df):
    # msi pval is [i forgor]
    # tissue is the type of cancer (breast, lung, etc.)
    # ic50 effect size is measure of difference in drug response between mutated and non mutated samples
    # genetic feature is the mutation
    # feature pval is [i forgor]
    # ic50 is a measure of effectiveness
    features = ["msi_pval", "Tissue", "ic50_effect_size", "Genetic Feature", "feature_pval"]
    labels = ["IC50"]
    df = df[features + labels].dropna()
    """
    Trains a Random Forest model and generates two plots: Observed vs. Predicted IC50 and Feature Importance.
    ['Tissue', 'Genetic Feature', 'msi_pval', 'Recurrent Gain Loss', 'fdr', 'IC50'] - most effective so far

    Args:
        df (DataFrame): The input DataFrame containing the necessary features and the target variable
        ('IC50').
    """
    # train the model and retrieve predictions and data
    y_test, y_pred, model, X = train_rf_model(df, features)

    # Plot 1: Expected vs. Observed IC50
    plt.figure(figsize=(10, 7))
    plt.scatter(y_test, y_pred, alpha=0.7, color='skyblue', edgecolors='w', s=80)
    # Diagonal line to indicate perfect prediction
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) 
    # plot aesthetics
    plt.xlabel('Observed IC50', fontsize=12)
    plt.ylabel('Predicted IC50', fontsize=12)
    plt.title('Observed vs. Predicted IC50 Values', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plots/observed_vs_predicted.png')

    # Plot 2: Feature Importance Chart
    feature_importances = model.feature_importances_  # extract feature importances from trained model
    features = X.columns

    # create DataFrame for easier sorting and plotting
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

    # aggregate importance for 'Tissue'
    tissue_importance = importance_df[importance_df['Feature'].str.startswith('Tissue_')]['Importance'].sum()
    # remove individual tissue columns from the dataframe
    importance_df = importance_df[~importance_df['Feature'].str.startswith('Tissue_')]
    # add the aggregated tissue importance back
    importance_df = pd.concat([importance_df,
                               pd.DataFrame([{'Feature': 'Tissue', 'Importance': tissue_importance}])])

    # aggregate importance for 'Genetic Feature' (similarly)
    genetic_feature_importance = importance_df[importance_df['Feature'].str.startswith('Genetic Feature_')][
        'Importance'].sum()
    importance_df = importance_df[~importance_df['Feature'].str.startswith('Genetic Feature_')]
    importance_df = pd.concat([importance_df, pd.DataFrame([{
        'Feature': 'Genetic Feature', 'Importance': genetic_feature_importance}])])

    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # select the top N important features to plot
    top_n = 20
    # ensure we don't try to take the head of an empty dataframe
    if not importance_df.empty:
        top_importance_df = importance_df.head(top_n)
    else:
        top_importance_df = pd.DataFrame(columns=['Feature', 'Importance'])

    # plotting the top N feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_importance_df, palette='viridis')
    # plot aesthetics
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance for IC50 Prediction (without AUC)', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/feature_importance_no_auc.png')


def plot5(df):
    # tissue is the type of cancer (breast, lung, etc.)
    # ic50 effect size is measure of difference in drug response between mutated and non mutated samples
    # genetic feature is the mutation
    # feature pval is [i forgor]
    # ic50 is a measure of effectiveness
    features = ["Tissue", "ic50_effect_size", "Genetic Feature", "feature_pval"]
    labels = ["IC50"]
    df = df[features + labels].dropna()
    # PyTorch model section
    y, y_hat, test_rmse = setup(df, features, labels)

    plt.figure(figsize=(10, 7))
    plt.scatter(y_hat, y, alpha=0.7, color="skyblue", edgecolors="w", s=80)

    plt.plot([y_hat.min(), y_hat.max()], [y_hat.min(), y_hat.max()], "r--", lw=2)

    plt.xlabel("Observed IC50", fontsize=12)
    plt.ylabel("Predicted IC50", fontsize=12)
    plt.title("Observed vs. Predicted IC50 Values", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/observed_vs_predicted_torch.png")


def main():
    df = pd.read_csv("data_organized/combined.csv")
    # plot1(df)
    # plot2(df)
    # plot3(df)
    plot4(df)
    # plot5(df) 


if __name__ == "__main__":
    main()
