"""
This file is intended to do the following types of work:
    * download data from APIs
    * screenscrape data from websites
    * reduce the size of large datasets to something more manageable
    * clean data: reduce/rename columns, normalize strings, adjust values
    * generate data through relatively complicated calculations
"""

import pandas


def main():
    # three datasets - genetic, ANOVA, and IC.
    anova = pandas.read_csv("raw_data/PANCANCER_ANOVA_Thu Mar 27 18_30_59 2025.csv")
    genetic = pandas.read_csv(
        "raw_data/PANCANCER_Genetic_features_cna_Thu Mar 27 18_35_49 2025.csv"
    )
    ic = pandas.read_csv("raw_data/PANCANCER_IC_Thu Mar 27 18_33_01 2025.csv")
    # filter datasets to ONLY the Nutlin-3a drug entries as per project focus
    anova = anova[anova["Drug name"] == "Nutlin-3a (-)"]
    anova = anova[anova["Feature Name"].str.contains("cnaPANCAN")]
    ic = ic[ic["Drug Name"] == "Nutlin-3a (-)"]

    # reduce to only columns necessary for visualizations
    anova = anova[
        [
            "Feature Name",
            "feature_pval",
            "ic50_effect_size",
            "feature_delta_mean_ic50",
            "msi_pval",
            "fdr",
            "log_ic50_mean_pos",
        ]
    ]
    genetic = genetic[["Cell Line Name", "Genetic Feature", "IS Mutated", "Recurrent Gain Loss"]]
    ic = ic[["Drug Name", "Cell Line Name", "TCGA Classification", "Tissue", "IC50", "AUC", "Z score"]]

    anova = anova.rename(columns={"Feature Name": "Genetic Feature"})

    # merge columns based on common ones; include only overlaps in both tables
    combined = anova.merge(genetic, on="Genetic Feature", how="inner")
    combined = combined[
        [
            "Cell Line Name",
            "Genetic Feature",
            "feature_pval",
            "ic50_effect_size",
            "feature_delta_mean_ic50",
            "msi_pval",
            "fdr",
            "IS Mutated",
            "Recurrent Gain Loss",
            "log_ic50_mean_pos",
        ]
    ]

    combined = combined.merge(ic, on="Cell Line Name", how="inner")

    # make CSVs
    anova.to_csv("data_organized/anova.csv")
    ic.to_csv("data_organized/drugdata.csv")
    combined.to_csv("data_organized/combined.csv")
    print("done")


if __name__ == "__main__":
    main()
