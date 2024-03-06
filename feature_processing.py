import numpy as np
import pandas as pd


def standard_scale(df):
    """
    Normalizes the data in each column of a Pandas dataframe to the mean and scales the standard deviation to 1.

    Args:
    df (DataFrame): Pandas dataframe to standard scale

    Returns:
    df_copy (DataFrame): a Pandas dataframe containing the scaled data

    """
    df_scaled = df.copy()

    for col in df_scaled.columns:
        data_mean = df_scaled[col].mean()
        data_stdev = df_scaled[col].std()

        df_scaled[col] = (df_scaled[col] - data_mean) / data_stdev

    return df_scaled


# Imports the various feature sets in the dataset

dezso_features = pd.read_csv("raw_data/dezso_features.csv")
go_components = pd.read_csv("raw_data/go_components_10-14-22.csv")
go_functions = pd.read_csv("raw_data/go_functions_10-14-22.csv")
go_processes = pd.read_csv("raw_data/go_processes_10-14-22.csv")
gdpc = pd.read_csv("raw_data/gdpc_10-14-22.csv")
paac = pd.read_csv("raw_data/paac_10-14-22.csv")
fpocket = pd.read_csv("raw_data/fpocket_output.csv")


# Imports the list of proteins with their sequences and labels

protein_list = pd.read_csv("raw_data/all_proteins.csv")

dezso_names = dezso_features["Protein"]

features_categorical = dezso_features[
    ["Enzyme Classification", "Localization", "Essentiality"]
]


# Obtains the one-hot encodings for the categorical features and categorical sub-networks

features_categorical = pd.get_dummies(features_categorical.astype(str))

features_numeric = dezso_features.drop(
    ["Protein", "Enzyme Classification", "Localization", "Essentiality"], axis=1
)


# Replaces string labels for two binary variables with 1 and 0

features_numeric = features_numeric.replace({"Signal Peptide": {"Y": 1, "N": 0}})
features_numeric = features_numeric.replace(
    {"PEST region": {"Potential": 1, "Poor": 0}}
)

features_numeric = standard_scale(features_numeric)

dezso_processed = pd.concat(
    [dezso_names, features_categorical, features_numeric], axis=1
)
