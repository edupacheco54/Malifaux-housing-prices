import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")


def load_data() -> tuple:
    """
    Load training and test datasets.
    """
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    return train_df, test_df


def handle_missing_values(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Fill missing values in categorical and numerical features.
    """
    categorical_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"]
    numerical_cols = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]

    for col in categorical_cols:
        if col in train_df.columns:
            train_df.loc[:, col] = train_df[col].fillna("None")
        if col in test_df.columns:
            test_df.loc[:, col] = test_df[col].fillna("None")

    for col in numerical_cols:
        if col in train_df.columns:
            train_df.loc[:, col] = train_df[col].fillna(train_df[col].median())
        if col in test_df.columns:
            test_df.loc[:, col] = test_df[col].fillna(test_df[col].median())

    return train_df, test_df


def encode_categorical_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Convert ordinal categorical variables to numeric labels.
    """
    ordinal_cols = ["ExternalQual", "BsmtQual", "KitchenQual", "HeatingQC"]
    ordinal_cols = [
        col
        for col in ordinal_cols
        if col in train_df.columns and col in test_df.columns
    ]

    for col in ordinal_cols:
        train_df[col] = train_df[col].fillna("Missing")
        test_df[col] = test_df[col].fillna("Missing")

        combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)

        lbl = LabelEncoder()
        lbl.fit(combined)

        if "Missing" not in lbl.classes_:
            lbl.classes_ = np.append(lbl.classes_, "Missing")

        train_df[col] = lbl.transform(train_df[col].astype(str))
        test_df[col] = lbl.transform(test_df[col].astype(str))

    train_df = pd.get_dummies(train_df, drop_first=True)
    test_df = pd.get_dummies(test_df, drop_first=True)

    test_df = test_df.reindex(columns=train_df.columns.drop("SalePrice"), fill_value=0)

    return train_df, test_df


def scale_numeric_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Scale numerical features.
    """
    scaler = StandardScaler()
    num_features = ["GrLivArea", "TotalBsmtSF", "GarageArea"]
    num_features = [
        col
        for col in num_features
        if col in train_df.columns and col in test_df.columns
    ]

    train_df[num_features] = scaler.fit_transform(train_df[num_features])
    test_df[num_features] = scaler.transform(test_df[num_features])

    return train_df, test_df


def save_clean_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Save cleaned datasets to disk.
    """
    train_clean_path = os.path.join(DATA_DIR, "train_clean.csv")
    test_clean_path = os.path.join(DATA_DIR, "test_clean.csv")

    train_df.to_csv(train_clean_path, index=False)
    test_df.to_csv(test_clean_path, index=False)

    print("Preprocessing complete. Cleaned data saved.")


def main():
    """
    Run the preprocessing pipeline.
    """
    train_df, test_df = load_data()
    train_df, test_df = handle_missing_values(train_df, test_df)
    train_df, test_df = encode_categorical_features(train_df, test_df)
    train_df, test_df = scale_numeric_features(train_df, test_df)
    save_clean_data(train_df, test_df)


if __name__ == "__main__":
    main()
