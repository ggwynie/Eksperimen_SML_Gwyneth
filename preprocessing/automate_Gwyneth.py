import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

COLUMNS_TO_DROP = ["alive", "embark_town", "class", "who", "adult_male", "deck"]
NUMERIC_COLS_TO_SCALE = ["age", "fare", "family_size"]


def load_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in COLUMNS_TO_DROP if c in df.columns]
    return df.drop(columns=existing)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "embarked" in df.columns:
        df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    if "age" in df.columns:
        df["age"] = df.groupby(["pclass", "sex"])["age"].transform(
            lambda x: x.fillna(x.median())
        )
    return df


def handle_outliers(df: pd.DataFrame, fare_quantile: float = 0.99) -> pd.DataFrame:
    df = df.copy()
    fare_cap = df["fare"].quantile(fare_quantile)
    df["fare"] = np.where(df["fare"] > fare_cap, fare_cap, df["fare"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["child", "teen", "young_adult", "adult", "senior"],
    )
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["embarked", "age_group"], drop_first=True)
    return df


def scale_numeric(df: pd.DataFrame, scaler: StandardScaler = None):
    df = df.copy()
    cols = [c for c in NUMERIC_COLS_TO_SCALE if c in df.columns]
    if scaler is None:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])
    return df, scaler


def preprocess_data(input_path: str, output_path: str = None):
    df = load_data(input_path)
    df = clean_columns(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = engineer_features(df)
    df = encode_categorical(df)
    df, _ = scale_numeric(df)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[automate_Gwyneth] Dataset siap-latih disimpan ke: {output_path}")

    print(f"[automate_Gwyneth] Shape akhir: {df.shape}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../namadataset_raw/titanic_raw.csv")
    parser.add_argument("--output", type=str, default="namadataset_preprocessing/titanic_preprocessed.csv")
    args = parser.parse_args()
    preprocess_data(args.input, args.output)


if __name__ == "__main__":
    main()
