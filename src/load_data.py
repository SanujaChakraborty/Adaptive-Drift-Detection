# src/load_data.py
import pandas as pd

def load_nsl_kdd(txt_path, nrows=None):
    """
    Load NSL-KDD from .txt file into DataFrame.
    Assumes last column = label.
    """
    df = pd.read_csv(txt_path, sep=",", header=None, nrows=nrows)
    df.columns = [f"f{i}" for i in range(df.shape[1]-1)] + ["label"]
    return df

def load_cicids(csv_path, nrows=None):
    """
    Load CICIDS dataset (already in CSV).
    """
    df = pd.read_csv(csv_path, nrows=nrows)
    # unify label column name
    if "Label" in df.columns:
        df.rename(columns={"Label": "label"}, inplace=True)
    return df

def load_and_combine_cicids(csv_paths, nrows_per_file=2000):
    """
    Loads, combines, and shuffles multiple CICIDS CSV files for robust analysis.
    Each file is sampled to nrows_per_file, columns normalized, and 'Label' unified to 'label'.
    """
    import pandas as pd
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path, nrows=nrows_per_file)
        if "Label" in df.columns:
            df.rename(columns={"Label": "label"}, inplace=True)
        dfs.append(df)
    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_combined
