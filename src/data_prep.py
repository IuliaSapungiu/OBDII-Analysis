import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Read original raw CSV
# drive_df = pd.read_csv("data/raw/drive11.csv", index_col=False)
# print(drive_df.head())

# Determine the repo root relative to current file/notebook
try:
    # If running a Python script (__file__ exists)
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    # If running in Jupyter/Colab (__file__ does not exist)
    REPO_ROOT = os.path.abspath("..")  # assumes notebook is inside notebooks/

# Set paths relative to repo root
RAW_PATH = os.path.join(REPO_ROOT, "data", "raw")
PROCESSED_PATH = os.path.join(REPO_ROOT, "data", "processed")
os.makedirs(PROCESSED_PATH, exist_ok=True)


# Read only drive11 dataset
drive11_df = pd.read_csv(os.path.join(RAW_PATH, "drive11.csv"), index_col=False)
print("Drive11 head:")
print(drive11_df.head())


# All datasets stored in a simple dictionary
datasets = {
    "drive11": "drive11.csv",
    "idle30": "idle30.csv",
    "live20": "live20.csv",
    "long12": "long12.csv",
    "ufpe1": "ufpe1.csv"
}

# Ensure output folder exists
os.makedirs(PROCESSED_PATH, exist_ok=True)

def clean_column(col):
    """
    Clean column names:
    - remove ' ()'
    - remove any '(' or ')'
    - replace spaces with '_'
    - lowercase everything
    """
    col = col.strip()
    col = col.replace(" ()", "")
    col = col.replace("(", "").replace(")", "")
    col = col.replace(" ", "_")
    return col.lower()

print("Starting dataset preparation...\n")


# Process each dataset in a simple loop
for name, file in datasets.items():
    print(f"\n------------------------------------------")
    print(f"Processing: {name} ({file})")
    print("------------------------------------------")

    # 1. Load dataset (IMPORTANT: index_col=False)
    df = pd.read_csv(f"{RAW_PATH}/{file}", index_col=False)

    # 2. Show original columns
    print("\nOriginal columns:")
    print(df.columns.tolist())

    # 3. Clean column names
    df.columns = [clean_column(c) for c in df.columns]

    # 4. Print cleaned columns
    print("\nCleaned columns:")
    print(df.columns.tolist())

    # 5. Show shape and dtypes
    print("\nShape:", df.shape)
    print("\nData types:")
    print(df.dtypes)

    # 6. Check for missing values
    print("\nMissing values per column:")
    print(df.isna().sum())

    # 7. Remove fully empty rows
    empty_rows = df.isna().all(axis=1).sum()
    if empty_rows > 0:
        print(f"\nRemoving {empty_rows} fully-empty rows...")
        df = df.dropna(how="all")

    # 8. Show sample rows
    print("\nFirst 5 rows:")
    print(df.head())

    # 9. Save cleaned dataset
    output_path = f"{PROCESSED_PATH}/{name}_clean.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved cleaned dataset → {output_path}")

print("\nAll datasets processed successfully! 🚀")
