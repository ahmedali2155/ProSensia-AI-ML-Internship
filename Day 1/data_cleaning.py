import pandas as pd
import numpy as np

# -----------------------------
# Load Dataset
# -----------------------------
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "datasets" / "titanic.csv"
OUTPUT_FILE = BASE_DIR / "Day 1" / "cleaned_titanic.csv"

df = pd.read_csv(INPUT_FILE)
OUTPUT_FILE = "cleaned_titanic.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 50)
print("Original Dataset")
print("=" * 50)
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# -----------------------------
# Remove Duplicate Rows
# -----------------------------
df = df.drop_duplicates()


# -----------------------------
# Handle Missing Values
# -----------------------------

# Age → Fill with median
if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())

# Fare → Fill with median
if "Fare" in df.columns:
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Embarked → Fill with mode
if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin → Too many missing values
if "Cabin" in df.columns:
    df["Cabin"] = df["Cabin"].fillna("Unknown")


# -----------------------------
# Clean String Columns
# -----------------------------
string_columns = df.select_dtypes(include="object").columns

df[string_columns] = (
    df[string_columns]
    .apply(lambda col: col.str.strip() if col.dtype == "object" else col)
)

# Standardize Gender
if "Sex" in df.columns:
    df["Sex"] = df["Sex"].str.lower().str.capitalize()


# -----------------------------
# Convert Data Types
# -----------------------------
if "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

if "Fare" in df.columns:
    df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")


# -----------------------------
# Handle Outliers (IQR Method)
# -----------------------------
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df[column] = df[column].clip(lower, upper)


# -----------------------------
# Final Dataset Information
# -----------------------------
print("\n")
print("=" * 50)
print("Cleaned Dataset")
print("=" * 50)

print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------
# Export Clean Dataset
# -----------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaned dataset saved as:", OUTPUT_FILE)