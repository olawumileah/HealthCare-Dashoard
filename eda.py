# ======================================
# Hospital Readmission Data Preparation
# ======================================

# Import libraries
import pandas as pd
import numpy as np

# --------------------------------------
# STEP 1: LOAD DATA
# --------------------------------------
data_path = r"C:\Users\DELL\Documents\Health Care Dashboarb\Dataset\Hospital Readmission.csv"
data = pd.read_csv(data_path)

# Preview dataset
print("\n First five rows of the dataset:")
print(data.head())

# Basic info
print("\n Dataset Info:")
print(data.info())

# Dataset shape
print(f"\n Dataset shape: {data.shape}")

# Check for missing data
print("\n Missing values per column:")
print(data.isnull().sum())

# Check data types
print("\n Data types:")
print(data.dtypes)


# --------------------------------------
# STEP 2: FEATURE ENGINEERING
# --------------------------------------

# Convert age ranges to numeric midpoints
def age_to_mid(age_range):
    """Convert an age range like '[70-80)' to a numeric midpoint (e.g., 75)."""
    low, high = age_range.strip('[]()').split('-')
    return int(round((int(low) + int(high)) / 2))

# Apply transformation
data['age_mid'] = data['age'].apply(age_to_mid)


# Handle target variable ('readmitted')
data['readmitted'] = data['readmitted'].map({'no': 0, 'yes': 1})

# Create new engineered features
data['total_visits'] = data['n_outpatient'] + data['n_inpatient'] + data['n_emergency']
data['procedure_intensity'] = data['n_lab_procedures'] / (data['time_in_hospital'] + 1)
data['medication_per_day'] = data['n_medications'] / (data['time_in_hospital'] + 1)


# --------------------------------------
# STEP 3: CLEAN BINARY YES/NO COLUMNS
# --------------------------------------

binary_cols = ['glucose_test', 'A1Ctest', 'change', 'diabetes_med']

for col in binary_cols:
    data[col] = (
        data[col]
        .astype(str)
        .str.lower()
        .replace({'missing': np.nan, 'unknown': np.nan, '?': np.nan, 'nan': np.nan})
        .map({'yes': 1, 'no': 0})
        .fillna(0)
        .astype(int)
    )


# --------------------------------------
# STEP 4: CHECK CLEAN DATA
# --------------------------------------
print("\n🧹 Missing values per column (after cleaning):")
print(data.isnull().sum())

print("\n Sample after feature engineering:")
print(data.head())

# --------------------------------------
# STEP 5: BASIC DESCRIPTIVE ANALYSIS
# --------------------------------------

print("\n Dataset Information:")
print(data.info())

print("\n Descriptive Statistics:")
print(data.describe().T)

categorical_cols = ['medical_specialty', 'diag_1', 'diag_2', 'diag_3']
for col in categorical_cols:
    print(f"\nUnique values in {col}: {data[col].nunique()}")

# --------------------------------------
# STEP 6: SAVE CLEANED DATA
# --------------------------------------
output_path = r"C:\Users\DELL\Documents\Health Care Dashboarb\Dataset\cleaned_hospital_data.csv"
data.to_csv(output_path, index=False)
print(f"\n Cleaned dataset saved to: {output_path}")
