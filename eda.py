import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS PROJECT")
print("=" * 50)

# Load Dataset
data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target

print("\nDataset Loaded Successfully!")

# Display First 5 Rows
print("\nFIRST 5 ROWS")
print(df.head())

# Shape
print("\nDATASET SHAPE")
print(df.shape)

# Columns
print("\nCOLUMN NAMES")
print(df.columns)

# Dataset Information
print("\nDATASET INFO")
print(df.info())

# Statistical Summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# Missing Values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Duplicate Rows
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# Unique Values
print("\nUNIQUE VALUES")
print(df.nunique())

# Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="target", data=df)
plt.title("Target Distribution")
plt.savefig("target_distribution.png")
plt.show()

# Histogram
df.hist(figsize=(18, 12))
plt.suptitle("Feature Distributions")
plt.savefig("feature_distribution.png")
plt.show()

# Correlation Matrix
corr_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# Top Correlations
target_corr = corr_matrix["target"].sort_values(ascending=False)

print("\nTOP CORRELATIONS WITH TARGET")
print(target_corr.head(10))

# Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df.iloc[:, :10])
plt.xticks(rotation=90)
plt.title("Boxplot of Features")
plt.tight_layout()
plt.savefig("boxplot.png")
plt.show()

# Pairplot (First 5 Columns Only)
sample_cols = list(df.columns[:5]) + ["target"]

sns.pairplot(
    df[sample_cols],
    hue="target"
)

plt.savefig("pairplot.png")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("\nGenerated Files:")
print("1. target_distribution.png")
print("2. feature_distribution.png")
print("3. correlation_heatmap.png")
print("4. boxplot.png")
print("5. pairplot.png")

