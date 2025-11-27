# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# %%
# Create folder for images
os.makedirs("figures", exist_ok=True)

data_path = "../data/processed/"
long = pd.read_csv(f"{data_path}long12_clean.csv")

#long = pd.read_csv("data/processed/long12_clean.csv")

# %%
# ============================================
# Drive Dataset Correlation Heatmap
# ============================================
# Check all columns
print(long.columns.tolist())

features = long.columns.tolist()
print("Features for correlation heatmap:", features)

print("\n" + "="*50)
print("CORRELATION WITH FEATURES")
print("="*50)
corr = long[features].corr()
print(corr)

# %%
# Plot heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(long[features].corr(), cmap='YlGnBu')
plt.title("Correlation Heatmap – Long12 Dataset")
plt.savefig("figures/long12_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
print("\nNaN values per column in correlation matrix:")
print(corr[features].isna().sum())

# Identify columns to drop
columns_to_drop_long = [
    'fuel_air_commanded_equiv_ratio', 
    'time_run_with_mil_on',
    'distance_traveled_with_mil_on'
]

# %%
# Remove them from feature list
features_clean_long = [f for f in features if f not in columns_to_drop_long]

# Recompute correlation
corr_long_clean = long[features_clean_long].corr()

print("\nCleaned correlation matrix for long:")
print(corr_long_clean)

# %%
##not necessary the copy but keeping for consistency
corr_matrix_long = corr_long_clean.copy()

# Get upper triangle of the correlation matrix
upper_tri_long = corr_matrix_long.where(
    np.triu(np.ones(corr_matrix_long.shape), k=1).astype(bool)
)

strongest_pos_long = upper_tri_long.stack().sort_values(ascending=False)
print("\n" + "="*50)
print("Strongest positive correlations (LONG):")
print("="*50)
print(strongest_pos_long.head(10))

strongest_neg_long = upper_tri_long.stack().sort_values(ascending=True)
print("\n" + "="*50)
print("Strongest negative correlations (LONG):")
print("="*50)
print(strongest_neg_long.head(10))

# %%
# ===========================================
# Scatter plots for strongest correlations
# ===========================================

os.makedirs("figures/long12_scatter/positive", exist_ok=True)
os.makedirs("figures/long12_scatter/negative", exist_ok=True)

# %%
# Top 5 positive correlations
top5_pos = strongest_pos_long.head(5)
print("\nCreating scatter plots for top 5 positive correlations...")
for i, ((col1, col2), value) in enumerate(top5_pos.items(), 1):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=long, x=col1, y=col2)
    plt.title(f"{i}. {col1} vs {col2} (Corr={value:.3f})")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.tight_layout()
    plt.savefig(f"figures/long12_scatter/positive/scatter_pos{i}_{col1}_{col2}.png", dpi=300)
    plt.close()

# %%
# Top 5 negative correlations
top5_neg = strongest_neg_long.head(5)
print("Creating scatter plots for top 5 negative correlations...")
for i, ((col1, col2), value) in enumerate(top5_neg.items(), 1):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=long, x=col1, y=col2)
    plt.title(f"{i}. {col1} vs {col2} (Corr={value:.3f})")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.tight_layout()
    plt.savefig(f"figures/long12_scatter/negative/scatter_neg{i}_{col1}_{col2}.png", dpi=300)
    plt.close()