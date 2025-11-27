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
ufpe = pd.read_csv(f"{data_path}ufpe1_clean.csv")

#ufpe = pd.read_csv("data/processed/ufpe1_clean.csv")

# %%
# ============================================
# Drive Dataset Correlation Heatmap
# ============================================
# Check all columns
print(ufpe.columns.tolist())

features = ufpe.columns.tolist()
print("Features for correlation heatmap:", features)

print("\n" + "="*50)
print("CORRELATION WITH FEATURES")
print("="*50)
corr = ufpe[features].corr()
print(corr)

# %%
# Plot heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(ufpe[features].corr(), cmap='YlGnBu')
plt.title("Correlation Heatmap – Ufpe1 Dataset")
plt.savefig("figures/ufpe1_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
print("\nNaN values per column in correlation matrix:")
print(corr[features].isna().sum())

# Identify columns to drop
columns_to_drop_ufpe = [
    'fuel_air_commanded_equiv_ratio', 'time_run_with_mil_on', 
    'distance_traveled_with_mil_on', 'warm_ups_since_codes_cleared'
]

# %%
# Remove them from feature list
features_clean_ufpe = [f for f in features if f not in columns_to_drop_ufpe]

# Recompute correlation
corr_ufpe_clean = ufpe[features_clean_ufpe].corr()

print("\nCleaned correlation matrix for ufpe:")
print(corr_ufpe_clean)

# %%
##not necessary the copy but keeping for consistency
corr_matrix_ufpe = corr_ufpe_clean.copy()

# Get upper triangle of the correlation matrix
upper_tri_ufpe = corr_matrix_ufpe.where(
    np.triu(np.ones(corr_matrix_ufpe.shape), k=1).astype(bool)
)

strongest_pos_ufpe = upper_tri_ufpe.stack().sort_values(ascending=False)
print("\n" + "="*50)
print("Strongest positive correlations (UFPE):")
print("="*50)
print(strongest_pos_ufpe.head(10))

strongest_neg_ufpe = upper_tri_ufpe.stack().sort_values(ascending=True)
print("\n" + "="*50)
print("Strongest negative correlations (UFPE):")
print("="*50)
print(strongest_neg_ufpe.head(10))

# %%
# ===========================================
# Scatter plots for strongest correlations
# ===========================================

os.makedirs("figures/ufpe1_scatter/positive", exist_ok=True)
os.makedirs("figures/ufpe1_scatter/negative", exist_ok=True)

# %%
# Top 5 positive correlations
top5_pos = strongest_pos_ufpe.head(5)
print("\nCreating scatter plots for top 5 positive correlations...")
for i, ((col1, col2), value) in enumerate(top5_pos.items(), 1):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=ufpe, x=col1, y=col2)
    plt.title(f"{i}. {col1} vs {col2} (Corr={value:.3f})")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.tight_layout()
    plt.savefig(f"figures/ufpe1_scatter/positive/scatter_pos{i}_{col1}_{col2}.png", dpi=300)
    plt.close()

# %%
# Top 5 negative correlations
top5_neg = strongest_neg_ufpe.head(5)
print("Creating scatter plots for top 5 negative correlations...")
for i, ((col1, col2), value) in enumerate(top5_neg.items(), 1):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=ufpe, x=col1, y=col2)
    plt.title(f"{i}. {col1} vs {col2} (Corr={value:.3f})")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.tight_layout()
    plt.savefig(f"figures/ufpe1_scatter/negative/scatter_neg{i}_{col1}_{col2}.png", dpi=300)
    plt.close()