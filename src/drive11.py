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
drive = pd.read_csv(f"{data_path}drive11_clean.csv")
#drive = pd.read_csv("data/processed/drive11_clean.csv")

# %%
# ============================================
# Drive Dataset Correlation Heatmap
# ============================================
print(drive.columns.tolist())

features = [
    'engine_run_tine', 'engine_rpm', 'vehicle_speed', 'throttle', 'engine_load',
    'coolant_temperature', 'long_term_fuel_trim_bank_1', 'short_term_fuel_trim_bank_1',
    'intake_manifold_pressure', 'fuel_tank', 'absolute_throttle_b', 'pedal_d', 'pedal_e',
    'commanded_throttle_actuator', 'fuel_air_commanded_equiv_ratio',
    'absolute_barometric_pressure', 'relative_throttle_position', 'intake_air_temp',
    'timing_advance', 'catalyst_temperature_bank1_sensor1',
    'catalyst_temperature_bank1_sensor2', 'control_module_voltage',
    'commanded_evaporative_purge', 'time_run_with_mil_on',
    'time_since_trouble_codes_cleared', 'distance_traveled_with_mil_on',
    'warm_ups_since_codes_cleared'
]

print("\n" + "="*50)
print("CORRELATION WITH FEATURES")
print("="*50)
corr = drive[features].corr()
print(corr)

# %%
# Plot heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(drive[features].corr(), cmap='YlGnBu')
plt.title("Correlation Heatmap – Drive11 Dataset")
plt.savefig("figures/drive11_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
print("\nNaN values per column in correlation matrix:")
print(corr[features].isna().sum())
columns_to_drop = ['fuel_air_commanded_equiv_ratio', 'time_run_with_mil_on', 
                   'distance_traveled_with_mil_on', 'warm_ups_since_codes_cleared']

# %%
# Remove them from feature list
features_clean = [f for f in features if f not in columns_to_drop]
# Recompute correlation without the problematic columns
corr_clean = drive[features_clean].corr()
print("\nCleaned correlation matrix:")
print(corr_clean)

# %%
# Copy for consistency
corr_matrix = corr_clean.copy()
# Flatten the matrix and keep only upper triangle
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

strongest_pos = upper_tri.stack().sort_values(ascending=False)
print("\n" + "="*50)
print("Strongest positive correlations:")
print("="*50)
print(strongest_pos.head(10))  

print("\n" + "="*50)
print("Strongest positive correlations:")
print("="*50)
strongest_neg = upper_tri.stack().sort_values(ascending=True)
print("Strongest negative correlations:")
print(strongest_neg.head(10))  

# %%
# ===========================================
# Scatter plots for strongest correlations
# ===========================================

os.makedirs("figures/drive11_scatter/positive", exist_ok=True)
os.makedirs("figures/drive11_scatter/negative", exist_ok=True)

# %%
# Top 5 positive correlations
top5_pos = strongest_pos.head(5)
print("\nCreating scatter plots for top 5 positive correlations...")
for i, ((col1, col2), value) in enumerate(top5_pos.items(), 1):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=drive, x=col1, y=col2)
    plt.title(f"{i}. {col1} vs {col2} (Corr={value:.3f})")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.tight_layout()
    plt.savefig(f"figures/drive11_scatter/positive/scatter_pos{i}_{col1}_{col2}.png", dpi=300)
    plt.close()

# %%
# Top 5 negative correlations
top5_neg = strongest_neg.head(5)
print("Creating scatter plots for top 5 negative correlations...")
for i, ((col1, col2), value) in enumerate(top5_neg.items(), 1):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=drive, x=col1, y=col2)
    plt.title(f"{i}. {col1} vs {col2} (Corr={value:.3f})")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.tight_layout()
    plt.savefig(f"figures/drive11_scatter/negative/scatter_neg{i}_{col1}_{col2}.png", dpi=300)
    plt.close()