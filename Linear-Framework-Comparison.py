import seaborn as sns
import matplotlib.pyplot as plt

# Set Times New Roman font and other style parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1)

# -------------------------------
# Prepare display DataFrames without "_Close"
# (Assuming group_norm_matrix and corr_matrix are defined above)
# -------------------------------
df_results_display = group_norm_matrix.copy()
df_results_display.index = df_results_display.index.str.replace('_Close', '')
df_results_display.columns = df_results_display.columns.str.replace('_Close', '')

# Force numeric (coercing any problematic cells to NaN)
df_results_display = df_results_display.apply(pd.to_numeric, errors='coerce')

corr_matrix_display = corr_matrix.copy()
corr_matrix_display.index = corr_matrix_display.index.str.replace('_Close', '')
corr_matrix_display.columns = corr_matrix_display.columns.str.replace('_Close', '')

# Also ensure correlation matrix is numeric if needed
corr_matrix_display = corr_matrix_display.apply(pd.to_numeric, errors='coerce')

# -------------------------------
# Visualize Granger Causality (Group Norms) Heatmap
# -------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(
    df_results_display,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    linewidths=0.5,
    square=True,
    cbar_kws={'shrink': 0.8}
)
plt.title('Granger Causality (Group Norms) Heatmap', fontsize=16, fontweight='bold')
plt.xlabel('Features (Predictors)', fontsize=14)
plt.ylabel('Target Variables', fontsize=14)
plt.tight_layout()
plt.show()

# -------------------------------
# Visualize Pearson Correlation Matrix Heatmap
# -------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix_display,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    linewidths=0.5,
    square=True,
    cbar_kws={'shrink': 0.8}
)
plt.title('Pearson Correlation Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Variables', fontsize=14)
plt.ylabel('Variables', fontsize=14)
plt.tight_layout()
plt.show()
