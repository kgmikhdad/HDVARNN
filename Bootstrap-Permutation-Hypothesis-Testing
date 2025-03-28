# -------------------------------
# Import necessary libraries
# -------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import percentileofscore

# -------------------------------
# USER-DEFINED PARAMETERS
# -------------------------------
# High computation mode adjusts hyperparameters to more demanding settings.
HIGH_COMPUTATION_MODE = False
num_permutations = 30 if not HIGH_COMPUTATION_MODE else 100  # Number of permutation tests
num_jobs = 4 if not HIGH_COMPUTATION_MODE else -1              # Parallel jobs (-1 uses all processors)
units = 10 if not HIGH_COMPUTATION_MODE else 20                # Number of LSTM units
epochs = 20 if not HIGH_COMPUTATION_MODE else 50               # Number of training epochs
batch_size = 32 if not HIGH_COMPUTATION_MODE else 16           # Training batch size

timesteps = 10  # Use 10 timesteps for LSTM input sequences (explicit time lag)
lmbda = 0.01    # Regularization strength for Group Lasso

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
# Load the dataset from Excel (update file path and sheet name if needed)
df_crypto = pd.read_excel('/content/combined_data.xlsx', sheet_name='AllData')

# Define the required columns that must exist in the data file.
required_cols = ['Date', 'BTC_Close', 'ETH_Close', 'ADA_Close',
                 'XRP_Close', 'BCH_Close', 'XAU_Close', 'CrudeOil_Close']

# Verify that all required columns exist; if not, raise an error.
missing_cols = [col for col in required_cols if col not in df_crypto.columns]
if missing_cols:
    raise KeyError(f"Missing columns in input file: {missing_cols}. Please ensure the file has the correct column names.")

# Select only the required columns and create a copy.
df_closing = df_crypto[required_cols].copy()

# Convert 'Date' column to datetime, drop missing values, and sort by date.
df_closing.loc[:, 'Date'] = pd.to_datetime(df_closing['Date'])
df_closing.dropna(inplace=True)
df_closing.sort_values('Date', inplace=True)

# Define the list of asset closing price columns (excluding 'Date').
crypto_columns = ['BTC_Close', 'ETH_Close', 'ADA_Close', 'XRP_Close',
                  'BCH_Close', 'XAU_Close', 'CrudeOil_Close']

# Ensure stationarity by taking the first difference of each series.
df_stationary = df_closing.copy()
for col in crypto_columns:
    df_stationary.loc[:, col] = df_stationary[col].diff()
df_stationary.dropna(inplace=True)

# Normalize the features using StandardScaler.
scaler = StandardScaler()
df_stationary[crypto_columns] = scaler.fit_transform(df_stationary[crypto_columns])

# -------------------------------
# Custom Group Lasso Regularizer
# -------------------------------
class GroupLassoRegularizer(Regularizer):
    """
    Custom regularizer that applies group Lasso on the LSTM kernel weights.
    It reshapes the kernel so that weights corresponding to each input feature (across all LSTM gates)
    are grouped together, and then computes their L2 norm.
    """
    def __init__(self, lmbda, n_features):
        self.lmbda = lmbda            # Regularization strength
        self.n_features = n_features  # Number of input features

    def __call__(self, weight_matrix):
        # LSTM kernel shape is (input_dim, 4 * units). Reshape to (n_features, -1)
        weight_2d = tf.reshape(weight_matrix, (self.n_features, -1))
        # Compute the L2 norm for each feature group.
        group_norms = tf.norm(weight_2d, ord=2, axis=1)
        return self.lmbda * tf.reduce_sum(group_norms)

    def get_config(self):
        return {'lmbda': self.lmbda, 'n_features': self.n_features}

# -------------------------------
# LSTM-Based Granger Causality Analysis Function
# -------------------------------
def analyze_granger(df, target, features, timesteps, lmbda, units, epochs, batch_size):
    """
    Build and train an LSTM model with group Lasso regularization to determine feature importance
    for predicting the target variable. The function creates time-series sequences using the specified
    timesteps, trains the model, and computes the group norm (importance) for each feature.
    """
    # Extract predictor (X) and target (y) values from the DataFrame.
    X = df[features].values
    y = df[target].values

    # Create sequential data using a sliding window of length 'timesteps'.
    X_seq, y_seq = [], []
    for i in range(timesteps, len(X)):
        X_seq.append(X[i-timesteps:i, :])
        y_seq.append(y[i])
    X_seq = np.array(X_seq, dtype='float32')
    y_seq = np.array(y_seq, dtype='float32')

    # Split the data into training and validation sets while preserving time order.
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    # Build the LSTM model.
    input_layer = Input(shape=(timesteps, len(features)))
    # Add an LSTM layer with the custom Group Lasso regularizer applied to its kernel weights.
    lstm_layer = LSTM(units, kernel_regularizer=GroupLassoRegularizer(lmbda, len(features)))(input_layer)
    # Add a Dense output layer for regression (predicting a continuous target value).
    output = Dense(1)(lstm_layer)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mse')

    # Set up callbacks for early stopping and learning rate reduction.
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]

    # Train the model (verbose=0 suppresses output).
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)

    # Extract the LSTM kernel weights using get_weights().
    # For LSTM, the first element of get_weights() is the kernel with shape (input_dim, 4*units).
    lstm_weights = model.layers[1].get_weights()[0]
    # Reshape weights so that rows correspond to each feature (n_features, 4*units).
    feature_importance = np.linalg.norm(lstm_weights.reshape(len(features), -1), axis=1)

    # Return a dictionary mapping each feature to its computed importance (group norm).
    return dict(zip(features, feature_importance))

# -------------------------------
# Permutation Testing (Parallelized)
# -------------------------------
def parallel_permutation_test(df, target, features, timesteps, lmbda, units, epochs, batch_size, num_permutations, num_jobs):
    """
    Generate a null distribution for feature importance by shuffling the target variable and
    re-running the analysis multiple times in parallel.
    """
    def single_permutation(_):
        # Copy the DataFrame to shuffle the target without affecting the original.
        df_shuffled = df.copy()
        # Shuffle the target column while preserving the index (to maintain time alignment).
        df_shuffled[target] = df_shuffled[target].sample(frac=1).values
        # Compute feature importance for the shuffled data.
        return analyze_granger(df_shuffled, target, features, timesteps, lmbda, units, epochs, batch_size)

    # Run the permutation tests in parallel using joblib's Parallel.
    null_dists = Parallel(n_jobs=num_jobs)(delayed(single_permutation)(i) for i in range(num_permutations))
    # Organize the null distribution into a dictionary with one list per feature.
    return {f: [res[f] for res in null_dists] for f in features}

# -------------------------------
# Main Analysis Workflow
# -------------------------------
results = {}       # To store actual feature importance scores for each target
p_values_all = {}  # To store p-values for each predictor for each target

# Iterate over each asset (target) and treat the others as predictors.
for target in crypto_columns:
    print(f"Analyzing target: {target}")
    # Define predictors as all columns except the current target.
    features = [col for col in crypto_columns if col != target]

    # Compute true feature importance using the actual (non-shuffled) data.
    true_importance = analyze_granger(df_stationary, target, features, timesteps, lmbda, units, epochs, batch_size)

    # Generate a null distribution via permutation testing.
    null_distribution = parallel_permutation_test(df_stationary, target, features, timesteps, lmbda, units, epochs, batch_size, num_permutations, num_jobs)

    # Compute p-values using the standard procedure:
    # p-value = ((100 - percentile) / 100) * (number of features), capped at 1.0.
    pvals = {}
    for f in features:
        obs = true_importance[f]
        null = null_distribution[f]
        perc = percentileofscore(null, obs)  # This returns a value between 0 and 100.
        p_value = ((100 - perc) / 100) * len(features)
        p_value = min(p_value, 1.0)  # Ensure p-value does not exceed 1.0.
        pvals[f] = p_value

    results[target] = true_importance
    p_values_all[target] = pvals

# -------------------------------
# Convert p-values into a DataFrame for visualization
# -------------------------------
# Create a matrix with targets as rows and predictors as columns.
p_value_df = pd.DataFrame(index=crypto_columns, columns=crypto_columns)
for target in crypto_columns:
    for feature in crypto_columns:
        if target == feature:
            p_value_df.loc[target, feature] = np.nan  # Self-causality is not tested.
        else:
            p_value_df.loc[target, feature] = p_values_all[target].get(feature, np.nan)

# Print the resulting p-value matrix.
print("\nP-values for Granger Causality (Permutation Testing):")
print(p_value_df)

# -------------------------------
# Visualization of Results
# -------------------------------
# Plot a heatmap of the full p-value matrix.
plt.figure(figsize=(12, 10))
sns.heatmap(p_value_df.astype(float), annot=True, fmt=".3f", cmap='rocket_r', linewidths=0.5, square=True,
            cbar_kws={'label': 'Adjusted p-value'})
plt.title(f'LSTM-Based Granger Causality (Timesteps = {timesteps})', fontsize=16, fontweight='bold')
plt.xlabel('Predictor Variables', fontsize=14)
plt.ylabel('Target Variables', fontsize=14)
plt.tight_layout()
plt.show()


# -------------------------------
# Run the main analysis loop and store null distributions for each target
# -------------------------------
results = {}                # Stores true feature importance for each target.
p_values_all = {}           # Stores p-values for each predictor for each target.
null_distributions_all = {} # New: To store null distributions per target.

for target in crypto_columns:
    print(f"Analyzing target: {target}")
    # Define predictors as all columns except the current target.
    features = [col for col in crypto_columns if col != target]

    # Compute true feature importance using the actual (non-shuffled) data.
    true_importance = analyze_granger(df_stationary, target, features, timesteps, lmbda, units, epochs, batch_size)

    # Generate a null distribution via permutation testing.
    null_distribution = parallel_permutation_test(df_stationary, target, features, timesteps, lmbda, units, epochs, batch_size, num_permutations, num_jobs)

    # Store the null distribution for this target.
    null_distributions_all[target] = null_distribution

    # Compute p-values using the standard procedure.
    pvals = {}
    for f in features:
        obs = true_importance[f]
        null = null_distribution[f]
        perc = percentileofscore(null, obs)  # Value between 0 and 100.
        p_value = ((100 - perc) / 100) * len(features)
        pvals[f] = min(p_value, 1.0)

    results[target] = true_importance
    p_values_all[target] = pvals

# -------------------------------
# Visualization: Plot the null distribution for each predictor for every target
# -------------------------------
for target in crypto_columns:
    # Define the predictors (explanatory variables) for the current target.
    features = [col for col in crypto_columns if col != target]

    for feature in features:
        null_vals = null_distributions_all[target][feature]  # Norm values from each permutation trial.
        true_val = results[target][feature]  # True importance from the actual data.

        plt.figure(figsize=(8, 6))
        plt.hist(null_vals, bins=10, alpha=0.7, label='Null Distribution')
        plt.axvline(x=true_val, color='red', linestyle='--', label='True Importance')
        plt.title(f'Null Distribution for {feature}\n(Target: {target})')
        plt.xlabel('Norm Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()

