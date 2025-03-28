# Install necessary packages in Colab
!pip install tensorflow pandas numpy openpyxl scikit-learn seaborn matplotlib networkx

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set visualization style for publication-quality graphics
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1)

# -------------------------------
# Custom Group Lasso Regularizer for LSTM
# -------------------------------
class GroupLassoRegularizer(Regularizer):
    def __init__(self, lmbda):
        self.lmbda = lmbda
    def __call__(self, weight_matrix):
        group_norms = tf.norm(weight_matrix, axis=1)
        return self.lmbda * tf.reduce_sum(group_norms)
    def get_config(self):
        return {'lmbda': self.lmbda}

# -------------------------------
# Custom Group Lasso Regularizer for MLP
# -------------------------------
class GroupLassoMLPRegularizer(Regularizer):
    def __init__(self, lmbda, n_features, timesteps):
        self.lmbda = lmbda
        self.n_features = n_features
        self.timesteps = timesteps
    def __call__(self, weight_matrix):
        # Reshape weight matrix to (n_features, timesteps, units)
        new_shape = (self.n_features, self.timesteps, -1)
        weight_matrix_reshaped = tf.reshape(weight_matrix, new_shape)
        group_norms = tf.norm(weight_matrix_reshaped, ord='euclidean', axis=[1, 2])
        return self.lmbda * tf.reduce_sum(group_norms)
    def get_config(self):
        return {'lmbda': self.lmbda, 'n_features': self.n_features, 'timesteps': self.timesteps}

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
# Load data from the Excel file (update path if necessary)
df_crypto = pd.read_excel('/content/combined_data.xlsx', sheet_name='AllData')

# Select and rename columns for BTC, ETH, ADA, XRP, BCH, XAU, and CrudeOil.
# (Note: 'USDX_Close' is removed as it is not present in the updated data.)
df_closing = df_crypto[['Date', 'BTC_Close', 'ETH_Close', 'ADA_Close',
                         'XRP_Close', 'BCH_Close', 'XAU_Close', 'CrudeOil_Close']]
df_closing.columns = ['Date', 'BTC_Close', 'ETH_Close', 'ADA_Close',
                        'XRP_Close', 'BCH_Close', 'XAU_Close', 'CrudeOil_Close']
df_closing['Date'] = pd.to_datetime(df_closing['Date'])
df_closing.dropna(inplace=True)
df_closing.sort_values('Date', inplace=True)

# Define asset columns
crypto_columns = ['BTC_Close', 'ETH_Close', 'ADA_Close', 'XRP_Close', 'BCH_Close', 'XAU_Close', 'CrudeOil_Close']

# Ensure stationarity: take the first difference
df_stationary = df_closing.copy()
for col in crypto_columns:
    df_stationary[col] = df_stationary[col].diff()
df_stationary.dropna(inplace=True)

# Normalize features for improved NN convergence
scaler = StandardScaler()
df_stationary[crypto_columns] = scaler.fit_transform(df_stationary[crypto_columns])

# -------------------------------
# Neural Network Granger Causality Analysis Functions
# -------------------------------

# LSTM-based analysis
def analyze_granger_lstm(df, target, features, timesteps=10, units=20, lmbda=0.01, epochs=50, batch_size=16):
    X = df[features].values
    y = df[target].values
    X_seq, y_seq = [], []
    for i in range(timesteps, len(X)):
        X_seq.append(X[i-timesteps:i, :])
        y_seq.append(y[i])
    X_seq = np.array(X_seq, dtype='float32')
    y_seq = np.array(y_seq, dtype='float32')
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
    input_seq = Input(shape=(timesteps, len(features)))
    lstm_out = LSTM(units, kernel_regularizer=GroupLassoRegularizer(lmbda), return_sequences=False)(input_seq)
    dropout_out = Dropout(0.2)(lstm_out)
    output = Dense(1)(dropout_out)
    model = Model(inputs=input_seq, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
    kernel_weights = model.layers[1].cell.kernel.numpy()
    group_norms = np.linalg.norm(kernel_weights, axis=1)
    return dict(zip(features, group_norms))

# MLP-based analysis
def analyze_granger_mlp(df, target, features, timesteps, units, lmbda, epochs=30, batch_size=16):
    X = df[features].values
    y = df[target].values
    X_seq, y_seq = [], []
    for i in range(timesteps, len(X)):
        X_seq.append(X[i-timesteps:i].flatten())
        y_seq.append(y[i])
    X_seq = np.array(X_seq, dtype='float32')
    y_seq = np.array(y_seq, dtype='float32')
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
    input_seq = Input(shape=(timesteps * len(features),))
    dense_out = Dense(units, activation='relu', kernel_initializer='glorot_uniform',
                      kernel_regularizer=GroupLassoMLPRegularizer(lmbda, n_features=len(features), timesteps=timesteps))(input_seq)
    output = Dense(1)(dense_out)
    model = Model(inputs=input_seq, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
    kernel_weights = model.layers[1].kernel.numpy().reshape(len(features), timesteps, units)
    group_norms = np.linalg.norm(kernel_weights, axis=(1, 2))
    return dict(zip(features, group_norms))

# -------------------------------
# Run Iterative Granger Causality Analysis for Both Models
# -------------------------------
results_lstm = {}
results_mlp = {}
for target in crypto_columns:
    features = [col for col in crypto_columns if col != target]
    print(f"\nAnalyzing target: {target} using LSTM...")
    norms_lstm = analyze_granger_lstm(df_stationary, target, features, timesteps=10, units=20, lmbda=0.01, epochs=50, batch_size=16)
    results_lstm[target] = norms_lstm
    print(f"Analyzing target: {target} using MLP...")
    norms_mlp = analyze_granger_mlp(df_stationary, target, features, timesteps=10, units=20, lmbda=0.01, epochs=30, batch_size=16)
    results_mlp[target] = norms_mlp

# Function to build square DataFrame from results dictionary
def build_group_norm_matrix(results, order):
    mat = pd.DataFrame(index=order, columns=order)
    for target in order:
        for feature in order:
            if target == feature:
                mat.loc[target, feature] = np.nan
            else:
                mat.loc[target, feature] = results[target].get(feature, np.nan)
    return mat

desired_order = ['BTC_Close', 'ETH_Close', 'ADA_Close', 'XRP_Close', 'BCH_Close', 'XAU_Close', 'CrudeOil_Close']
group_norm_matrix_lstm = build_group_norm_matrix(results_lstm, desired_order)
group_norm_matrix_mlp = build_group_norm_matrix(results_mlp, desired_order)

print("LSTM-based Granger causality (group norm) analysis results (rearranged):")
print(group_norm_matrix_lstm)
print("MLP-based Granger causality (group norm) analysis results (rearranged):")
print(group_norm_matrix_mlp)

# -------------------------------
# Compute Pearson Correlation Matrix (for reference)
# -------------------------------
corr_matrix = df_stationary[crypto_columns].corr()
corr_matrix = corr_matrix.reindex(index=desired_order, columns=desired_order)
print("Reordered Pearson correlation matrix:")
print(corr_matrix)

# -------------------------------
# Network Diagram Construction Function
# -------------------------------
def build_network_diagram(group_norm_matrix, threshold, title):
    G = nx.DiGraph()
    # Add nodes with cleaned labels
    for node in group_norm_matrix.index:
        G.add_node(node.replace('_Close',''))
    # Add edges if the value is above the threshold (edge from predictor to target)
    for target in group_norm_matrix.index:
        for predictor in group_norm_matrix.columns:
            if target != predictor:
                val = group_norm_matrix.loc[target, predictor]
                if pd.notna(val) and val >= threshold:
                    G.add_edge(predictor.replace('_Close',''), target.replace('_Close',''), weight=val)
    pos = nx.circular_layout(G)
    plt.figure(figsize=(10,8))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_family='Times New Roman', font_size=12)
    edges = G.edges(data=True)
    edge_widths = [data['weight']*2 for u, v, data in edges]  # scale for visualization
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, width=edge_widths, edge_color='gray')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.show()

# Function to compute threshold based on the 60th percentile of non-NaN values in the matrix
def compute_threshold(mat):
    vals = mat.values.flatten()
    vals = vals[~pd.isna(vals)]
    return np.percentile(vals, 60)

threshold_lstm = compute_threshold(group_norm_matrix_lstm)
threshold_mlp = compute_threshold(group_norm_matrix_mlp)
print("LSTM threshold:", threshold_lstm)
print("MLP threshold:", threshold_mlp)

# Build and display network diagrams for LSTM and MLP-based results
build_network_diagram(group_norm_matrix_lstm, threshold_lstm, "LSTM-based Neural Granger Causality Network Diagram")
build_network_diagram(group_norm_matrix_mlp, threshold_mlp, "MLP-based Neural Granger Causality Network Diagram")
