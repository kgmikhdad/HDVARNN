High Dimensional Granger Causality Detection Using LSTM and MLP Neural Networks and High dimesional VAR


### Data Description

The dataset consists of quarterly observations of various economic and financial indices and their corresponding returns, spanning from the first quarter of 1990 to the fourth quarter of 2022. The dataset includes the following columns:

1. **date**: The date of the observation, in the format 'MMM-YY' (e.g., 'Mar-90').
2. **year**: The year of the observation.
3. **quarter**: The quarter of the observation (1, 2, 3, or 4).
4. **PE_index**: Private Equity Index values.
5. **PE_r**: Private Equity returns.
6. **VC_index**: Venture Capital Index values.
7. **VC_r**: Venture Capital returns.
8. **Bond10**: Bond 10-year Index values.
9. **Bond10_r**: Bond 10-year returns.
10. **SP500**: S&P 500 Index values.
11. **SP500_r**: S&P 500 returns.
12. **GSCI**: Goldman Sachs Commodity Index values.
13. **GSCI_r**: Goldman Sachs Commodity returns.
14. **HFRI**: Hedge Fund Research Index values.
15. **HFRI_r**: Hedge Fund Research returns.
16. **NFCI**: National Financial Conditions Index values.
17. **NFCI_r**: National Financial Conditions returns.
18. **PMI**: Purchasing Managers' Index values.
19. **PMI_r**: Purchasing Managers' Index returns.

The data includes a total of 132 quarterly observations for each index and return pair. The index values represent the performance of the respective indices, while the returns represent the percentage change in the index values over the quarter.

### Plot Description

For each pair of index and return columns, we have generated a dual y-axis time series plot using Python and Matplotlib. Each plot includes the following features:

1. **Primary y-axis on the left**: Displays the index values.
2. **Secondary y-axis on the right**: Displays the return values (scaled to percentage).
3. **x-axis**: Represents the date, formatted to show both the year and quarter.
4. **Two lines**:
   - A solid blue line representing the index values.
   - A dashed red line representing the return values, scaled to percentage.

The plots are titled with the format "Time Series Plot of [Index Name] and [Return Name]" and saved with filenames corresponding to the index, e.g., "plot_PE_index.png" for the Private Equity Index.

### Summary of Plots

1. **PE_index and PE_r**:
   - Title: "Time Series Plot of PE Index and PE Return"
   - Filename: "plot_PE_index.png"
![PE_INDEX](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/plot_PE_index.png) 

2. **VC_index and VC_r**:
   - Title: "Time Series Plot of VC Index and VC Return"
   - Filename: "plot_VC_index.png"
![VC_INDEX](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/plot_VC_index.png) 
3. **Bond10 and Bond10_r**:
   - Title: "Time Series Plot of Bond10 Index and Bond10 Return"
   - Filename: "plot_Bond10.png"
![Bond10](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/plot_Bond10.png) 
4. **SP500 and SP500_r**:
   - Title: "Time Series Plot of SP500 Index and SP500 Return"
   - Filename: "plot_SP500.png"
![SP_500](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/plot_SP500.png) 
5. **GSCI and GSCI_r**:
   - Title: "Time Series Plot of GSCI Index and GSCI Return"
   - Filename: "plot_GSCI.png"
![GSCI](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/plot_GSCI.png) 
6. **HFRI and HFRI_r**:
   - Title: "Time Series Plot of HFRI Index and HFRI Return"
   - Filename: "plot_HFRI.png"
![HFRI](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/plot_HFRI.png) 
7. **NFCI and NFCI_r**:
   - Title: "Time Series Plot of NFCI Index and NFCI Return"
   - Filename: "plot_NFCI.png"
![NFCI](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/plot_NFCI.png) 
8. **PMI and PMI_r**:
   - Title: "Time Series Plot of PMI Index and PMI Return"
   - Filename: "plot_PMI.png"
![PMI](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/plot_PMI.png) 
These plots visually represent the trends and relationships between the indices and their returns over time, facilitating analysis of their dynamics and potential causal relationships.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("C:/Users/muham/Desktop/Time series data.csv")

# Convert 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'], format='%b-%y')

# Function to create dual y-axis plot
def create_dual_axis_plot(df, index_col, return_col, title, filename):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_title(title)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Index Values', color='blue')
    ax1.plot(df['date'], df[index_col], color='blue', label='Index')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Return Values (%)', color='red')
    ax2.plot(df['date'], df[return_col] * 100, color='red', linestyle='--', label='Return')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.tight_layout()
    plt.legend(loc='upper left')
    plt.savefig(filename)
    plt.close()

# List of columns to plot
index_return_pairs = [
    ("PE_index", "PE_r"),
    ("VC_index", "VC_r"),
    ("Bond10", "Bond10_r"),
    ("SP500", "SP500_r"),
    ("GSCI", "GSCI_r"),
    ("HFRI", "HFRI_r"),
    ("NFCI", "NFCI_r"),
    ("PMI", "PMI_r")
]

# Generate plots for each pair
for pair in index_return_pairs:
    index_col, return_col = pair
    title = f"Time Series Plot of {index_col.replace('_', ' ')} and {return_col.replace('_', ' ')}"
    filename = f"plot_{index_col}.png"
    
    create_dual_axis_plot(data, index_col, return_col, title, filename)
```





## Normalize and Plot Data

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = '/mnt/data/Time series data.csv'
data = pd.read_csv(file_path)

# Extract the year column for the x-axis
years = data['year']

# Remove the date, year, and quarter columns
data = data.drop(columns=['date', 'year', 'quarter'], errors='ignore')

# Normalize the data to make it more comparable
data_normalized = (data - data.min()) / (data.max() - data.min())

# Convert normalized data to matrix for plotting
data_matrix = data_normalized.to_numpy()

# Plot the normalized time series with the year on the x-axis
plt.figure(figsize=(10, 6))
plt.plot(years, data_matrix)
plt.xlabel('Year')
plt.ylabel('Normalized Values')
plt.title('Normalized Time Series Data')
plt.legend(data.columns, loc='upper right', fontsize='small')
plt.show()
```

### Normalized Data Plot

![Normalized Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/normalised.png)

## Plot Individual Normalized Index Data with Growth Rate Data

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = '/content/Time series data.csv'  # Update this path to your file location
data = pd.read_csv(file_path)

# Extract the year column for the x-axis
years = data['year']

# Remove the date, year, and quarter columns
data = data.drop(columns=['date', 'year', 'quarter'], errors='ignore')

# Normalize the data to make it more comparable
data_normalized = (data - data.min()) / (data.max() - data.min())

# Separate data for plotting
PE_index = data_normalized[['PE_index', 'PE_r']]
VC_index = data_normalized[['VC_index', 'VC_r']]
Bond10 = data_normalized[['Bond10', 'Bond10_r']]
SP500 = data_normalized[['SP500', 'SP500_r']]
GSCI = data_normalized[['GSCI', 'GSCI_r']]
HFRI = data_normalized[['HFRI', 'HFRI_r']]
NFCI = data_normalized[['NFCI', 'NFCI_r']]
PMI = data_normalized[['PMI', 'PMI_r']]

# Function to plot and save each pair of indices and returns in separate figures
def plot_and_save_index_and_return(index, return_col, index_label, return_label, years, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(years, index, label=index_label, color='blue')
    plt.plot(years, return_col, label=return_label, color='orange')
    plt.title(f'{index_label} and {return_label}')
    plt.xlabel('Year')
    plt.ylabel('Normalized Values')
    plt.legend()
    plt.savefig(filename)
    plt.close()

# Plot and save PE_index and PE_r
plot_and_save_index_and_return(PE_index['PE_index'], PE_index['PE_r'], 'PE_index', 'PE_r', years, 'PE_index_PE_r.png')

# Plot and save VC_index and VC_r
plot_and_save_index_and_return(VC_index['VC_index'], VC_index['VC_r'], 'VC_index', 'VC_r', years, 'VC_index_VC_r.png')

# Plot and save Bond10 and Bond10_r
plot_and_save_index_and_return(Bond10['Bond10'], Bond10['Bond10_r'], 'Bond10', 'Bond10_r', years, 'Bond10_Bond10_r.png')

# Plot and save SP500 and SP500_r
plot_and_save_index_and_return(SP500['SP500'], SP500['SP500_r'], 'SP500', 'SP500_r', years, 'SP500_SP500_r.png')

# Plot and save GSCI and GSCI_r
plot_and_save_index_and_return(GSCI['GSCI'], GSCI['GSCI_r'], 'GSCI', 'GSCI_r', years, 'GSCI_GSCI_r.png')

# Plot and save HFRI and HFRI_r
plot_and_save_index_and_return(HFRI['HFRI'], HFRI['HFRI_r'], 'HFRI', 'HFRI_r', years, 'HFRI_HFRI_r.png')

# Plot and save NFCI and NFCI_r
plot_and_save_index_and_return(NFCI['NFCI'], NFCI['NFCI_r'], 'NFCI', 'NFCI_r', years, 'NFCI_NFCI_r.png')

# Plot and save PMI and PMI_r
plot_and_save_index_and_return(PMI['PMI'], PMI['PMI_r'], 'PMI', 'PMI_r', years, 'PMI_PMI_r.png')


```




### Separated Normalized Data Plots

![Individual_Normalized Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Bond10_Bond10_r.png)
![Individual_Normalized Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/GSCI_GSCI_r.png)
![Individual_Normalized Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/HFRI_HFRI_r.png)
![Individual_Normalized Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/NFCI_NFCI_r.png)
![Individual_Normalized Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/PE_index_PE_r.png)
![Individual_Normalized Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/PMI_PMI_r.png)
![Individual_Normalized Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/SP500_SP500_r.png)
![Individual_Normalized Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/VC_index_VC_r.png)




### Granger Causality Testing and Causal Network Visualization



```r
# Load necessary libraries
library(HDGCvar)
library(igraph)

# Load your data (assuming your data is saved as 'time_series_data.csv')
data <- read.csv("C:/Users/muham/Desktop/Time series data.csv")

# Set the dependent variable and the dataset
dependent_variable <- 'PE_index'
independent_variables <- c('VC_index', 'Bond10', 'SP500', 'GSCI', 'HFRI', 'NFCI', 'PMI', 'PE_r', 'VC_r', 'Bond10_r', 'SP500_r', 'GSCI_r', 'HFRI_r', 'NFCI_r', 'PMI_r')

# Select the lag length
selected_lag <- lags_upbound_BIC(data[, c(dependent_variable, independent_variables)], p_max = 10)
print(selected_lag)

# Prepare the list of interest variables
interest_variables <- lapply(independent_variables, function(var) {
  list(GCto = dependent_variable, GCfrom = var)
})

# Test for Granger causality for each variable
results <- lapply(interest_variables, function(pair) {
  HDGC_VAR(GCpair = pair, data = data[, c(dependent_variable, pair$GCfrom)], p = selected_lag, d = 2, bound = 0.5 * nrow(data), parallel = TRUE, n_cores = 3)
})

# Print results
print(results)

# Optional: Estimate the full network of causality and plot the estimated network with multiple testing correction
network <- HDGC_VAR_all(data[, c(dependent_variable, independent_variables)], p = selected_lag, d = 2, bound = 0.5 * nrow(data), parallel = TRUE, n_cores = 3)
Plot_GC_all(network, Stat_type = "FS_cor", alpha = 0.05, multip_corr = list(TRUE, method = "BH"), directed = TRUE, layout = layout.circle, main = "Network", edge.arrow.size = .2, vertex.size = 5, vertex.color = c("lightblue"), vertex.frame.color = "blue", vertex.label.size = 2, vertex.label.color = "black", vertex.label.cex = 0.6, vertex.label.dist = 1, edge.curved = 0, cluster = list(TRUE, 5, "black", 0.8, 1, 0))


```



---

### Test Results

#### Test 1
- **Asymp**: 
  - LM_stat: 1.8140219 
  - p_value: 0.4037292
- **FS_cor**: 
  - LM_stat: 0.8769226 
  - p_value: 0.4186671
- **Selections**: 
  - PE_index l1: TRUE 
  - PE_index l2: TRUE

---

#### Test 2
- **Asymp**: 
  - LM_stat: 4.4698064 
  - p_value: 0.1070025
- **FS_cor**: 
  - LM_stat: 2.2072190 
  - p_value: 0.1143806
- **Selections**: 
  - PE_index l1: TRUE 
  - PE_index l2: TRUE

---

#### Test 3
- **Asymp**: 
  - LM_stat: 1.0942569 
  - p_value: 0.5786089
- **FS_cor**: 
  - LM_stat: 0.5259783 
  - p_value: 0.5923109
- **Selections**: 
  - PE_index l1: TRUE 
  - PE_index l2: TRUE

---

#### Test 4
- **Asymp**: 
  - LM_stat: 4.1442320 
  - p_value: 0.1259191
- **FS_cor**: 
  - LM_stat: 2.0410689 
  - p_value: 0.1343019
- **Selections**: 
  - PE_index l1: TRUE 
  - PE_index l2: TRUE

---

#### Test 5
- **Asymp**: 
  - LM_stat: 0.9944839 
  - p_value: 0.6082058
- **FS_cor**: 
  - LM_stat: 0.4776448 
  - p_value: 0.6213974
- **Selections**: 
  - PE_index l1: TRUE 
  - PE_index l2: TRUE

---

#### Test 6
- **Asymp**: 
  - LM_stat: 0.1920133 
  - p_value: 0.9084580
- **FS_cor**: 
  - LM_stat: 0.09164383 
  - p_value: 0.91249282
- **Selections**: 
  - PE_index l1: TRUE 
  - PE_index l2: TRUE

---

#### Test 7
- **Asymp**: 
  - LM_stat: 0.4494550 
  - p_value: 0.7987338
- **FS_cor**: 
  - LM_stat: 0.2149482 
  - p_value: 0.8068881
- **Selections**: 
  - PE_index l1: TRUE 
  - PE_index l2: TRUE

---

#### Test 8
- **Asymp**: 
  - LM_stat: 0.4082490 
  - p_value: 0.8153609
- **FS_cor**: 
  - LM_stat: 0.1951787 
  - p_value: 0.8229440
- **Selections**: 
  - PE_index l1: TRUE 
  - PE_index l2: TRUE

---

#### Test 9
- **Asymp**: 
  - LM_stat: 0.1961634 
  - p_value: 0.9065748
- **FS_cor**: 
  - LM_stat: 0.0936276 
  - p_value: 0.9106872
- **Selections**: 
  - PE_index l1: TRUE 
  - PE_index l2: TRUE

---

#### Test 10
- **Asymp**: 
  - LM_stat: 2.1276638 
  - p_value: 0.3451308
- **FS_cor**: 
  - LM_stat: 1.0311042 
  - p_value: 0.3596994
- **Selections**: 
  - PE_index l1: TRUE 
  - PE_index l2: TRUE

---

#### Test 11
- **Asymp**: 
  - LM_stat: 0.1007208 
  - p_value: 0.9508866
- **FS_cor**: 
  - LM_stat: 0.04803757 
  - p_value: 0.95311599
- **Selections**: 
  - PE_index l1: TRUE 
  - PE_index l2: TRUE

---

#### Test 12
- **Asymp**: 
  - LM_stat: 6.78940185 
  - p_value: 0.03355059
- **FS_cor**: 
  - LM_stat: 3.41680941 
  - p_value: 0.03598886
- **Selections**: 
  - PE_index l1: TRUE 
  - PE_index l2: TRUE

---

#### Test 13
- **Asymp**: 
  - LM_stat: 1.2621274 
  - p_value: 0.5320256
- **FS_cor**: 
  - LM_stat: 0.6074725 
  - p_value: 0.5463652
- **Selections**: 
  - PE_index l1: TRUE 
  - PE_index l2: TRUE

---

#### Test 14
- **Asymp**: 
  - LM_stat: 0.1094028 
  - p_value: 0.9467678
- **FS_cor**: 
  - LM_stat: 0.05218186 
  - p_value: 0.94917741
- **Selections**: 
  - PE_index l1: TRUE 
  - PE_index l2: TRUE

---

#### Test 15
- **Asymp**: 
  - LM_stat: 0.3831062 
  - p_value: 0.8256758
- **FS_cor**: 
  - LM_stat: 0.1831221 
  - p_value: 0.8328949
- **Selections**: 
  - PE_index l1: TRUE 
  - PE_index l2: TRUE

---
### Interpretation of Granger Causality Test Results

The Granger causality tests conducted on the dataset involving the dependent variable 'PE_index' and various independent variables provide insights into the causal relationships in the data. Each test result comprises two parts: the test statistics (LM_stat) and the corresponding p-values for both Asymptotic and FS_cor versions of the tests. Additionally, the selections indicate which lags (l1, l2) of the variables were included in the final model.

#### Key Points of Interpretation:

1. **Test Statistics and p-values**:
   - The LM_stat values indicate the strength of the evidence for Granger causality from the independent variable to 'PE_index'.
   - The p-values help determine the statistical significance of these relationships. A low p-value (typically < 0.05) suggests that the independent variable Granger causes the dependent variable.

2. **Asymptotic and FS_cor**:
   - Both tests provide similar insights, but the FS_cor version includes a small-sample correction, which can be more reliable in smaller datasets.
   - Consistency between the Asymp and FS_cor results strengthens the validity of the findings.

3. **Selections**:
   - The selections part indicates which lags of the variables (e.g., PE_index l1, PE_index l2) were included in the model.
   - A 'TRUE' value means that the respective lag was included in the model, indicating its importance in explaining the dependent variable's behavior.

#### Detailed Interpretation:

- **Test 1 (VC_index -> PE_index)**:
  - Both the Asymp and FS_cor tests yield high p-values (0.4037 and 0.4187, respectively), indicating no significant Granger causality from VC_index to PE_index.
  - Both lags (l1, l2) of PE_index were included in the model.

- **Test 2 (Bond10 -> PE_index)**:
  - The p-values (0.0075 and 0.0104) are below 0.05, suggesting a significant Granger causality from Bond10 to PE_index.
  - Both lags (l1, l2) of PE_index were included in the model.

- **Test 3 (SP500 -> PE_index)**:
  - The p-values (0.2204 and 0.2439) are higher than 0.05, indicating no significant Granger causality from SP500 to PE_index.
  - Both lags (l1, l2) of PE_index were included in the model.

- **Test 4 (GSCI -> PE_index)**:
  - The p-values (0.1259 and 0.1343) suggest no significant Granger causality from GSCI to PE_index.
  - Both lags (l1, l2) of PE_index were included in the model.

- **Test 5 (HFRI -> PE_index)**:
  - The p-values (0.6082 and 0.6214) indicate no significant Granger causality from HFRI to PE_index.
  - Both lags (l1, l2) of PE_index were included in the model.

- **Test 6 (NFCI -> PE_index)**:
  - The p-values (0.9085 and 0.9125) suggest no significant Granger causality from NFCI to PE_index.
  - Both lags (l1, l2) of PE_index were included in the model.

- **Test 7 (PMI -> PE_index)**:
  - The p-values (0.7987 and 0.8069) indicate no significant Granger causality from PMI to PE_index.
  - Both lags (l1, l2) of PE_index were included in the model.

- **Test 8 (PE_r -> PE_index)**:
  - The p-values (0.8154 and 0.8229) suggest no significant Granger causality from PE_r to PE_index.
  - Both lags (l1, l2) of PE_index were included in the model.

- **Test 9 (VC_r -> PE_index)**:
  - The p-values (0.9066 and 0.9107) indicate no significant Granger causality from VC_r to PE_index.
  - Both lags (l1, l2) of PE_index were included in the model.

- **Test 10 (Bond10_r -> PE_index)**:
  - The p-values (0.3451 and 0.3597) suggest no significant Granger causality from Bond10_r to PE_index.
  - Both lags (l1, l2) of PE_index were included in the model.

- **Test 11 (SP500_r -> PE_index)**:
  - The p-values (0.9509 and 0.9531) indicate no significant Granger causality from SP500_r to PE_index.
  - Both lags (l1, l2) of PE_index were included in the model.

- **Test 12 (GSCI_r -> PE_index)**:
  - The p-values (0.0336 and 0.0360) are below 0.05, suggesting significant Granger causality from GSCI_r to PE_index.
  - Both lags (l1, l2) of PE_index were included in the model.

- **Test 13 (HFRI_r -> PE_index)**:
  - The p-values (0.5320 and 0.5464) suggest no significant Granger causality from HFRI_r to PE_index.
  - Both lags (l1, l2) of PE_index were included in the model.

- **Test 14 (NFCI_r -> PE_index)**:
  - The p-values (0.9468 and 0.9492) indicate no significant Granger causality from NFCI_r to PE_index.
  - Both lags (l1, l2) of PE_index were included in the model.

- **Test 15 (PMI_r -> PE_index)**:
  - The p-values (0.8257 and 0.8329) suggest no significant Granger causality from PMI_r to PE_index.
  - Both lags (l1, l2) of PE_index were included in the model.

### Summary:
- **Significant Granger Causality**:
  - GSCI_r (Test 12) shows significant Granger causality to PE_index.

- **No Significant Granger Causality**:
  - The remaining independent variables do not show significant Granger causality to PE_index based on the p-values at a 5 percentage level.

---
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot20.png)
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot21.png)


#### First Graph: General Network of Granger Causality Relationships

- **Overview**:
  The first graph depicts the network of Granger causality relationships among the variables at a 5% significance level. Each node represents a different variable from the dataset, and directed edges between nodes indicate significant Granger causality relationships.

- **Key Relationships**:
  - **GSCI**: This variable stands out as a central node with multiple connections. It has several incoming and outgoing edges, suggesting that it both influences and is influenced by other variables in the network.
  - **SP500**: The S&P 500 index also shows multiple causal connections, indicating its importance in the financial network.
  - **Bond10**: The 10-year bond yield is another crucial node, influencing and being influenced by several other variables.
  - **PE_index**: The private equity index has notable connections, indicating it is both a predictor and an outcome in the network of variables.

- **Interpretation**:
  The directed edges signify that past values of the source node help predict future values of the target node. For example, an edge from `Bond10` to `PE_index` suggests that historical data on the 10-year bond yield can be used to predict future values of the private equity index. The presence of multiple connections for nodes like `GSCI`, `SP500`, and `Bond10` highlights their significant roles within the financial and economic system.

#### Second Graph: Clustered Network of Granger Causality Relationships

- **Overview**:
  The second graph builds upon the first by incorporating clustering to highlight groups of variables with stronger internal causal relationships. The clusters are color-coded regions, grouping variables that exhibit significant interactions among themselves.

- **Notable Clusters**:
  - **Red Cluster (PE_index, SP500, NFCI_r)**: This cluster indicates a tight interplay of causality among these variables. It suggests that the private equity index, the S&P 500, and the returns of the National Financial Conditions Index are closely related in terms of their predictive relationships.
  - **Green Cluster (PMI, GSCI, Bond10, PMI_r)**: This group suggests a strong internal causal connection among the Purchasing Managers' Index, the Goldman Sachs Commodity Index, the 10-year bond yield, and the returns of the Purchasing Managers' Index.
  - **Other Clusters**: Smaller clusters such as the one containing `VC_r` or `VC_index` show isolated groups where variables have stronger internal interactions but fewer connections to the broader network.

- **Interpretation**:
  The clustering analysis provides deeper insights into the structure of the data. The red cluster, for example, suggests that movements in the S&P 500 and NFCI returns are closely linked to the private equity index, potentially indicating that these variables should be analyzed together for better predictive accuracy. The green cluster highlights the interconnectedness of various economic indicators, suggesting that they move together and influence each other significantly.

### Conclusion:

- **Significant Variables**:
  - **GSCI**: A key variable with multiple causal interactions.
  - **SP500** and **Bond10**: Important indices with several predictive relationships.

- **Clusters**:
  - **PE_index, SP500, NFCI_r**: A tightly knit group indicating strong mutual influences.
  - **PMI, GSCI, Bond10, PMI_r**: Another closely related group, highlighting significant economic interactions.

These network graphs and their interpretations provide valuable insights into the complex causal relationships within the dataset. By identifying key variables and clusters, we can better understand the dynamics of economic and financial indicators, guiding more informed decision-making and predictive modeling efforts.






---
All Bivariate case
---

```r
# Load necessary libraries
library(HDGCvar)
library(igraph)

# Load your data (assuming your data is saved as 'time_series_data.csv')
data <- read.csv("C:/Users/muham/Desktop/Time series data.csv")

# Set the dependent variable and the dataset
dependent_variable <- 'PE_index'
independent_variables <- c('VC_index', 'Bond10', 'SP500', 'GSCI', 'HFRI', 'NFCI', 'PMI', 'PE_r', 'VC_r', 'Bond10_r', 'SP500_r', 'GSCI_r', 'HFRI_r', 'NFCI_r', 'PMI_r')

# Prepare the dataset
all_variables <- c(dependent_variable, independent_variables)
data_subset <- data[, all_variables]

# Select the lag length
selected_lag <- lags_upbound_BIC(data_subset, p_max = 10)
print(selected_lag)

# Test for Granger causality for all bivariate combinations
network_bivariate <- HDGC_VAR_all(data = data_subset, p = selected_lag, d = 2, bound = 0.5 * nrow(data_subset), parallel = TRUE, n_cores = 3)

# Plot the estimated network for all bivariate combinations
Plot_GC_all(network_bivariate, Stat_type = "FS_cor", alpha = 0.05, multip_corr = list(FALSE), directed = TRUE, layout = layout.circle, main = "Bivariate Network", edge.arrow.size = .2, vertex.size = 5, vertex.color = c("lightblue"), vertex.frame.color = "blue", vertex.label.size = 2, vertex.label.color = "black", vertex.label.cex = 0.6, vertex.label.dist = 1, edge.curved = 0, cluster = list(TRUE, 5, "black", 0.8, 1, 0))


```
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot22.png)
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot23.png)

---
Multiple combination case
---

```r
# Load necessary libraries
library(HDGCvar)
library(igraph)

# Load your data (assuming your data is saved as 'time_series_data.csv')
data <- read.csv("C:/Users/muham/Desktop/Time series data.csv")

# Set the dependent variable and the dataset
dependent_variable <- 'PE_index'
independent_variables <- c('VC_index', 'Bond10', 'SP500', 'GSCI', 'HFRI', 'NFCI', 'PMI', 'PE_r', 'VC_r', 'Bond10_r', 'SP500_r', 'GSCI_r', 'HFRI_r', 'NFCI_r', 'PMI_r')

# Prepare the dataset
all_variables <- c(dependent_variable, independent_variables)
data_subset <- data[, all_variables]

# Select the lag length
selected_lag <- lags_upbound_BIC(data_subset, p_max = 10)
print(selected_lag)

# Prepare the list of variable pairs for multiple combinations
variable_pairs <- lapply(independent_variables, function(var) {
  list(GCto = dependent_variable, GCfrom = var)
})

# Test for Granger causality for multiple combinations
results_multiple <- HDGC_VAR_multiple(GCpairs = variable_pairs, data = data_subset, p = selected_lag, d = 2, bound = 0.5 * nrow(data_subset), parallel = TRUE, n_cores = 4)

# Print results
print(results_multiple)

# Optional: Plot the estimated network for multiple combinations
network_multiple <- HDGC_VAR_all(data = data_subset, p = selected_lag, d = 2, bound = 0.5 * nrow(data_subset), parallel = TRUE, n_cores = 4)
Plot_GC_all(network_multiple, Stat_type = "FS_cor", alpha = 0.01, multip_corr = list(FALSE), directed = TRUE, layout = layout.circle, main = "Multiple Combinations Network", edge.arrow.size = .2, vertex.size = 5, vertex.color = c("lightblue"), vertex.frame.color = "blue", vertex.label.size = 2, vertex.label.color = "black", vertex.label.cex = 0.6, vertex.label.dist = 1, edge.curved = 0, cluster = list(TRUE, 5, "black", 0.8, 1, 0))



```

Result


```r
$tests
, , GCtests = VC_index -> PE_index

         type
stat          Asymp    FS_cor
  LM_stat 1.8140219 0.8769226
  p_value 0.4037292 0.4186671

, , GCtests = Bond10 -> PE_index

         type
stat            Asymp     FS_cor
  LM_stat 9.775980324 4.75469258
  p_value 0.007536555 0.01037542

, , GCtests = SP500 -> PE_index

         type
stat          Asymp    FS_cor
  LM_stat 3.0249652 1.4280688
  p_value 0.2203622 0.2438847

, , GCtests = GSCI -> PE_index

         type
stat          Asymp    FS_cor
  LM_stat 0.4192842 0.1889693
  p_value 0.8108744 0.8280684

, , GCtests = HFRI -> PE_index

         type
stat          Asymp    FS_cor
  LM_stat 2.9549333 1.3469640
  p_value 0.2282151 0.2641349

, , GCtests = NFCI -> PE_index

         type
stat         Asymp    FS_cor
  LM_stat 1.915335 0.8354975
  p_value 0.383787 0.4363917

, , GCtests = PMI -> PE_index

         type
stat          Asymp     FS_cor
  LM_stat 0.2275559 0.09973301
  p_value 0.8924561 0.90515932

, , GCtests = PE_r -> PE_index

         type
stat          Asymp    FS_cor
  LM_stat 0.6594896 0.2822525
  p_value 0.7191072 0.7546327

, , GCtests = VC_r -> PE_index

         type
stat          Asymp    FS_cor
  LM_stat 1.0856395 0.4790302
  p_value 0.5811074 0.6206469

, , GCtests = Bond10_r -> PE_index

         type
stat          Asymp    FS_cor
  LM_stat 3.8669217 1.7756310
  p_value 0.1446467 0.1740286

, , GCtests = SP500_r -> PE_index

         type
stat          Asymp    FS_cor
  LM_stat 0.7098242 0.3290091
  p_value 0.7012351 0.7202945

, , GCtests = GSCI_r -> PE_index

         type
stat          Asymp    FS_cor
  LM_stat 1.0918552 0.4774947
  p_value 0.5793042 0.6216038

, , GCtests = HFRI_r -> PE_index

         type
stat          Asymp    FS_cor
  LM_stat 4.1981552 1.9498411
  p_value 0.1225694 0.1469725

, , GCtests = NFCI_r -> PE_index

         type
stat          Asymp    FS_cor
  LM_stat 1.2356440 0.5117473
  p_value 0.5391173 0.6009346

, , GCtests = PMI_r -> PE_index

         type
stat          Asymp    FS_cor
  LM_stat 2.0436456 0.8923766
  p_value 0.3599382 0.4126256


$selections
$selections$`VC_index -> PE_index`
PE_index l1   Bond10 l1    SP500 l1     GSCI l1     HFRI l1     NFCI l1      PMI l1 
       TRUE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
    PE_r l1     VC_r l1 Bond10_r l1  SP500_r l1   GSCI_r l1   HFRI_r l1   NFCI_r l1 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
   PMI_r l1 PE_index l2   Bond10 l2    SP500 l2     GSCI l2     HFRI l2     NFCI l2 
      FALSE        TRUE       FALSE       FALSE       FALSE       FALSE       FALSE 
     PMI l2     PE_r l2     VC_r l2 Bond10_r l2  SP500_r l2   GSCI_r l2   HFRI_r l2 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
  NFCI_r l2    PMI_r l2 
      FALSE       FALSE 

$selections$`Bond10 -> PE_index`
PE_index l1 VC_index l1    SP500 l1     GSCI l1     HFRI l1     NFCI l1      PMI l1 
       TRUE        TRUE        TRUE        TRUE        TRUE       FALSE       FALSE 
    PE_r l1     VC_r l1 Bond10_r l1  SP500_r l1   GSCI_r l1   HFRI_r l1   NFCI_r l1 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
   PMI_r l1 PE_index l2 VC_index l2    SP500 l2     GSCI l2     HFRI l2     NFCI l2 
      FALSE        TRUE        TRUE       FALSE        TRUE        TRUE       FALSE 
     PMI l2     PE_r l2     VC_r l2 Bond10_r l2  SP500_r l2   GSCI_r l2   HFRI_r l2 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
  NFCI_r l2    PMI_r l2 
      FALSE       FALSE 

$selections$`SP500 -> PE_index`
PE_index l1 VC_index l1   Bond10 l1     GSCI l1     HFRI l1     NFCI l1      PMI l1 
       TRUE        TRUE       FALSE       FALSE        TRUE       FALSE       FALSE 
    PE_r l1     VC_r l1 Bond10_r l1  SP500_r l1   GSCI_r l1   HFRI_r l1   NFCI_r l1 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
   PMI_r l1 PE_index l2 VC_index l2   Bond10 l2     GSCI l2     HFRI l2     NFCI l2 
      FALSE        TRUE        TRUE       FALSE       FALSE        TRUE       FALSE 
     PMI l2     PE_r l2     VC_r l2 Bond10_r l2  SP500_r l2   GSCI_r l2   HFRI_r l2 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
  NFCI_r l2    PMI_r l2 
      FALSE       FALSE 

$selections$`GSCI -> PE_index`
PE_index l1 VC_index l1   Bond10 l1    SP500 l1     HFRI l1     NFCI l1      PMI l1 
       TRUE        TRUE        TRUE        TRUE        TRUE       FALSE       FALSE 
    PE_r l1     VC_r l1 Bond10_r l1  SP500_r l1   GSCI_r l1   HFRI_r l1   NFCI_r l1 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
   PMI_r l1 PE_index l2 VC_index l2   Bond10 l2    SP500 l2     HFRI l2     NFCI l2 
      FALSE        TRUE        TRUE        TRUE       FALSE        TRUE       FALSE 
     PMI l2     PE_r l2     VC_r l2 Bond10_r l2  SP500_r l2   GSCI_r l2   HFRI_r l2 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
  NFCI_r l2    PMI_r l2 
      FALSE       FALSE 

$selections$`HFRI -> PE_index`
PE_index l1 VC_index l1   Bond10 l1    SP500 l1     GSCI l1     NFCI l1      PMI l1 
       TRUE        TRUE        TRUE        TRUE        TRUE       FALSE       FALSE 
    PE_r l1     VC_r l1 Bond10_r l1  SP500_r l1   GSCI_r l1   HFRI_r l1   NFCI_r l1 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
   PMI_r l1 PE_index l2 VC_index l2   Bond10 l2    SP500 l2     GSCI l2     NFCI l2 
      FALSE        TRUE        TRUE        TRUE        TRUE        TRUE       FALSE 
     PMI l2     PE_r l2     VC_r l2 Bond10_r l2  SP500_r l2   GSCI_r l2   HFRI_r l2 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
  NFCI_r l2    PMI_r l2 
      FALSE       FALSE

 

$selections$`NFCI -> PE_index`
PE_index l1 VC_index l1   Bond10 l1    SP500 l1     GSCI l1     HFRI l1      PMI l1 
       TRUE        TRUE        TRUE        TRUE        TRUE        TRUE        TRUE 
    PE_r l1     VC_r l1 Bond10_r l1  SP500_r l1   GSCI_r l1   HFRI_r l1   NFCI_r l1 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
   PMI_r l1 PE_index l2 VC_index l2   Bond10 l2    SP500 l2     GSCI l2     HFRI l2 
      FALSE        TRUE        TRUE        TRUE        TRUE        TRUE        TRUE 
     PMI l2     PE_r l2     VC_r l2 Bond10_r l2  SP500_r l2   GSCI_r l2   HFRI_r l2 
       TRUE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
  NFCI_r l2    PMI_r l2 
      FALSE       FALSE 

$selections$`PMI -> PE_index`
PE_index l1 VC_index l1   Bond10 l1    SP500 l1     GSCI l1     HFRI l1     NFCI l1 
       TRUE        TRUE        TRUE        TRUE       FALSE        TRUE       FALSE 
    PE_r l1     VC_r l1 Bond10_r l1  SP500_r l1   GSCI_r l1   HFRI_r l1   NFCI_r l1 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
   PMI_r l1 PE_index l2 VC_index l2   Bond10 l2    SP500 l2     GSCI l2     HFRI l2 
      FALSE        TRUE        TRUE        TRUE        TRUE        TRUE        TRUE 
    NFCI l2     PE_r l2     VC_r l2 Bond10_r l2  SP500_r l2   GSCI_r l2   HFRI_r l2 
       TRUE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
  NFCI_r l2    PMI_r l2 
      FALSE       FALSE 

$selections$`PE_r -> PE_index`
PE_index l1 VC_index l1   Bond10 l1    SP500 l1     GSCI l1     HFRI l1     NFCI l1 
       TRUE        TRUE        TRUE        TRUE        TRUE        TRUE       FALSE 
     PMI l1     VC_r l1 Bond10_r l1  SP500_r l1   GSCI_r l1   HFRI_r l1   NFCI_r l1 
       TRUE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
   PMI_r l1 PE_index l2 VC_index l2   Bond10 l2    SP500 l2     GSCI l2     HFRI l2 
      FALSE        TRUE        TRUE        TRUE        TRUE        TRUE        TRUE 
    NFCI l2      PMI l2     VC_r l2 Bond10_r l2  SP500_r l2   GSCI_r l2   HFRI_r l2 
      FALSE        TRUE        TRUE       FALSE       FALSE       FALSE       FALSE 
  NFCI_r l2    PMI_r l2 
      FALSE       FALSE 

$selections$`VC_r -> PE_index`
PE_index l1 VC_index l1   Bond10 l1    SP500 l1     GSCI l1     HFRI l1     NFCI l1 
       TRUE        TRUE        TRUE        TRUE       FALSE        TRUE       FALSE 
     PMI l1     PE_r l1 Bond10_r l1  SP500_r l1   GSCI_r l1   HFRI_r l1   NFCI_r l1 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
   PMI_r l1 PE_index l2 VC_index l2   Bond10 l2    SP500 l2     GSCI l2     HFRI l2 
      FALSE        TRUE        TRUE        TRUE        TRUE        TRUE        TRUE 
    NFCI l2      PMI l2     PE_r l2 Bond10_r l2  SP500_r l2   GSCI_r l2   HFRI_r l2 
      FALSE        TRUE       FALSE       FALSE       FALSE       FALSE       FALSE 
  NFCI_r l2    PMI_r l2 
      FALSE       FALSE 

$selections$`Bond10_r -> PE_index`
PE_index l1 VC_index l1   Bond10 l1    SP500 l1     GSCI l1     HFRI l1     NFCI l1 
       TRUE        TRUE        TRUE       FALSE       FALSE        TRUE       FALSE 
     PMI l1     PE_r l1     VC_r l1  SP500_r l1   GSCI_r l1   HFRI_r l1   NFCI_r l1 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
   PMI_r l1 PE_index l2 VC_index l2   Bond10 l2    SP500 l2     GSCI l2     HFRI l2 
      FALSE        TRUE        TRUE        TRUE        TRUE        TRUE        TRUE 
    NFCI l2      PMI l2     PE_r l2     VC_r l2  SP500_r l2   GSCI_r l2   HFRI_r l2 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
  NFCI_r l2    PMI_r l2 
      FALSE       FALSE 

$selections$`SP500_r -> PE_index`
PE_index l1 VC_index l1   Bond10 l1    SP500 l1     GSCI l1     HFRI l1     NFCI l1 
       TRUE       FALSE        TRUE        TRUE       FALSE       FALSE       FALSE 
     PMI l1     PE_r l1     VC_r l1 Bond10_r l1   GSCI_r l1   HFRI_r l1   NFCI_r l1 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
   PMI_r l1 PE_index l2 VC_index l2   Bond10 l2    SP500 l2     GSCI l2     HFRI l2 
      FALSE        TRUE        TRUE       FALSE        TRUE       FALSE       FALSE 
    NFCI l2      PMI l2     PE_r l2     VC_r l2 Bond10_r l2   GSCI_r l2   HFRI_r l2 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
  NFCI_r l2    PMI_r l2 
      FALSE       FALSE 

$selections$`GSCI_r -> PE_index`
PE_index l1 VC_index l1   Bond10 l1    SP500 l1     GSCI l1     HFRI l1     NFCI l1 
       TRUE        TRUE        TRUE        TRUE        TRUE        TRUE       FALSE 
     PMI l1     PE_r l1     VC_r l1 Bond10_r l1  SP500_r l1   HFRI_r l1   NFCI_r l1 
       TRUE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
   PMI_r l1 PE_index l2 VC_index l2   Bond10 l2    SP500 l2     GSCI l2     HFRI l2 
      FALSE        TRUE        TRUE        TRUE        TRUE        TRUE        TRUE 
    NFCI l2      PMI l2     PE_r l2     VC_r l2 Bond10_r l2  SP500_r l2   HFRI_r l2 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
  NFCI_r l2    PMI_r l2 
      FALSE       FALSE 

$selections$`HFRI_r -> PE_index`
PE_index l1 VC_index l1   Bond10 l1    SP500 l1     GSCI l1     HFRI l1     NFCI l1 
       TRUE        TRUE        TRUE        TRUE       FALSE        TRUE       FALSE 
     PMI l1     PE_r l1     VC_r l1 Bond10_r l1  SP500_r l1   GSCI_r l1   NFCI_r l1 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
   PMI_r l1 PE_index l2 VC_index l2   Bond10 l2    SP500 l2     GSCI l2     HFRI l2 
      FALSE        TRUE        TRUE       FALSE       FALSE        TRUE        TRUE 
    NFCI l2      PMI l2     PE_r l2     VC_r l2 Bond10_r l2  SP500_r l2   GSCI_r l2 
      FALSE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
  NFCI_r l2    PMI_r l2 
      FALSE       FALSE 

$selections$`NFCI_r -> PE_index`
PE_index l1 VC_index l1   Bond10 l1    SP500 l1     GSCI l1     HFRI l1     NFCI l1 
       TRUE        TRUE        TRUE        TRUE        TRUE        TRUE        TRUE 
     PMI l1     PE_r l1     VC_r l1 Bond10_r l1  SP500_r l1   GSCI_r l1   HFRI_r l1 
       TRUE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
   PMI_r l1 PE_index l2 VC_index l2   Bond10 l2    SP500 l2     GSCI l2     HFRI l2

 
      FALSE        TRUE        TRUE        TRUE        TRUE        TRUE        TRUE 
    NFCI l2      PMI l2     PE_r l2     VC_r l2 Bond10_r l2  SP500_r l2   GSCI_r l2 
       TRUE        TRUE       FALSE       FALSE       FALSE        TRUE        TRUE 
  HFRI_r l2    PMI_r l2 
      FALSE        TRUE 

$selections$`PMI_r -> PE_index`
PE_index l1 VC_index l1   Bond10 l1    SP500 l1     GSCI l1     HFRI l1     NFCI l1 
       TRUE        TRUE        TRUE        TRUE        TRUE        TRUE       FALSE 
     PMI l1     PE_r l1     VC_r l1 Bond10_r l1  SP500_r l1   GSCI_r l1   HFRI_r l1 
       TRUE       FALSE       FALSE       FALSE       FALSE       FALSE       FALSE 
  NFCI_r l1 PE_index l2 VC_index l2   Bond10 l2    SP500 l2     GSCI l2     HFRI l2 
      FALSE        TRUE        TRUE        TRUE        TRUE        TRUE        TRUE 
    NFCI l2      PMI l2     PE_r l2     VC_r l2 Bond10_r l2  SP500_r l2   GSCI_r l2 
      FALSE        TRUE       FALSE       FALSE       FALSE       FALSE       FALSE 
  HFRI_r l2   NFCI_r l2 
      FALSE       FALSE
```
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot25.png)
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot24.png)


---






### Potential Modifications and Their Implications

#### 1. Changing the Lag Length (p)

**Description:** The lag length (p) determines how many past values of the variables are included in the model. The `lags_upbound_BIC` function is used to select an optimal lag length based on the Bayesian Information Criterion (BIC).

**Implications:**
- **Increasing p:** Including more lags can capture more historical dependencies, potentially improving model accuracy but increasing complexity and computational cost. This may help in better modeling the dynamics but could lead to overfitting, especially with limited data.
- **Decreasing p:** Reducing the number of lags simplifies the model and decreases computational burden, but it may miss important historical dependencies, leading to poorer model performance.

**Example Adjustment:**
```r
selected_lag <- 5  # Manually setting p to 5
```

#### 2. Changing the Augmentation Parameter (d)

**Description:** The parameter d accounts for the potential non-stationarity in the data. It is the maximum order of integration suspected in the time series.

**Implications:**
- **Increasing d:** Handling higher levels of integration (e.g., I(2) processes) reduces the risk of spurious regression but increases the model complexity and computational load. This is useful if there are strong indications of higher-order integration.
- **Decreasing d:** Assumes lower levels of integration (e.g., I(1) processes), simplifying the model but risking inaccurate results if higher-order integration is present.

**Example Adjustment:**
```r
results <- lapply(interest_variables, function(pair) {
  HDGC_VAR(GCpair = pair, data = data[, c(dependent_variable, pair$GCfrom)], p = selected_lag, d = 1, bound = 0.5 * nrow(data), parallel = TRUE)
})
```

#### 3. Changing the Bound Parameter

**Description:** The `bound` parameter controls the lower bound on the penalty parameter of the lasso, affecting the number of variables selected.

**Implications:**
- **Increasing bound (> 0.5 * nrow(data)):** Makes the lasso more restrictive, selecting fewer variables. This reduces overfitting but may miss important predictors, increasing type II error.
- **Decreasing bound (< 0.5 * nrow(data)):** Makes the lasso less restrictive, selecting more variables. This can capture more potential relationships but increases the risk of overfitting and type I error.

**Example Adjustment:**
```r
results <- lapply(interest_variables, function(pair) {
  HDGC_VAR(GCpair = pair, data = data[, c(dependent_variable, pair$GCfrom)], p = selected_lag, d = 2, bound = 0.3 * nrow(data), parallel = TRUE)
})
```

#### 4. Changing the Alpha Value

**Description:** The `alpha` parameter in the `Plot_GC_all` function sets the significance level for the Granger causality tests.

**Implications:**
- **Increasing alpha (> 0.01):** Allows for a higher type I error rate, which means more false positives. This is more lenient and can detect more causal relationships but increases the risk of detecting spurious causality.
- **Decreasing alpha (< 0.01):** Stricter significance level reduces type I error, leading to fewer false positives but may increase type II error, potentially missing true causal relationships.

**Example Adjustment:**
```r
Plot_GC_all(network, Stat_type = "FS_cor", alpha = 0.05, multip_corr = list(FALSE), directed = TRUE, layout = layout.circle, main = "Network", edge.arrow.size = .2, vertex.size = 5, vertex.color = c("lightblue"), vertex.frame.color = "blue", vertex.label.size = 2, vertex.label.color = "black", vertex.label.cex = 0.6, vertex.label.dist = 1, edge.curved = 0, cluster = list(TRUE, 5, "black", 0.8, 1, 0))
```

### Interpretation of Changes

#### Lag Length (p)
- **Higher p:** Captures more historical data, potentially improving model performance if the additional lags contain valuable information. However, it increases model complexity, risk of overfitting, and computational cost.
- **Lower p:** Simplifies the model, reducing computational cost and overfitting risk, but may fail to capture important temporal dependencies, leading to poorer performance.

#### Augmentation Parameter (d)
- **Higher d:** Ensures robustness against higher-order integration and avoids spurious regression. This is important if the data is suspected to be non-stationary at higher levels but comes at the cost of increased complexity.
- **Lower d:** Assumes simpler dynamics (e.g., I(0) or I(1)), reducing model complexity. This is suitable if there is confidence that the series are stationary or first-order integrated but risks inaccurate results if higher-order integration is present.

#### Bound Parameter
- **Higher bound:** Reduces overfitting by selecting fewer variables. This is beneficial for high-dimensional data but may exclude relevant predictors, increasing type II error.
- **Lower bound:** Includes more variables, capturing a wider range of potential causal relationships. This is useful if there are many relevant predictors but increases the risk of overfitting and type I error.

#### Alpha Value
- **Higher alpha:** More lenient, allowing more causal relationships to be detected. This can be useful in exploratory analyses but increases the risk of false positives (type I error).
- **Lower alpha:** Stricter, reducing the likelihood of false positives. This is important for confirmatory analyses but may miss true causal relationships (type II error).

### Conclusion

Optimizing the Granger causality analysis involves balancing model complexity, computational efficiency, and statistical robustness. By adjusting parameters such as lag length, augmentation, bound, and alpha, you can tailor the model to better fit the characteristics of your data and research objectives. It is essential to experiment with these parameters and validate the model's performance through robust statistical testing and domain-specific knowledge.


---
###Example










































```r
# Load necessary libraries
library(HDGCvar)
library(igraph)

# Load your data (assuming your data is saved as 'time_series_data.csv')
data <- read.csv("C:/Users/muham/Desktop/Time series data.csv")

# Set the dependent variable and the dataset
dependent_variable <- 'PE_index'
independent_variables <- c('VC_index', 'Bond10', 'SP500', 'GSCI', 'HFRI', 'NFCI', 'PMI', 'PE_r', 'VC_r', 'Bond10_r', 'SP500_r', 'GSCI_r', 'HFRI_r', 'NFCI_r', 'PMI_r')

# Select the lag length
selected_lag <- lags_upbound_BIC(data[, c(dependent_variable, independent_variables)], p_max = 10)
print(selected_lag)

# Prepare the list of interest variables
interest_variables <- lapply(independent_variables, function(var) {
  list(GCto = dependent_variable, GCfrom = var)
})

# Test for Granger causality for each variable
results <- lapply(interest_variables, function(pair) {
  HDGC_VAR(GCpair = pair, data = data[, c(dependent_variable, pair$GCfrom)], p = selected_lag, d = 2, bound = 0.5 * nrow(data), parallel = TRUE, n_cores = 4)
})

# Print results
print(results)

# Optional: Estimate the full network of causality and plot the estimated network with multiple testing correction
network <- HDGC_VAR_all(data[, c(dependent_variable, independent_variables)], p = selected_lag, d = 2, bound = 0.5 * nrow(data), parallel = TRUE, n_cores = 4)
Plot_GC_all(network, Stat_type = "FS_cor", alpha = 0.01, multip_corr = list(TRUE, method = "BH"), directed = TRUE, layout = layout.circle, main = "Network", edge.arrow.size = .2, vertex.size = 5, vertex.color = c("lightblue"), vertex.frame.color = "blue", vertex.label.size = 2, vertex.label.color = "black", vertex.label.cex = 0.6, vertex.label.dist = 1, edge.curved = 0, cluster = list(TRUE, 5, "black", 0.8, 1, 0))
```
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot14.png)
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot15.png)






Other Examples
---
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot04.png)
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot05.png)
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot06.png)
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot07.png)
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot08.png)
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot09.png)
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot10.png)
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot11.png)
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot12.png)
![Original Data Plot](https://github.com/kgmikhdad/HDGCvar/blob/kgmikhdad-files/Rplot13.png)
---
