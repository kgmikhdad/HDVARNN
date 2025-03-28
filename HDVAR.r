library(ggplot2)
library(readr)
library(scales)
# Load the dataset
data <- read_csv("C:/Users/muham/Desktop/Time series data.csv")
# Create a function to generate the dual y-axis plot
create_dual_axis_plot <- function(df, index_col, return_col, title, filename) {
ggplot(df, aes(x = as.Date(date, format = "%b-%y"))) +
geom_line(aes(y = !!sym(index_col), color = "Index"), size = 1) +
geom_line(aes(y = !!sym(return_col) * 100, color = "Return"), size = 1, linetype = "dashed") +
scale_y_continuous(
name = "Index Values",
sec.axis = sec_axis(~./100, name = "Return Values (%)")
) +
scale_x_date(date_labels = "%Y-%q", breaks = date_breaks("year")) +
labs(title = title, x = "Year and Quarter") +
theme_minimal() +
scale_color_manual(
name = "Series",
values = c("Index" = "blue", "Return" = "red")
) +
theme(legend.position = "bottom")
ggsave(filename, width = 10, height = 6)
}
# List of columns to plot
index_return_pairs <- list(
c("PE_index", "PE_r"),
c("VC_index", "VC_r"),
c("Bond10", "Bond10_r"),
c("SP500", "SP500_r"),
c("GSCI", "GSCI_r"),
c("HFRI", "HFRI_r"),
c("NFCI", "NFCI_r"),
c("PMI", "PMI_r")
)
# Generate plots for each pair
for (pair in index_return_pairs) {
index_col <- pair[1]
return_col <- pair[2]
title <- paste("Time Series Plot of", gsub("_", " ", index_col), "and", gsub("_", " ", return_col))
filename <- paste0("plot_", index_col, ".png")
create_dual_axis_plot(data, index_col, return_col, title, filename)
}
library(ggplot2)
library(readr)
library(scales)
# Load the dataset
data <- read_csv("C:/Users/muham/Desktop/Time series data.csv")
# Convert the 'date' column to Date type
data$date <- as.Date(data$date, format = "%b-%y")
# Create a function to generate the dual y-axis plot
create_dual_axis_plot <- function(df, index_col, return_col, title, filename) {
ggplot(df, aes(x = date)) +
geom_line(aes(y = !!sym(index_col), color = "Index"), linewidth = 1) +
geom_line(aes(y = !!sym(return_col) * 100, color = "Return"), linewidth = 1, linetype = "dashed") +
scale_y_continuous(
name = "Index Values",
sec.axis = sec_axis(~./100, name = "Return Values (%)")
) +
scale_x_date(date_labels = "%Y-%q", breaks = date_breaks("year")) +
labs(title = title, x = "Year and Quarter") +
theme_minimal() +
scale_color_manual(
name = "Series",
values = c("Index" = "blue", "Return" = "red")
) +
theme(legend.position = "bottom")
ggsave(filename, width = 10, height = 6)
}
# List of columns to plot
index_return_pairs <- list(
c("PE_index", "PE_r"),
c("VC_index", "VC_r"),
c("Bond10", "Bond10_r"),
c("SP500", "SP500_r"),
c("GSCI", "GSCI_r"),
c("HFRI", "HFRI_r"),
c("NFCI", "NFCI_r"),
c("PMI", "PMI_r")
)
# Generate plots for each pair
for (pair in index_return_pairs) {
index_col <- pair[1]
return_col <- pair[2]
title <- paste("Time Series Plot of", gsub("_", " ", index_col), "and", gsub("_", " ", return_col))
filename <- paste0("plot_", index_col, ".png")
create_dual_axis_plot(data, index_col, return_col, title, filename)
}
library(ggplot2)
library(readr)
library(scales)
# Load the dataset
data <- read_csv("C:/Users/muham/Desktop/Time series data.csv")
# Convert the 'date' column to Date type
data$date <- as.Date(paste(data$year, data$quarter * 3, "01", sep = "-"), format = "%Y-%m-%d")
# Create a function to generate the dual y-axis plot
create_dual_axis_plot <- function(df, index_col, return_col, title, filename) {
p <- ggplot(df, aes(x = date)) +
geom_line(aes(y = !!sym(index_col), color = "Index"), linewidth = 1) +
geom_line(aes(y = !!sym(return_col) * 100, color = "Return"), linewidth = 1, linetype = "dashed") +
scale_y_continuous(
name = "Index Values",
sec.axis = sec_axis(~./100, name = "Return Values (%)")
) +
scale_x_date(date_labels = "%Y-Q%q", date_breaks = "1 year") +
labs(title = title, x = "Year and Quarter") +
theme_minimal() +
scale_color_manual(
name = "Series",
values = c("Index" = "blue", "Return" = "red")
) +
theme(legend.position = "bottom")
ggsave(filename, plot = p, width = 10, height = 6)
}
# List of columns to plot
index_return_pairs <- list(
c("PE_index", "PE_r"),
c("VC_index", "VC_r"),
c("Bond10", "Bond10_r"),
c("SP500", "SP500_r"),
c("GSCI", "GSCI_r"),
c("HFRI", "HFRI_r"),
c("NFCI", "NFCI_r"),
c("PMI", "PMI_r")
)
# Generate plots for each pair
for (pair in index_return_pairs) {
index_col <- pair[1]
return_col <- pair[2]
title <- paste("Time Series Plot of", gsub("_", " ", index_col), "and", gsub("_", " ", return_col))
filename <- paste0("plot_", index_col, ".png")
create_dual_axis_plot(data, index_col, return_col, title, filename)
}
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
HDGC_VAR(GCpair = pair, data = data[, c(dependent_variable, pair$GCfrom)], p = selected_lag, d = 2, bound = 0.1 * nrow(data), parallel = TRUE, n_cores = 4)
})
# Print results
print(results)
# Optional: Estimate the full network of causality and plot the estimated network with multiple testing correction
network <- HDGC_VAR_all(data[, c(dependent_variable, independent_variables)], p = selected_lag, d = 2, bound = 0.1 * nrow(data), parallel = TRUE, n_cores = 4)
Plot_GC_all(network, Stat_type = "FS_cor", alpha = 0.05, multip_corr = list(TRUE, method = "BH"), directed = TRUE, layout = layout.circle, main = "Network", edge.arrow.size = .2, vertex.size = 5, vertex.color = c("lightblue"), vertex.frame.color = "blue", vertex.label.size = 2, vertex.label.color = "black", vertex.label.cex = 0.6, vertex.label.dist = 1, edge.curved = 0, cluster = list(TRUE, 5, "black", 0.8, 1, 0))
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
HDGC_VAR(GCpair = pair, data = data[, c(dependent_variable, pair$GCfrom)], p = selected_lag, d = 2, bound = 0.8 * nrow(data), parallel = TRUE, n_cores = 4)
})
# Print results
print(results)
# Optional: Estimate the full network of causality and plot the estimated network with multiple testing correction
network <- HDGC_VAR_all(data[, c(dependent_variable, independent_variables)], p = selected_lag, d = 2, bound = 0.8 * nrow(data), parallel = TRUE, n_cores = 4)
Plot_GC_all(network, Stat_type = "FS_cor", alpha = 0.05, multip_corr = list(TRUE, method = "BH"), directed = TRUE, layout = layout.circle, main = "Network", edge.arrow.size = .2, vertex.size = 5, vertex.color = c("lightblue"), vertex.frame.color = "blue", vertex.label.size = 2, vertex.label.color = "black", vertex.label.cex = 0.6, vertex.label.dist = 1, edge.curved = 0, cluster = list(TRUE, 5, "black", 0.8, 1, 0))
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
HDGC_VAR(GCpair = pair, data = data[, c(dependent_variable, pair$GCfrom)], p = selected_lag, d = 2, bound = 1 * nrow(data), parallel = TRUE, n_cores = 3)
})
# Print results
print(results)
# Optional: Estimate the full network of causality and plot the estimated network with multiple testing correction
network <- HDGC_VAR_all(data[, c(dependent_variable, independent_variables)], p = selected_lag, d = 2, bound = 1 * nrow(data), parallel = TRUE, n_cores = 3)
Plot_GC_all(network, Stat_type = "FS_cor", alpha = 0.05, multip_corr = list(TRUE, method = "BH"), directed = TRUE, layout = layout.circle, main = "Network", edge.arrow.size = .2, vertex.size = 5, vertex.color = c("lightblue"), vertex.frame.color = "blue", vertex.label.size = 2, vertex.label.color = "black", vertex.label.cex = 0.6, vertex.label.dist = 1, edge.curved = 0, cluster = list(TRUE, 5, "black", 0.8, 1, 0))
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
# Test for Granger causality for each variable with bound = 0.5
results_bound_0.5 <- lapply(interest_variables, function(pair) {
HDGC_VAR(GCpair = pair, data = data[, c(dependent_variable, pair$GCfrom)], p = selected_lag, d = 2, bound = 0.5 * nrow(data), parallel = TRUE, n_cores = 3)
})
# Print results for bound = 0.5
print(results_bound_0.5)
# Estimate the full network of causality and plot the estimated network with multiple testing correction for bound = 0.5
network_bound_0.5 <- HDGC_VAR_all(data[, c(dependent_variable, independent_variables)], p = selected_lag, d = 2, bound = 0.5 * nrow(data), parallel = TRUE, n_cores = 3)
Plot_GC_all(network_bound_0.5, Stat_type = "FS_cor", alpha = 0.05, multip_corr = list(TRUE, method = "BH"), directed = TRUE, layout = layout.circle, main = "Network (Bound = 0.5)", edge.arrow.size = .2, vertex.size = 5, vertex.color = c("lightblue"), vertex.frame.color = "blue", vertex.label.size = 2, vertex.label.color = "black", vertex.label.cex = 0.6, vertex.label.dist = 1, edge.curved = 0, cluster = list(TRUE, 5, "black", 0.8, 1, 0))
# Test for Granger causality for each variable with bound = 0.7
results_bound_0.7 <- lapply(interest_variables, function(pair) {
HDGC_VAR(GCpair = pair, data = data[, c(dependent_variable, pair$GCfrom)], p = selected_lag, d = 2, bound = 0.7 * nrow(data), parallel = TRUE, n_cores = 3)
})
# Print results for bound = 0.7
print(results_bound_0.7)
# Estimate the full network of causality and plot the estimated network with multiple testing correction for bound = 0.7
network_bound_0.7 <- HDGC_VAR_all(data[, c(dependent_variable, independent_variables)], p = selected_lag, d = 2, bound = 0.7 * nrow(data), parallel = TRUE, n_cores = 3)
Plot_GC_all(network_bound_0.7, Stat_type = "FS_cor", alpha = 0.05, multip_corr = list(TRUE, method = "BH"), directed = TRUE, layout = layout.circle, main = "Network (Bound = 0.7)", edge.arrow.size = .2, vertex.size = 5, vertex.color = c("lightblue"), vertex.frame.color = "blue", vertex.label.size = 2, vertex.label.color = "black", vertex.label.cex = 0.6, vertex.label.dist = 1, edge.curved = 0, cluster = list(TRUE, 5, "black", 0.8, 1, 0))
# Test for Granger causality for each variable with bound = 0.3
results_bound_0.3 <- lapply(interest_variables, function(pair) {
HDGC_VAR(GCpair = pair, data = data[, c(dependent_variable, pair$GCfrom)], p = selected_lag, d = 2, bound = 0.3 * nrow(data), parallel = TRUE, n_cores = 3)
})
# Print results for bound = 0.3
print(results_bound_0.3)
# Estimate the full network of causality and plot the estimated network with multiple testing correction for bound = 0.3
network_bound_0.3 <- HDGC_VAR_all(data[, c(dependent_variable, independent_variables)], p = selected_lag, d = 2, bound = 0.3 * nrow(data), parallel = TRUE, n_cores = 3)
Plot_GC_all(network_bound_0.3, Stat_type = "FS_cor", alpha = 0.05, multip_corr = list(TRUE, method = "BH"), directed = TRUE, layout = layout.circle, main = "Network (Bound = 0.3)", edge.arrow.size = .2, vertex.size = 5, vertex.color
# Load necessary libraries
library(HDGCvar)
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
results_multiple <- HDGC_VAR_multiple(GCpairs = variable_pairs, data = data_subset, p = selected_lag, d = 2, bound = 0.5 * nrow(data_subset), parallel = TRUE, n_cores = 3)
# Print results
print(results_multiple)
# Optional: Plot the estimated network for multiple combinations
network_multiple <- HDGC_VAR_all(data = data_subset, p = selected_lag, d = 2, bound = 0.5 * nrow(data_subset), parallel = TRUE, n_cores = 3)
Plot_GC_all(network_multiple, Stat_type = "FS_cor", alpha = 0.05, multip_corr = list(FALSE), directed = TRUE, layout = layout.circle, main = "Multiple Combinations Network", edge.arrow.size = .2, vertex.size = 5, vertex.color = c("lightblue"), vertex.frame.color = "blue", vertex.label.size = 2, vertex.label.color = "black", vertex.label.cex = 0.6, vertex.label.dist = 1, edge.curved = 0, cluster = list(TRUE, 5, "black", 0.8, 1, 0))
# Load necessary libraries
library(quantreg)
library(dplyr)
library(tidyr)
# Load the data
data <- read.csv("C:/Users/muham/Desktop/Paper two/RUSSIA/final_russia.csv")
# Preview the data
head(data)
# List of economic indicators and export categories
economic_indicators <- c("EPU", "PPI", "CPI")
export_categories <- colnames(data)[!(colnames(data) %in% economic_indicators)]
# Define quantiles of interest
quantiles <- seq(0.05, 0.95, by = 0.05)
# Function to perform quantile regression
quantile_regression <- function(y, x, quantiles) {
results <- list()
for (q in quantiles) {
model <- rq(y ~ x, tau = q)
results[[as.character(q)]] <- summary(model)
}
return(results)
}
# Loop through each combination of economic indicator and export category
results_list <- list()
for (indicator in economic_indicators) {
for (category in export_categories) {
y <- data[[indicator]]
x <- data[[category]]
results <- quantile_regression(y, x, quantiles)
results_list[[paste(indicator, category, sep = "_")]] <- results
}
}
# Load necessary libraries
library(quantreg)
library(dplyr)
library(tidyr)
# Load the data
data <- read.csv("C:/Users/muham/Desktop/Paper two/RUSSIA/final_russia.csv")
# Remove rows with NA/NaN/Inf values
data <- data %>%
mutate(across(everything(), ~na_if(., Inf))) %>%  # Convert Inf to NA
drop_na()
# Load necessary libraries
install.packages("quantreg")  # If not already installed
install.packages("dplyr")     # If not already installed
install.packages("tidyr")     # If not already installed
install.packages("network")   # For network analysis and visualization
install.packages("igraph")    # For network analysis and visualization
library(quantreg)
library(dplyr)
library(tidyr)
library(network)
library(igraph)
# Load the data
data <- read.csv("C:/Users/muham/Desktop/Paper two/RUSSIA/final_russia.csv")
# Remove rows with NA/NaN/Inf values
data <- data %>%
mutate(across(everything(), ~na_if(., Inf))) %>%
drop_na()
# Load necessary libraries
library(quantreg)
library(tseries)
library(ggplot2)
library(dplyr)
# Load the data
data <- read.csv("C:/Users/muham/Desktop/Paper two/RUSSIA/final_russia.csv")
# Function to check stationarity and difference if needed
make_stationary <- function(series) {
adf_test <- adf.test(series, alternative = "stationary")
if (adf_test$p.value > 0.05) {
return(diff(series, differences = 1))
} else {
return(series)
}
}
# Apply stationarity function to necessary columns
data$EPU <- make_stationary(data$EPU)
data$PET_EX_MON <- make_stationary(data$PET_EX_MON)
# Load necessary libraries
library(quantreg)
library(tseries)
library(ggplot2)
library(dplyr)
# Load the data
data <- read.csv("C:/Users/muham/Desktop/Paper two/RUSSIA/final_russia.csv")
# Function to check stationarity and difference if needed
make_stationary <- function(series) {
adf_test <- adf.test(series, alternative = "stationary")
if (adf_test$p.value > 0.05) {
return(diff(series, differences = 1))
} else {
return(series)
}
}
# Apply stationarity function to necessary columns
data$EPU <- c(NA, make_stationary(data$EPU))
# Load necessary libraries
library(quantreg)
library(tseries)
library(ggplot2)
library(dplyr)
# Load the data
data <- read.csv("C:/Users/muham/Desktop/Paper two/RUSSIA/final_russia.csv")
# Function to check stationarity and difference if needed
make_stationary <- function(series) {
adf_test <- adf.test(series, alternative = "stationary")
if (adf_test$p.value > 0.05) {
return(diff(series, differences = 1))
} else {
return(series)
}
}
# Apply stationarity function to necessary columns and pad with NA
data$EPU <- c(NA, make_stationary(data$EPU))
# Load necessary libraries
library(quantreg)
library(tseries)
library(ggplot2)
library(dplyr)
# Load the data
data <- read.csv("path_to_your_file/final_russia.csv")
# Load necessary libraries
library(quantreg)
library(tseries)
library(ggplot2)
library(dplyr)
# Load the data
data <- read.csv("C:/Users/muham/Desktop/Paper two/RUSSIA/final_russia.csv")
# Function to check stationarity and difference if needed
make_stationary <- function(series) {
adf_test <- adf.test(series, alternative = "stationary")
if (adf_test$p.value > 0.05) {
return(diff(series, differences = 1))
} else {
return(series)
}
}
# Apply stationarity function to necessary columns
data$EPU <- make_stationary(data$EPU)
data$PET_EX_MON <- make_stationary(data$PET_EX_MON)
