# Customer-Segmentation-for-Targeted-Marketing

# RFM Analysis and Clustering Project

## Project Overview
This project performs RFM (Recency, Frequency, Monetary) analysis and clustering on an e-commerce dataset. The goal is to segment customers into clusters based on their purchasing behavior, visualize the results, and perform regression analysis to predict customer spending patterns.

## Dataset
- **File:** Online Retail.xlsx
- **Source:** Kaggle (Online Retail Dataset)
- **Sheet:** 1

## Objective
- Clean and preprocess the data.
- Perform RFM analysis to calculate Recency, Frequency, and Monetary metrics.
- Use K-means and hierarchical clustering techniques to segment customers.
- Visualize clusters using PCA (Principal Component Analysis).
- Perform regression analysis to evaluate relationships between RFM metrics.
- Generate insights and cluster-level summaries.

---

## Project Workflow

### 1. Data Loading and Cleaning
- The dataset is loaded using the `readxl` library.
- Rows with missing `CustomerID` values and negative/zero quantities or prices are removed.
- A new column `TotalPrice` is created by multiplying `Quantity` and `UnitPrice`.

### 2. RFM Calculation
- RFM metrics are calculated by grouping the data by `CustomerID`.
- Recency is measured from the latest purchase date relative to `2011-12-10`.
- Frequency and Monetary values are computed for each customer.
- Cleaned RFM data is exported to `Cleaned_RFM_Data.csv`.

### 3. Clustering
- Data is scaled using the `scale()` function.
- K-means clustering is performed to identify optimal clusters using the Elbow Method and Silhouette Score.
- Hierarchical clustering is applied, and a dendrogram is plotted.
- Cluster results are saved in `Clustered_Data.csv`.

### 4. Principal Component Analysis (PCA)
- PCA is conducted to reduce dimensionality.
- The top 2 principal components are visualized and saved as `PCA_Cluster_Visualization.png`.
- PCA data is exported to `PCA_Results.csv`.

### 5. Regression Analysis
- A linear regression model predicts Monetary value based on Recency and Frequency.
- Model performance is evaluated using MSE (Mean Squared Error) and R².
- Predicted results are saved to `Regression_Predictions.csv`.

### 6. Visualization and Plots
- Histograms are plotted to display distributions of RFM metrics:
  - `Recency_Distribution.png`
  - `Frequency_Distribution.png`
  - `Monetary_Distribution.png`
- Correlation matrices of RFM metrics are visualized in `Correlation_Matrix.png`.
- Cluster-level summaries are plotted to show average values for each metric across clusters.

---

## Output Files
- `Cleaned_RFM_Data.csv` – Cleaned and processed RFM data.
- `Clustered_Data.csv` – Cluster assignments for each customer.
- `PCA_Results.csv` – PCA-transformed data.
- `Regression_Predictions.csv` – Regression predictions for monetary values.
- Cluster Visualization Plots:
  - `PCA_Cluster_Visualization.png`
  - `Recency_Distribution.png`
  - `Frequency_Distribution.png`
  - `Monetary_Distribution.png`
  - `Correlation_Matrix.png`
  - `Avg_Recency_By_Cluster.png`
  - `Avg_Frequency_By_Cluster.png`
  - `Avg_Monetary_By_Cluster.png`

---

## Requirements
- R (version 4.0 or higher)
- Libraries:
  - `readxl`
  - `dplyr`
  - `cluster`
  - `ggplot2`
  - `caret`
  - `ggcorrplot`

Install the required packages using the following command:
```r
install.packages(c("readxl", "dplyr", "cluster", "ggplot2", "caret", "ggcorrplot"))
