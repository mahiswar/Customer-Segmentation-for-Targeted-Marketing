library(readxl)
library(dplyr)
library(cluster)
library(ggplot2)
library(ggcorrplot)
library(caret)

# Load and Clean Data
dataset_path <- "C:\\Users\\desir\\Downloads\\Online Retail.xlsx"
data <- read_excel(dataset_path, sheet = 1)

cleaned_data <- data %>%
  filter(!is.na(CustomerID)) %>%           
  filter(Quantity > 0, UnitPrice > 0) %>% 
  mutate(TotalPrice = Quantity * UnitPrice)  

# Fix Recency Calculation
current_date <- as.Date("2011-12-10") 
rfm_data <- cleaned_data %>%
  group_by(CustomerID) %>%
  summarise(
    Recency = as.numeric(difftime(current_date, max(InvoiceDate), units = "days")),
    Frequency = n(),
    Monetary = sum(TotalPrice)
  )

write.csv(rfm_data, "Cleaned_RFM_Data.csv", row.names = FALSE)

# Scale Data
rfm_scaled <- scale(rfm_data[, -1])

# K-Means Clustering
set.seed(123)
wcss <- numeric(10)
silhouette_scores <- numeric(10)

for (k in 2:10) {  # Silhouette score is undefined for k = 1
  kmeans_model <- kmeans(rfm_scaled, centers = k)
  wcss[k] <- sum(kmeans_model$withinss)
  
  # Silhouette Score Calculation
  sil <- silhouette(kmeans_model$cluster, dist(rfm_scaled))
  silhouette_scores[k] <- mean(sil[, 3])
}

# Elbow Method Plot
plot(1:10, wcss, type = "b", xlab = "Number of Clusters", ylab = "WCSS", 
     main = "Elbow Method for Optimal Clusters")

# Silhouette Score Plot
plot(2:10, silhouette_scores[2:10], type = "b", xlab = "Number of Clusters", 
     ylab = "Average Silhouette Score", main = "Silhouette Score for Clustering")

# Optimal K and K-Means Results
optimal_k <- 3
kmeans_result <- kmeans(rfm_scaled, centers = optimal_k)

# Add Clusters to RFM Data
rfm_data$Cluster <- kmeans_result$cluster
write.csv(rfm_data, "Clustered_Data.csv", row.names = FALSE)

# Silhouette Plot for Final Clusters
silhouette_score <- silhouette(kmeans_result$cluster, dist(rfm_scaled))
plot(silhouette_score, main = "Silhouette Plot for K-Means Clustering")

# Hierarchical Clustering
hclust_result <- hclust(dist(rfm_scaled), method = "ward.D2")
plot(hclust_result, main = "Dendrogram for Hierarchical Clustering")

# PCA for Dimensionality Reduction
pca_result <- prcomp(rfm_scaled, center = TRUE, scale. = TRUE)
screeplot(pca_result, type = "lines", main = "Scree Plot")

pca_data <- as.data.frame(pca_result$x[, 1:2])
pca_data$Cluster <- as.factor(rfm_data$Cluster)
write.csv(pca_data, "PCA_Results.csv", row.names = FALSE)

# Visualization: PCA Cluster Plot
ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3) +
  theme_minimal() +
  labs(title = "PCA Cluster Visualization", x = "Principal Component 1", 
       y = "Principal Component 2", color = "Cluster")

ggsave("PCA_Cluster_Visualization.png")

# Regression Analysis with Evaluation
set.seed(123)
trainIndex <- createDataPartition(rfm_data$Monetary, p = 0.7, list = FALSE)
train <- rfm_data[trainIndex,]
test <- rfm_data[-trainIndex,]

linear_model <- lm(Monetary ~ Recency + Frequency, data = train)
predictions <- predict(linear_model, test)
mse <- mean((test$Monetary - predictions)^2)
r2 <- summary(linear_model)$r.squared
print(paste("MSE:", mse, "R²:", r2))

rfm_data$Predicted_Monetary <- predict(linear_model, rfm_data)
write.csv(rfm_data, "Regression_Predictions.csv", row.names = FALSE)

# Histograms for RFM Features
ggplot(rfm_data, aes(x = Recency)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Recency Distribution", x = "Recency (Days)", y = "Frequency")

ggsave("Recency_Distribution.png")

ggplot(rfm_data, aes(x = Frequency)) +
  geom_histogram(bins = 30, fill = "green", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Frequency Distribution", x = "Frequency", y = "Count")

ggsave("Frequency_Distribution.png")

ggplot(rfm_data, aes(x = Monetary)) +
  geom_histogram(bins = 30, fill = "red", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Monetary Distribution", x = "Monetary Value", y = "Count")

ggsave("Monetary_Distribution.png")

# Correlation Matrix for RFM Features
cor_matrix <- cor(rfm_data[, -c(1, 5)])  # Exclude CustomerID and Cluster
corr_plot <- ggcorrplot(cor_matrix, method = "circle", type = "lower", lab = TRUE) +
  labs(title = "Correlation Matrix of RFM Features") +
  theme_minimal()

ggsave("Correlation_Matrix.png", plot = corr_plot)
print(corr_plot)

# Cluster-Level Summary (New Addition)
cluster_summary <- rfm_data %>%
  group_by(Cluster) %>%
  summarise(
    Avg_Recency = mean(Recency),
    Avg_Frequency = mean(Frequency),
    Avg_Monetary = mean(Monetary),
    Count = n()
  )
print(cluster_summary)
write.csv(cluster_summary, "Cluster_Summary.csv", row.names = FALSE)

# Visualization of Cluster Characteristics
ggplot(cluster_summary, aes(x = factor(Cluster), y = Avg_Recency, fill = factor(Cluster))) +
  geom_bar(stat = "identity") +
  labs(title = "Average Recency by Cluster", x = "Cluster", y = "Average Recency") +
  theme_minimal()

ggsave("Avg_Recency_By_Cluster.png")

ggplot(cluster_summary, aes(x = factor(Cluster), y = Avg_Frequency, fill = factor(Cluster))) +
  geom_bar(stat = "identity") +
  labs(title = "Average Frequency by Cluster", x = "Cluster", y = "Average Frequency") +
  theme_minimal()

ggsave("Avg_Frequency_By_Cluster.png")

ggplot(cluster_summary, aes(x = factor(Cluster), y = Avg_Monetary, fill = factor(Cluster))) +
  geom_bar(stat = "identity") +
  labs(title = "Average Monetary by Cluster", x = "Cluster", y = "Average Monetary") +
  theme_minimal()

ggsave("Avg_Monetary_By_Cluster.png")

# End of Code
print("Code execution completed successfully.")
