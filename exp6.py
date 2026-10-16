import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Step 1: Load and clean dataset
df = pd.read_csv('pune.csv')
df_clean = df.dropna(subset=['tempC', 'humidity', 'pressure']).copy()

# Step 2: Apply 3 Data Normalization Techniques on 'tempC'
# 1. Min-Max Normalization [0, 1]
df_clean['tempC_MinMax'] = (df_clean['tempC'] - df_clean['tempC'].min()) / (df_clean['tempC'].max() - df_clean['tempC'].min())

# 2. Z-Score Normalization
df_clean['tempC_ZScore'] = (df_clean['tempC'] - df_clean['tempC'].mean()) / df_clean['tempC'].std()

# 3. Decimal Scaling
j = np.ceil(np.log10(df_clean['tempC'].abs().max()))
df_clean['tempC_Decimal'] = df_clean['tempC'] / (10 ** j)

# Step 3: Apply K-Means Discretization
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean['tempC_Cluster'] = kmeans.fit_predict(df_clean[['tempC']])

# Map cluster indices to human-readable categories based on centroid order
centroids = kmeans.cluster_centers_.flatten()
sorted_indices = np.argsort(centroids)
label_mapping = {sorted_indices[0]: 'Low Temp', sorted_indices[1]: 'Moderate Temp', sorted_indices[2]: 'High Temp'}
df_clean['tempC_Category'] = df_clean['tempC_Cluster'].map(label_mapping)

# Step 4: Display Consolidated Output
print("=== CONSOLIDATED NORMALIZATION OUTPUT ===")
print(df_clean[['tempC', 'tempC_MinMax', 'tempC_ZScore', 'tempC_Decimal']].head(10))

print("\n=== K-MEANS DISCRETIZATION OUTPUT ===")
print(df_clean[['date_time', 'tempC', 'tempC_Cluster', 'tempC_Category']].head(10))

print("\nComputed Centroids (°C):", centroids)
