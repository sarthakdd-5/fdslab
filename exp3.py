import numpy as np
import pandas as pd 
import math
# Load dataset
df = pd.read_csv('pune.csv')
df.columns = df.columns.str.strip()  # Clean whitespace from column names

# Automatically select valid numerical columns present in CSV
target_cols = [
    c for c in ['tempC', 'humidity', 'windspeed', 'DewPoint', 'cloudcover']
    if c in df.columns
]
# ACTIVITY 1: Single & Multi-Attribute Analysis
# (USING IN-BUILT FUNCTIONS)
print("==================================================")
print("ACTIVITY 1: Statistical Analysis (In-Built Functions)")
print("==================================================")

# Single attribute (tempC)
print("--- Single Attribute ('tempC') ---")

print(f"Mean:       {df['tempC'].mean():.4f}")
print(f"Median:     {df['tempC'].median():.4f}")
print(f"Mode:       {df['tempC'].mode().iloc[0]}")
print(f"Range:      {df['tempC'].max() - df['tempC'].min():.4f}")
print(f"Variance:   {df['tempC'].var():.4f}")
print(f"Std Dev:    {df['tempC'].std():.4f}")
print(f"IQR:        {df['tempC'].quantile(0.75) - df['tempC'].quantile(0.25):.4f}")

# Multi-attribute analysis
print("\n--- Multi-Attribute Summary ---")

multi_stats = pd.DataFrame({
    'Mean': df[target_cols].mean(),
    'Median': df[target_cols].median(),
    'Mode': df[target_cols].mode().iloc[0],
    'Range': df[target_cols].max() - df[target_cols].min(),
    'Variance': df[target_cols].var(),
    'Std Dev': df[target_cols].std(),
    'IQR': df[target_cols].quantile(0.75) -
           df[target_cols].quantile(0.25)
})

print(multi_stats)
# ACTIVITY 2: Manual Computation
# (WITHOUT IN-BUILT FUNCTIONS)
print("\n==================================================")
print("ACTIVITY 2: Procedure Without In-Built Functions")  
print("==================================================")


def custom_mean(data):
    return sum(data) / len(data)
def custom_median(data):
    sorted_d = sorted(data)
    n = len(sorted_d)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_d[mid - 1] + sorted_d[mid]) / 2.0

    return float(sorted_d[mid])

def custom_mode(data):
    counts = {}

    for x in data:
        counts[x] = counts.get(x, 0) + 1

    max_count = max(counts.values())

    modes = [
        k for k, v in counts.items()
        if v == max_count
    ]
    return modes[0] if len(modes) == 1 else modes


def custom_range(data):
    return max(data) - min(data)

def custom_variance(data, sample=True):
    m = custom_mean(data)

    denom = (
        (len(data) - 1)
        if (sample and len(data) > 1)
        else len(data)
    )

    return sum((x - m) ** 2 for x in data) / denom

def custom_std_dev(data, sample=True):
    return math.sqrt(custom_variance(data, sample))

def custom_percentile(data, p):
    sorted_d = sorted(data)

    k = (len(sorted_d) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return float(sorted_d[int(k)])
    return (
        sorted_d[int(f)] * (c - k)
        + sorted_d[int(c)] * (k - f)
    )
def custom_iqr(data):
    return (
        custom_percentile(data, 75)
        - custom_percentile(data, 25)
    )
# Execute manual function on sample data column
sample_data = df['tempC'].dropna().tolist()

print(f"Mean:       {custom_mean(sample_data):.4f}")
print(f"Median:     {custom_median(sample_data):.4f}")
print(f"Mode:       {custom_mode(sample_data)}")
print(f"Range:      {custom_range(sample_data):.4f}")
print(f"Variance:   {custom_variance(sample_data):.4f}")
print(f"Std Dev:    {custom_std_dev(sample_data):.4f}")
print(f"IQR:        {custom_iqr(sample_data):.4f}")

# ACTIVITY 3: Grouped Data Analysis
# (USING IN-BUILT FUNCTIONS)
print("\n==================================================")
print("ACTIVITY 3: Grouped Data Analysis (In-Built)")
print("==================================================")

# Grouping by categorical column if present
if 'location' in df.columns:

    print("--- Grouped Statistics by Location ---")
    grouped_location = df.groupby('location')[
        ['tempC', 'humidity']
    ].agg(
        ['count', 'mean', 'median', 'std', 'var']
    )
    print(grouped_location)

# Binned Frequency Distribution for Grouped Numerical Data
print("\n--- Temperature Frequency & Binning Distribution ---")
bins = [15, 20, 25, 30, 35]
df['tempC_grouped'] = pd.cut(
    df['tempC'],
    bins=bins
)

freq_distribution = (
    df['tempC_grouped']
    .value_counts()
    .sort_index()
)
print(freq_distribution)
