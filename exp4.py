import pandas as pd
import math

# Step 1 & 2: Load Dataset and Import math
df = pd.read_csv('pune.csv')
# Clean headers (strip spaces and trailing punctuation like open brackets)
df.columns = df.columns.str.strip().str.rstrip('(').str.rstrip(')')
# Helper function to find column names regardless of minor truncations
def get_col(candidates):
    for col in df.columns:
        for c in candidates:
            if c.lower() in col.lower():
                return col
    raise KeyError(f"None of {candidates} found in dataset: {df.columns.tolist()}")

temp_col = get_col(['tempC'])
feels_col = get_col(['FeelsLike', 'HeatIndex'])
humidity_col = get_col(['humidity'])
wind_col = get_col(['windspeed'])

# Step 3: Define Correlation Coefficient Function (Without in-built functions)
def compute_correlation(X, Y):
    n = len(X)
    mean_x = sum(X) / n
    mean_y = sum(Y) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(X, Y))
    sum_sq_x = sum((x - mean_x) ** 2 for x in X)
    sum_sq_y = sum((y - mean_y) ** 2 for y in Y)
    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    if denominator == 0:
        return 0.0
    return numerator / denominator

# Step 4: Extract Data Lists & Compute Correlation
df_pos = df[[temp_col, feels_col]].dropna()
x_pos, y_pos = df_pos[temp_col].tolist(), df_pos[feels_col].tolist()
df_neg = df[[temp_col, humidity_col]].dropna()
x_neg, y_neg = df_neg[temp_col].tolist(), df_neg[humidity_col].tolist()
df_zero = df[[temp_col, wind_col]].dropna()
x_zero, y_zero = df_zero[temp_col].tolist(), df_zero[wind_col].tolist()
r_pos = compute_correlation(x_pos, y_pos)
r_neg = compute_correlation(x_neg, y_neg)
r_zero = compute_correlation(x_zero, y_zero)

# Step 5: Print Correlation Results
print("==================================================")
print("Correlation Coefficients on Weather Dataset")
print("==================================================")
print(f"1. Positive Correlation ({temp_col} vs {feels_col}) : r = {r_pos:.4f}")
print(f"2. Negative Correlation ({temp_col} vs {humidity_col}): r = {r_neg:.4f}")
print(f"3. Weak/No Correlation  ({temp_col} vs {wind_col}): r = {r_zero:.4f}")

# Step 6: Terminal-Based ASCII Scatter Plot Function (No Matplotlib)
def ascii_scatter_plot(X, Y, title, width=50, height=15):
    print("\n" + "=" * 55)
    print(f"Scatter PLOT: {title}")
    print("=" * 55)
    min_x, max_x = min(X), max(X)
    min_y, max_y = min(Y), max(Y)
    # Handle zero range to avoid division by zero
    range_x = (max_x - min_x) if max_x != min_x else 1.0
    range_y = (max_y - min_y) if max_y != min_y else 1.0
    # Initialize empty grid canvas
    grid = [[" " for _ in range(width)] for _ in range(height)]
    # Map each data point to grid indices
    for x, y in zip(X, Y):
        col_idx = int((x - min_x) / range_x * (width - 1))
        row_idx = int((y - min_y) / range_y * (height - 1))
        # Invert row index so higher values appear at the top
        row_idx = height - 1 - row_idx
        grid[row_idx][col_idx] = "*"
    # Render top-to-bottom
    for r in range(height):
        y_val = max_y - (r / (height - 1)) * range_y
        row_str = "".join(grid[r])
        print(f"{y_val:6.1f} | {row_str}")
    
    print("       +" + "-" * width)
    print(f"        {min_x:<20.1f} {max_x:>27.1f}")

# Render terminal scatter plots
ascii_scatter_plot(x_pos, y_pos, f"Positive: {temp_col} vs {feels_col} (r = {r_pos:.2f})")
ascii_scatter_plot(x_neg, y_neg, f"Negative: {temp_col} vs {humidity_col} (r = {r_neg:.2f})")
ascii_scatter_plot(x_zero, y_zero, f"Weak/No: {temp_col} vs {wind_col} (r = {r_zero:.2f})")