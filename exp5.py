import pandas as pd
import numpy as np

# 1. Load Dataset
df = pd.read_csv('pune.csv')

# --- SIMPLE LINEAR REGRESSION ---
# Y = w0 + w1 * X (Predicting FeelsLikeC from tempC)
data_slr = df[['tempC', 'FeelsLikeC']].dropna()
X_slr = data_slr['tempC'].values
Y_slr = data_slr['FeelsLikeC'].values

mean_x = np.mean(X_slr)
mean_y = np.mean(Y_slr)

# Formulas from manual: w1 = sum((x - mean_x)*(y - mean_y)) / sum((x - mean_x)^2)
w1 = np.sum((X_slr - mean_x) * (Y_slr - mean_y)) / np.sum((X_slr - mean_x)**2)
w0 = mean_y - w1 * mean_x

print(f"SLR Model: FeelsLikeC = {w0:.4f} + {w1:.4f} * tempC")

# --- MULTIPLE LINEAR REGRESSION ---
# Y = b0 + b1*X1 + b2*X2 + b3*X3
# Matrix Formula: b = (X^T * X)^(-1) * X^T * Y
data_mlr = df[['tempC', 'humidity', 'DewPointC', 'FeelsLikeC']].dropna()
X_mlr = data_mlr[['tempC', 'humidity', 'DewPointC']].values
Y_mlr = data_mlr['FeelsLikeC'].values.reshape(-1, 1)

# Add column of 1s for intercept b0
X_mat = np.column_stack((np.ones(len(X_mlr)), X_mlr))

# Calculate coefficients using Normal Equation: b = inv(X^T * X) * X^T * Y
b = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ Y_mlr

print(f"MLR Model: FeelsLikeC = {b[0][0]:.4f} + {b[1][0]:.4f}*(tempC) + {b[2][0]:.4f}*(humidity) + {b[3][0]:.4f}*(DewPointC)")

# --- ACCURACY EVALUATION ---
# Predictions
pred_slr = w0 + w1 * X_slr
pred_mlr = (X_mat @ b).flatten()

# Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
mae_slr = np.mean(np.abs(Y_slr - pred_slr))
rmse_slr = np.sqrt(np.mean((Y_slr - pred_slr)**2))

mae_mlr = np.mean(np.abs(Y_mlr.flatten() - pred_mlr))
rmse_mlr = np.sqrt(np.mean((Y_mlr.flatten() - pred_mlr)**2))

print("\n--- Model Evaluation ---")
print(f"Simple Linear Regression   -> MAE: {mae_slr:.4f}, RMSE: {rmse_slr:.4f}")
print(f"Multiple Linear Regression -> MAE: {mae_mlr:.4f}, RMSE: {rmse_mlr:.4f}")

# --- MISSING VALUE IMPUTATION EXAMPLE ---
# Example: Predict missing 'FeelsLikeC' when tempC=32, humidity=65, DewPointC=21
sample_temp, sample_hum, sample_dew = 32.0, 65.0, 21.0

imputed_slr = w0 + w1 * sample_temp
imputed_mlr = b[0][0] + b[1][0]*sample_temp + b[2][0]*sample_hum + b[3][0]*sample_dew

print("\n--- Imputation Results for Missing Data Point ---")
print(f"Imputed 'FeelsLikeC' (SLR): {imputed_slr:.2f}°C")
print(f"Imputed 'FeelsLikeC' (MLR): {imputed_mlr:.2f}°C")