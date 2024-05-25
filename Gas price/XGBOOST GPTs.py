import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd

# Load your data
file_path = r"C:\Users\7iCK\Desktop\10_features.csv"
df = pd.read_csv(file_path)

# Parse dates and set index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Creating lag features and targets, assuming 'Gas Close' is your target variable
features = df.drop(columns='Gas close').shift(1).dropna()
target = df['Gas close'].iloc[1:]

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit the XGBRegressor
xgb_model = XGBRegressor(n_estimators=50, max_depth=3)
xgb_model.fit(X_train_scaled, y_train)

# Predictions
predictions_xgb = xgb_model.predict(X_test_scaled)

# Performance metrics
r2_xgb = r2_score(y_test, predictions_xgb)
mse_xgb = mean_squared_error(y_test, predictions_xgb)
rmse_xgb = np.sqrt(mse_xgb)
mae_xgb = mean_absolute_error(y_test, predictions_xgb)

# Print performance metrics
print(f"R-squared (R2): {r2_xgb}")
print(f"Mean Squared Error (MSE): {mse_xgb}")
print(f"Root Mean Squared Error (RMSE): {rmse_xgb}")
print(f"Mean Absolute Error (MAE): {mae_xgb}")

# Plotting
index = np.arange(len(y_test))
plt.figure(figsize=(15, 7))
plt.plot(index, y_test, label='Actual', color='red')
plt.plot(index, predictions_xgb, label='Predicted', color='blue')
plt.title('XGBoost Actual vs Predicted - Next Day Gas Close Price')
plt.xlabel('Index')
plt.ylabel('Gas Close Price')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()
