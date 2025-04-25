
from google.colab import files
files.upload()

import pandas as pd

# Load your dataset
df = pd.read_csv("power consumption data.csv")



print(df.head())

# Create a new column with the total power consumption
df["Total_Power_Consumption"] = df["PowerConsumption_Zone1"] + df["PowerConsumption_Zone2"] + df["PowerConsumption_Zone3"]

print(df.head())

df['Datetime'] = pd.to_datetime(df['Datetime'], format='mixed')

df["hour"] = pd.to_datetime(df["Datetime"]).dt.hour
df["dayofweek"] = pd.to_datetime(df["Datetime"]).dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

df.set_index('Datetime', inplace=True)

print(df.head())

import pandas as pd


features = [
    "Temperature", "Humidity", "WindSpeed", "hour", "dayofweek", "is_weekend"
]

target = 'Total_Power_Consumption'


X = df[features]
y = df[target]

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Define the regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Define the hyperparameters grid to search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'bootstrap': [True]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid,
                           cv=5, verbose=2, n_jobs=-1, scoring='neg_mean_squared_error')

# Fit GridSearchCV on the training data
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)

# Since datetime is the index, we can use it directly from X_test
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
}, index=X_test.index)

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.plot(results_df.index, results_df['Actual'], label='Actual', color='blue')
plt.plot(results_df.index, results_df['Predicted'], label='Predicted', color='orange')
plt.title('Actual vs Predicted Power Consumption Over Time')
plt.xlabel('Datetime')
plt.ylabel('Power Consumption')
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


mae = mean_absolute_error(y_test, y_pred)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# R^2 Score (Coefficient of Determination)
r2 = r2_score(y_test, y_pred)

# Print results
print("📊 Regression Evaluation Metrics:")
print(f"MAE  (Mean Absolute Error):      {mae:.3f}")
print(f"MSE  (Mean Squared Error):       {mse:.3f}")
print(f"RMSE (Root Mean Squared Error):  {rmse:.3f}")
print(f"R²   (Coefficient of Determination): {r2:.3f}")

