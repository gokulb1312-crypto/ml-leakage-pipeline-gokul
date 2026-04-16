#Task 1 : Create a data & train 

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Create synthetic dataset
np.random.seed(42)

n = 60

area_sqft = np.random.randint(500, 3000, n)
num_bedrooms = np.random.randint(1, 5, n)
age_years = np.random.randint(0, 30, n)

# Price formula (with noise)
price_lakhs = (
    area_sqft * 0.05 +
    num_bedrooms * 10 -
    age_years * 0.3 +
    np.random.normal(0, 5, n)
)

# Create DataFrame
df = pd.DataFrame({
    'area_sqft': area_sqft,
    'num_bedrooms': num_bedrooms,
    'age_years': age_years,
    'price_lakhs': price_lakhs
})

# 2. Features and target
X = df[['area_sqft', 'num_bedrooms', 'age_years']]
y = df['price_lakhs']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Print intercept and coefficients
print("Intercept:", f"{model.intercept_:.2f}")

coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nCoefficients:")
print(coeff_df.round(2))

# 6. Predictions
y_pred = model.predict(X_test)

# Compare actual vs predicted
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print("\nFirst 5 Actual vs Predicted:")
print(comparison.head().round(2))

#Task 2 : Model Evaluation

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", f"{mae:.2f}")
print("RMSE:", f"{rmse:.2f}")
print("R²:", f"{r2:.2f}")

# Interpretation:
# MAE tells the average absolute error in predicted house prices (in lakhs).
# RMSE penalizes larger errors more heavily, indicating how spread out errors are.
# R² shows how much variance in price is explained by the model (closer to 1 is better).

#Task 3 — Residual Analysis

# Residuals
residuals = y_test - y_pred

# Plot histogram
plt.figure()
plt.hist(residuals, bins=15, edgecolor='black')
plt.title("Residuals Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Residual (Actual - Predicted)", fontsize=14, fontweight="bold")
plt.ylabel("Frequency", fontsize=14, fontweight="bold")
plt.show()

# Explanation:
# A residual is the difference between actual and predicted values.
# If the histogram is roughly symmetric and centered around zero,
# it suggests the model is unbiased and errors are normally distributed.