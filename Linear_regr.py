import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('manufacturing_dataset_1000_samples.csv')

# Drop the 'Timestamp' column as it's not a feature for the model
df = df.drop('Timestamp', axis=1)

# Identify features (X) and target (y)
# The target variable is 'Parts_Per_Hour'
y = df['Parts_Per_Hour']
# The features are all other columns
X = df.drop('Parts_Per_Hour', axis=1)

# --- Feature Engineering: Handle Categorical Variables ---
# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Apply one-hot encoding to the categorical columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# --- Pre-processing: Handle Missing Values ---
# Check for missing values in the prepared features
print("Missing values before imputation:")
print(X.isnull().sum())

# Impute missing values with the mean of each column
X = X.fillna(X.mean())

# Verify that there are no more missing values
print("\nMissing values after imputation:")
print(X.isnull().sum())

# --- Model Training and Testing ---
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# --- Test the model's performance on the test data ---
print("\n--- Model Performance Evaluation ---")
y_pred = model.predict(X_test)

# Calculate key metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R^2) Score: {r2:.4f}")

# Optional: Print the coefficients to see the impact of each feature
print("\n--- Model Coefficients ---")
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# --- Visualize the Model's Performance ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs. Actual')
plt.title('Actual vs. Predicted Parts Per Hour')
plt.xlabel('Actual Parts Per Hour')
plt.ylabel('Predicted Parts Per Hour')
plt.grid(True)

# Plot the perfect prediction line (y = x)
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

plt.legend()
plt.tight_layout()

# Save the plot to a file
plt.savefig('prediction_scatter_plot.png')
print("\nScatter plot saved as 'prediction_scatter_plot.png'")

# --- Save the trained model ---
with open('linear_regression_model_all_features.pkl', 'wb') as file:
    pickle.dump(model, file)

print("\nModel saved as linear_regression_model_all_features.pkl")
print("\nModel training, evaluation, and visualization complete!")
