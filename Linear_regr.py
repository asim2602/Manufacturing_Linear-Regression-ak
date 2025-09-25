import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset (assuming you've done this already)
df = pd.read_csv('manufacturing_dataset_1000_samples.csv')

# Drop the first column as it's just a timestamp
df = df.iloc[:, 1:]

# Identify features (X) and target (y)
#all features
X = df[['Injection_Temperature', 'Injection_Pressure', 'Material_Viscosity']]
y = df['Parts_Per_Hour']

# --- ADD THIS CODE TO HANDLE MISSING VALUES ---
# Check for missing values in X
print(X.isnull().sum())

# Impute missing values in X with the mean of each column
X = X.fillna(X.mean())

# Verify that there are no more missing values
print("\nMissing values after imputation:")
print(X.isnull().sum())
# ---------------------------------------------

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
import pickle

# Save the model to a file
#write biinary
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as linear_regression_model.pkl")


print("\nModel training successful!")