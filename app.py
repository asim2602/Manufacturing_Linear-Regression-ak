import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Define the path to the saved model
model_path = 'linear_regression_model_all_features.pkl'

# Load the saved model
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: Model file '{model_path}' not found. Please run the Python script in the Canvas first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Set up the Streamlit app title and description
st.title('Manufacturing Parts Per Hour Predictor')
st.write('Use this app to predict the number of parts produced per hour based on manufacturing parameters.')

# Create a dictionary to hold the feature names and their default values.
# This ensures that the order of the features matches the trained model.
# NOTE: The one-hot encoded features must be included here with a default of 0.
feature_defaults = {
    'Injection_Temperature': 220.0,
    'Injection_Pressure': 130.0,
    'Cycle_Time': 30.0,
    'Cooling_Time': 15.0,
    'Material_Viscosity': 350.0,
    'Ambient_Temperature': 25.0,
    'Machine_Age': 5.0,
    'Operator_Experience': 10.0,
    'Maintenance_Hours': 50.0,
    'Temperature_Pressure_Ratio': 1.692,
    'Total_Cycle_Time': 45.0,
    'Efficiency_Score': 0.05,
    'Machine_Utilization': 0.6,
    # One-hot encoded categorical features
    'Shift_Night': 0,
    'Shift_Evening': 0,
    'Machine_Type_Type_B': 0,
    'Machine_Type_Type_C': 0,
    'Material_Grade_Standard': 0,
    'Day_of_Week_Monday': 0,
    'Day_of_Week_Saturday': 0,
    'Day_of_Week_Sunday': 0,
    'Day_of_Week_Thursday': 0,
    'Day_of_Week_Tuesday': 0,
    'Day_of_Week_Wednesday': 0
}

# Define a function to get user input for all features
def user_input_features():
    st.sidebar.header('Input Parameters')
    
    # Get user inputs for numerical features
    injection_temp = st.sidebar.slider('Injection Temperature', 150.0, 300.0, feature_defaults['Injection_Temperature'])
    injection_pressure = st.sidebar.slider('Injection Pressure', 100.0, 200.0, feature_defaults['Injection_Pressure'])
    cycle_time = st.sidebar.slider('Cycle Time', 15.0, 60.0, feature_defaults['Cycle_Time'])
    cooling_time = st.sidebar.slider('Cooling Time', 5.0, 30.0, feature_defaults['Cooling_Time'])
    material_viscosity = st.sidebar.slider('Material Viscosity', 200.0, 500.0, feature_defaults['Material_Viscosity'])
    ambient_temp = st.sidebar.slider('Ambient Temperature', 20.0, 35.0, feature_defaults['Ambient_Temperature'])
    machine_age = st.sidebar.slider('Machine Age', 0.0, 20.0, feature_defaults['Machine_Age'])
    operator_exp = st.sidebar.slider('Operator Experience', 0.0, 25.0, feature_defaults['Operator_Experience'])
    maintenance_hours = st.sidebar.slider('Maintenance Hours', 0.0, 100.0, feature_defaults['Maintenance_Hours'])
    
    # Get user inputs for categorical features using radio buttons or selectboxes
    shift = st.sidebar.radio('Shift', ['Day', 'Evening', 'Night'])
    machine_type = st.sidebar.radio('Machine Type', ['Type_A', 'Type_B', 'Type_C'])
    material_grade = st.sidebar.radio('Material Grade', ['Economy', 'Standard'])
    day_of_week = st.sidebar.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Create a dictionary with the inputs, including one-hot encoded values
    input_data = {
        'Injection_Temperature': injection_temp,
        'Injection_Pressure': injection_pressure,
        'Cycle_Time': cycle_time,
        'Cooling_Time': cooling_time,
        'Material_Viscosity': material_viscosity,
        'Ambient_Temperature': ambient_temp,
        'Machine_Age': machine_age,
        'Operator_Experience': operator_exp,
        'Maintenance_Hours': maintenance_hours,
        # Calculated features
        'Temperature_Pressure_Ratio': injection_temp / injection_pressure,
        'Total_Cycle_Time': cycle_time + cooling_time,
        'Efficiency_Score': (injection_temp / injection_pressure) / cycle_time,
        'Machine_Utilization': (cycle_time / (cycle_time + cooling_time + 10)) # A simplified example
    }
    
    # Add one-hot encoded features
    input_data['Shift_Night'] = 1 if shift == 'Night' else 0
    input_data['Shift_Evening'] = 1 if shift == 'Evening' else 0
    input_data['Machine_Type_Type_B'] = 1 if machine_type == 'Type_B' else 0
    input_data['Machine_Type_Type_C'] = 1 if machine_type == 'Type_C' else 0
    input_data['Material_Grade_Standard'] = 1 if material_grade == 'Standard' else 0
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday', 'Sunday']
    for day in days:
        input_data[f'Day_of_Week_{day}'] = 1 if day_of_week == day else 0

    # The model was trained with 'Friday' as the reference category, so it's not needed here
    
    # Ensure all features from the trained model are present, even if not directly from user input
    # This is a robust way to handle any missing features from the UI
    final_data = {key: input_data.get(key, 0) for key in feature_defaults.keys()}
    
    # Convert the dictionary to a pandas DataFrame in the correct order
    features = pd.DataFrame([final_data], columns=list(feature_defaults.keys()))
    return features

# Get the user's input features
df_input = user_input_features()

# Display the user inputs
st.subheader('User Input Parameters')
st.write(df_input)

# Make a prediction
prediction = model.predict(df_input)

# Display the prediction
st.subheader('Predicted Parts Per Hour')
st.write(f'{prediction[0]:.2f} parts per hour')
