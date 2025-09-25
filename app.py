
#UI design
import streamlit as st
#reading saved model
import pickle
#read dataset and more
import pandas as pd

#read binary
# Load the saved model
# Load the saved model
try:
    with open('linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Set up the Streamlit app title and description
st.title('Parts Per Hour Predictor')
st.write('Use this app to predict the number of parts produced per hour based on manufacturing parameters.')

# Create input widgets for the user
st.sidebar.header('Input Parameters')
def user_input_features():
    injection_temp = st.sidebar.slider('Injection Temperature', 150.0, 300.0, 220.0)
    injection_pressure = st.sidebar.slider('Injection Pressure', 100.0, 200.0, 130.0)
    material_viscosity = st.sidebar.slider('Material Viscosity', 200.0, 500.0, 350.0)
    data = {'Injection_Temperature': injection_temp,
            'Injection_Pressure': injection_pressure,
            'Material_Viscosity': material_viscosity}
    #more params needed
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

# Display the user inputs
st.subheader('User Input Parameters')
st.write(df_input)

# Make a prediction
prediction = model.predict(df_input)

# Display the prediction
st.subheader('Predicted Parts Per Hour')
st.write(f'{prediction[0]:.2f} parts per hour')