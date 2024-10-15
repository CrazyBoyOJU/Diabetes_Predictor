import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title and description
st.title("Diabetes Prediction Web App")
st.markdown("""
This app predicts whether a person is diabetic based on medical details. 
Please fill in the required information below.
""")

# Input fields in columns
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200)
    blood_pressure = st.number_input("Blood Pressure Value", min_value=0, max_value=150)
    skin_thickness = st.number_input("Skin Thickness Value", min_value=0, max_value=100)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
    age = st.number_input("Age", min_value=1, max_value=120)

# Prediction button
if st.button("Predict"):
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    standardized_data = scaler.transform(input_data_as_numpy_array)
    
    prediction = model.predict(standardized_data)
    
    if prediction[0] == 1:
        st.success("The person is diabetic.")
    else:
        st.success("The person is not diabetic.")

#style
st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Display
st.image('diabetes_image.jpeg', caption='Diabetes Prediction', use_column_width=True)

# Sidebar info
st.sidebar.title("About")
st.sidebar.info("We here Predict the Likelihood of a Person having Diabetes based on certain information provided by the user.")
