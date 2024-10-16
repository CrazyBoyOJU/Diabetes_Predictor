import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for sleek and modern design
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        color: #3d5afe;
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0;
    }
    .description {
        color: #606060;
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton > button {
        background-color: #3d5afe;
        color: white;
        font-size: 18px;
        padding: 15px 25px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #536dfe;
        transform: translateY(-3px);
    }
    .stNumberInput > div > div > input {
        border-radius: 12px;
        border: 1px solid #ccc;
        padding: 10px;
        font-size: 16px;
    }
    .result-container {
        text-align: center;
        padding: 20px;
        margin-top: 20px;
        border-radius: 10px;
        font-size: 22px;
        font-weight: bold;
    }
    .result-diabetic {
        background-color: #ffdddd;
        color: #d32f2f;
        border: 1px solid #d32f2f;
    }
    .result-non-diabetic {
        background-color: #e7f9ed;
        color: #4caf50;
        border: 1px solid #4caf50;
    }
    .stImage > img {
        border-radius: 15px;
        margin-top: 30px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .sidebar-text {
        font-size: 16px;
        line-height: 1.6;
        color: #7f8a78;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.markdown('<div class="main-header">Diabetes Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Predict whether you are at risk of diabetes based on your health metrics. Enter the details below and receive an instant prediction.</div>', unsafe_allow_html=True)

# Input fields in a grid layout for better user experience
st.markdown("### Patient Information")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200)

with col2:
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100)
    insulin = st.number_input("Insulin Level (IU/mL)", min_value=0, max_value=900)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=80.0, step=0.1)

with col3:
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120)

# Prediction button with result display
if st.button("Predict"):
    # Preparing data for prediction
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    standardized_data = scaler.transform(input_data_as_numpy_array)
    
    # Prediction logic
    prediction = model.predict(standardized_data)
    
    # Display the result with red for diabetic and green for non-diabetic
    if prediction[0] == 1:
        st.markdown('<div class="result-container result-diabetic">The prediction result: Diabetic</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-container result-non-diabetic">The prediction result: Not Diabetic</div>', unsafe_allow_html=True)

# Display image for aesthetic appeal
st.image('diabetes_image.jpeg', caption='Stay Informed, Stay Healthy', use_column_width=True)

# Sidebar for additional info and credits
st.sidebar.title("About")
st.sidebar.markdown("""
<div class="sidebar-text">
This Diabetes Prediction App is designed to help you understand your risk of diabetes. By entering basic health metrics such as glucose level, BMI, and more, you will receive an instant prediction using machine learning. 
</div>
""", unsafe_allow_html=True)

# Sidebar for contact details
st.sidebar.markdown("""
<div class="sidebar-text">
\n**Contact Us**  
If you have any questions or need support, feel free to reach out:  
📧 **Email:** viveksupport@diabetesapp.com  
📞 **Phone:** +91 9854631459
</div>
""", unsafe_allow_html=True)
