import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Prediction of Disease Outbreaks",
                   layout='wide',
                   page_icon='🩺')

# Load models
diabetes_model = pickle.load(open(r"C:\Users\malli\Documents\Predictions\training_models\diabetes_model.sav", 'rb'))
heart_model = pickle.load(open(r"C:\Users\malli\Documents\Predictions\training_models\heart_model.sav", 'rb'))
parkinsons_model = pickle.load(open(r"C:\Users\malli\Documents\Predictions\training_models\parkinsons_model.sav", 'rb'))

# Sidebar menu
with st.sidebar:
    selected = option_menu("Prediction of Disease Outbreak System",
                           ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
                           menu_icon="hospital-fill",
                           icons=["activity", "heart", "person"],
                           default_index=0)

# Diabetes Prediction
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies", key="pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level", key="glucose")
    with col3:
        BloodPressure = st.text_input("Blood Pressure", key="blood_pressure")
    with col1:
        SkinThickness = st.text_input("Skin Thickness", key="skin_thickness")
    with col2:
        Insulin = st.text_input("Insulin Level", key="insulin")
    with col3:
        BMI = st.text_input("BMI Value", key="bmi")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function", key="dpf")
    with col2:
        Age = st.text_input("Age", key="age_diabetes")

    diab_diagnosis = ""

    if st.button("Diabetes Test Result"):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        try:
            user_input = [float(x) for x in user_input]
            diab_prediction = diabetes_model.predict([user_input])
            diab_diagnosis = "The person is diabetic" if diab_prediction[0] == 1 else "The person is not diabetic"
        except ValueError:
            diab_diagnosis = "Please enter valid numeric values!"

    st.success(diab_diagnosis)

# Heart Disease Prediction
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.text_input("Age", key="age_heart")
    with col2:
        Sex = st.text_input("Sex (0 = Female, 1 = Male)", key="sex")
    with col3:
        CP = st.text_input("Chest Pain Type", key="cp")
    with col1:
        Trestbps = st.text_input("Resting Blood Pressure", key="trestbps")
    with col2:
        Chol = st.text_input("Serum Cholesterol (mg/dl)", key="chol")
    with col3:
        FBS = st.text_input("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", key="fbs")
    with col1:
        Restecg = st.text_input("Resting Electrocardiographic Results", key="restecg")
    with col2:
        Thalach = st.text_input("Maximum Heart Rate Achieved", key="thalach")
    with col3:
        Exang = st.text_input("Exercise Induced Angina (1 = Yes, 0 = No)", key="exang")
    with col1:
        Oldpeak = st.text_input("ST Depression Induced by Exercise", key="oldpeak")
    with col2:
        Slope = st.text_input("Slope of the Peak Exercise ST Segment", key="slope")
    with col3:
        CA = st.text_input("Number of Major Vessels Colored by Fluoroscopy", key="ca")
    with col1:
        Thal = st.text_input("Thalassemia (0 = Normal, 1 = Fixed, 2 = Reversible Defect)", key="thal")

    heart_diagnosis = ""

    if st.button("Heart Test Result"):
        user_input = [Age, Sex, CP, Trestbps, Chol, FBS, Restecg, Thalach, Exang, Oldpeak, Slope, CA, Thal]
        try:
            user_input = [float(x) for x in user_input]
            heart_prediction = heart_model.predict([user_input])
            heart_diagnosis = "The person has heart disease" if heart_prediction[0] == 1 else "The person does not have heart disease"
        except ValueError:
            heart_diagnosis = "Please enter valid numeric values!"

    st.success(heart_diagnosis)

# Parkinson's Prediction
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Prediction using ML")

    feature_names = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ",
        "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
        "NHR", "HNR", "RPDE", "DFA", "Spread1", "Spread2", "D2", "PPE"
    ]

    input_values = []

    # Create 5 columns dynamically
    columns = st.columns(5)

    for i, feature in enumerate(feature_names):
        with columns[i % 5]:  # Distribute inputs across columns
            value = st.text_input(feature, key=f"parkinson_{i}")
            input_values.append(value)

    parkinsons_diagnosis = ""

    if st.button("Parkinson's Test Result"):
        try:
            user_input = [float(x) for x in input_values]
            parkinsons_prediction = parkinsons_model.predict([user_input])
            parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
        except ValueError:
            parkinsons_diagnosis = "Please enter valid numeric values!"

    st.success(parkinsons_diagnosis)
