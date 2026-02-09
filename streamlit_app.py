import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    model = joblib.load("depression_best_rs_gbt_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'depression_best_rs_gbt_model.pkl' is in the directory.")
    st.stop()

# Set up the title and a brief description
st.set_page_config(page_title="Student Depression Prediction", page_icon="📚", layout="wide")
st.title('📚 Student Depression Prediction')
st.write("""
This app predicts if a student has a high risk of depression based on various personal and academic factors.
Enter the student's details in the sidebar to see the prediction.
""")

# Create the user input interface in the sidebar
st.sidebar.header('Student Information')

# Define input options
gender_options = ['Male', 'Female']
academic_pressure_options = ['1', '2', '3', '4', '5']
study_satisfaction_options = ['1', '2', '3', '4', '5']
sleep_duration_options = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours', 'Others']
dietary_habits_options = ['Healthy', 'Moderate', 'Unhealthy']
degree_mapping = {
    'Bachelor of Pharmacy': 'B.Pharm',
    'Bachelor of Science': 'BSc',
    'Bachelor of Arts': 'BA',
    'Bachelor of Computer Applications': 'BCA',
    'Master of Technology': 'M.Tech',
    'PhD': 'PhD',
    'Bachelor of Education': 'B.Ed',
    'Bachelor of Laws': 'LLB',
    'Bachelor of Engineering': 'BE',
    'Master of Education': 'M.Ed',
    'Master of Science': 'MSc',
    'Bachelor of Hotel Management': 'BHM',
    'Master of Pharmacy': 'M.Pharm',
    'Master of Computer Applications': 'MCA',
    'Master of Arts': 'MA',
    'Bachelor of Commerce': 'B.Com',
    'Doctor of Medicine': 'MD',
    'Master of Business Administration': 'MBA',
    'Bachelor of Medicine, Bachelor of Surgery': 'MBBS',
    'Master of Commerce': 'M.Com',
    'Bachelor of Architecture': 'B.Arch',
    'Master of Laws': 'LLM',
    'Bachelor of Technology': 'B.Tech',
    'Bachelor of Business Administration': 'BBA',
    'Master of Engineering': 'ME',
    'Master of Hotel Management': 'MHM'
}
suicidal_thoughts_options = ['Yes', 'No']
financial_stress_options = ['1', '2', '3', '4', '5']
family_history_options = ['No', 'Yes']

# Collect Inputs in Sidebar
gender = st.sidebar.selectbox("Select Gender", gender_options)
age = st.sidebar.slider("Age", 16, 100, 20)
family_history = st.sidebar.selectbox("Family History of Mental Illnesses?", family_history_options)
academic_pressure = st.sidebar.selectbox("Select Academic Pressure", academic_pressure_options)
cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 4.0, 0.1)
study_hours = st.sidebar.slider("Study Hours per Day", 0, 12, 4)
study_satisfaction = st.sidebar.selectbox("Select Study Satisfaction", study_satisfaction_options)
sleep_duration = st.sidebar.selectbox("Select Sleep Duration", sleep_duration_options)
dietary_habits = st.sidebar.selectbox("Select Dietary Habits", dietary_habits_options)
degree_display = st.sidebar.selectbox("Select Course of Study", list(degree_mapping.keys()))
degree_val = degree_mapping[degree_display]
suicidal_thoughts = st.sidebar.selectbox("Have you ever had any Suicidal Thoughts?", suicidal_thoughts_options)
financial_stress = st.sidebar.selectbox("Select Financial Stress Level", financial_stress_options)

# Create a button to make a prediction
if st.sidebar.button('Predict Depression'):
    
    # Create DataFrame for display (showing full names)
    input_display = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Family History': [family_history],
        'Academic Pressure': [academic_pressure],
        'CGPA': [cgpa],
        'Study Hours': [study_hours],
        'Study Satisfaction': [study_satisfaction],
        'Sleep Duration': [sleep_duration],
        'Dietary Habits': [dietary_habits],
        'Degree': [degree_display],
        'Suicidal Thoughts': [suicidal_thoughts],
        'Financial Stress': [financial_stress]
    })

    # Create DataFrame for model (using mapped values and column names model expects)
    df_input = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'family_history': [family_history],
        'academic_pressure': [academic_pressure],
        'cgpa': [cgpa],
        'study_hours': [study_hours],
        'study_satisfaction': [study_satisfaction],
        'sleep_duration': [sleep_duration],
        'dietary_habits': [dietary_habits],
        'degree': [degree_val],
        'suicidal_thoughts': [suicidal_thoughts],
        'financial_stress': [financial_stress]
    })

    # One-hot encoding (preprocessing logic from app.py)
    df_encoded = pd.get_dummies(df_input, 
                              columns = ['gender', 'age', 'family_history', 'academic_pressure',
                                         'cgpa', 'study_hours', 'study_satisfaction', 'sleep_duration', 
                                         'dietary_habits', 'degree', 'suicidal_thoughts', 'financial_stress']
                             )
    
    # Align columns with model
    df_encoded = df_encoded.reindex(columns = model.feature_names_in_, fill_value=0)

    # Predict
    prediction = model.predict(df_encoded)[0]
    
    # Display the results
    st.header('Prediction Result')
    
    if prediction == 0:
        st.subheader('**You likely do not have Depression**=')
        st.markdown("however, if you have concerns about your mental health, consider seeking professional advice.")
    else:
        st.subheader('**You likely have Depression**')
        st.markdown("see a healthcare professional for a proper diagnosis and support.")

    # Display the input data for reference
    st.write("---")
    st.subheader("Student Data Used for Prediction:")
    st.table(input_display)
