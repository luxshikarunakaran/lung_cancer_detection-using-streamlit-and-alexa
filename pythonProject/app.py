import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import pyttsx3

# Load the model, scaler, and metrics
best_model = joblib.load('best_model_rf.pkl')
scaler = joblib.load('scaler.pkl')
metrics = joblib.load('model_metrics.pkl')  # Assuming you have saved model metrics like accuracy

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Title
st.title('Lung Cancer Detection App')

# Sidebar for user input
st.sidebar.header('Patient Information')


def get_user_input():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.slider('Age', 0, 100, 50)
    smoking = st.sidebar.selectbox('Smoking', (1, 2))
    yellow_fingers = st.sidebar.selectbox('Yellow Fingers', (1, 2))
    anxiety = st.sidebar.selectbox('Anxiety', (1, 2))
    peer_pressure = st.sidebar.selectbox('Peer Pressure', (1, 2))
    chronic_disease = st.sidebar.selectbox('Chronic Disease', (1, 2))
    fatigue = st.sidebar.selectbox('Fatigue', (1, 2))
    allergy = st.sidebar.selectbox('Allergy', (1, 2))
    wheezing = st.sidebar.selectbox('Wheezing', (1, 2))
    alcohol_consuming = st.sidebar.selectbox('Alcohol Consuming', (1, 2))
    coughing = st.sidebar.selectbox('Coughing', (1, 2))
    shortness_of_breath = st.sidebar.selectbox('Shortness of Breath', (1, 2))
    swallowing_difficulty = st.sidebar.selectbox('Swallowing Difficulty', (1, 2))
    chest_pain = st.sidebar.selectbox('Chest Pain', (1, 2))

    user_data = {
        'GENDER': 1 if gender == 'Male' else 0,
        'AGE': age,
        'SMOKING': smoking,
        'YELLOW_FINGERS': yellow_fingers,
        'ANXIETY': anxiety,
        'PEER_PRESSURE': peer_pressure,
        'CHRONIC_DISEASE': chronic_disease,
        'FATIGUE': fatigue,
        'ALLERGY': allergy,
        'WHEEZING': wheezing,
        'ALCOHOL_CONSUMING': alcohol_consuming,
        'COUGHING': coughing,
        'SHORTNESS_OF_BREATH': shortness_of_breath,
        'SWALLOWING_DIFFICULTY': swallowing_difficulty,
        'CHEST_PAIN': chest_pain
    }

    return pd.DataFrame(user_data, index=[0])


# Get user input
input_df = get_user_input()

# Main panel
st.subheader('Patient Information')
st.write(input_df)

# Standardize the user input
input_scaled = scaler.transform(input_df)

# Make predictions
prediction = best_model.predict(input_scaled)
prediction_proba = best_model.predict_proba(input_scaled)

# Display results
st.subheader('Prediction')
st.write('Lung Cancer' if prediction[0] else 'No Lung Cancer')

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Example of the "actual" value
y_true = [1]  # For demonstration purposes, assume this is the actual outcome.
y_pred = prediction.tolist()  # Model's predicted value

# Add a button to show actual vs predicted values in a bar graph
if st.button('Show Actual vs Predicted Bar Graph'):
    comparison_df = pd.DataFrame({
        'Type': ['Actual', 'Predicted'],
        'Lung Cancer': [y_true[0], y_pred[0]]
    })

    st.subheader('Actual vs Predicted')
    st.write(comparison_df)

    # Plot the bar graph
    fig, ax = plt.subplots()
    ax.bar(comparison_df['Type'], comparison_df['Lung Cancer'], color=['blue', 'orange'])
    ax.set_xlabel('Type')
    ax.set_ylabel('Lung Cancer (1: Yes, 0: No)')
    ax.set_title('Actual vs Predicted Lung Cancer')

    st.pyplot(fig)

# Add a button to ask for Alexa-style information
if st.button('Ask Alexa'):
    accuracy = metrics.get('accuracy', 'N/A')  # Assuming you have an accuracy value in your metrics
    diagnosis = 'Lung Cancer' if y_pred[0] == 1 else 'No Lung Cancer'
    actual_vs_predicted = f"Actual: {y_true[0]}, Predicted: {y_pred[0]}"
    alexa_response_text = (
        f"Based on the input data, the predicted value is {y_pred[0]} "
        f"and the actual value is {y_true[0]}. "
        f"The model's accuracy is {accuracy}. "
        f"The diagnosis is: {diagnosis}."
    )

    # Speak the Alexa response
    engine.say(alexa_response_text)
    engine.runAndWait()

    # Display the response on the app
    st.subheader('Alexa Response')
    st.write(alexa_response_text)

    # If predicted lung cancer, provide solution information
    if prediction[0] == 1:
        st.subheader('Lung Cancer Solutions')
        st.markdown('### What to Do Next:')
        st.write('- Consult with a healthcare professional immediately.')
        st.write('- Consider further diagnostic tests such as imaging scans (CT scan, MRI).')
        st.write('- Follow medical advice for treatment options.')

        st.markdown('### Lifestyle Changes:')
        st.write('- Quit smoking if applicable.')
        st.write('- Maintain a healthy diet and exercise regularly.')
        st.write('- Manage stress levels.')

        st.markdown('### Support Resources:')
        st.write('- Contact local healthcare facilities for specialized lung cancer support.')
        st.write('- Explore online resources for patient support groups.')

# Add disclaimer
st.markdown('---')
st.write('Disclaimer: This app provides informational content and does not substitute professional medical advice. Consult a healthcare professional for personalized medical assistance.')
