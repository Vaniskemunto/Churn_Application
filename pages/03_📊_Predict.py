import json
import streamlit as st
from streamlit_lottie import st_lottie
import time
import datetime
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as impipeline
#from Transformer import LogTransformer, FeatureSelector

st.set_page_config(
    page_title = 'Predict Page',
    page_icon = '🏠',
    layout='wide'
)



st.title('Telcom Churn Prediction 🔮')

def load_Logistic_regression_pipeline():
    pipeline = joblib.load("./models/LogisticRegression_pipeline.joblib")
    return pipeline


def load_Gradient_boost_pipeline():
    pipeline = joblib.load("./models/GradientBoostingClassifier.joblib")
    return pipeline



@st.cache_data(show_spinner="Models Loading")
def cached_load_Gradient_boost_pipeline():
    return load_Gradient_boost_pipeline()


def select_model():
    col1, col2 = st.columns(2)

    with col1:
        selected_model = st.selectbox('Select a Model', options=['Logistic_regression', 'Gradient_boost'], key='selected_model')

    with col2:
        pass

 
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

if 'probability' not in st.session_state:
    st.session_state['probability'] = None

def make_prediction(pipeline, encoder):
    gender = st.session_state['gender']
    senior_citizen = st.session_state['SeniorCitizen']
    partner = st.session_state['Partner']
    dependents = st.session_state['Dependents']
    tenure = st.session_state['tenure']
    phone_service = st.session_state['PhoneService']
    multiple_lines = st.session_state['MultipleLines']
    internet_service = st.session_state['InternetService']
    online_security = st.session_state['OnlineSecurity']
    online_backup = st.session_state['OnlineBackup']
    device_protection = st.session_state['DeviceProtection']
    tech_support = st.session_state['TechSupport']
    streaming_tv = st.session_state['StreamingTV']
    streaming_movies = st.session_state['StreamingMovies']
    contract = st.session_state['Contract']
    paperless_billing = st.session_state['PaperlessBilling']
    payment_method = st.session_state['PaymentMethod']
    monthly_charges = st.session_state['MonthlyCharges']
    total_charges = st.session_state['TotalCharges']

    columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
               'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
               'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

    data = [[gender, senior_citizen, partner, dependents, tenure, phone_service,
             multiple_lines, internet_service, online_security, online_backup,
             device_protection, tech_support, streaming_tv, streaming_movies,
             contract, paperless_billing, payment_method, monthly_charges, total_charges]]

    # Create a DataFrame
    df = pd.DataFrame(data, columns=columns)

    df['PredictionTime'] = datetime.date.today()
    df['ModelUsed'] = st.session_state['selected_model']

    # Write to CSV
    df.to_csv('./data/history.csv', mode='a', header=not os.path.exists('./data/history.csv'), index=False)

    # Make prediction
    pred = pipeline.predict(df)
    pred = int(pred[0])
    prediction = encoder.inverse_transform([pred])

    # Get probabilities
    probability = pipeline.predict_proba(df)

    # Updating state
    st.session_state['prediction'] = prediction
    st.session_state['probability'] = probability

    return prediction, probability

def display_form():
    pipeline, encoder = select_model()

    with st.form('input-feature'):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write('### Personal Information')
            st.selectbox('Gender', options=['Male', 'Female'], key='gender')
            st.number_input('Senior Citizen', min_value=0, max_value=1, key='SeniorCitizen')
            st.selectbox('Partner', options=[0, 1], key='Partner')
            st.selectbox('Dependents', options=[0, 1], key='Dependents')
            st.number_input('Tenure', min_value=0, key='tenure')
            st.selectbox('Phone Service', options=[0, 1], key='PhoneService')
            st.selectbox('Multiple Lines', options=['No', 'Yes'], key='MultipleLines')

        with col2:
            st.write('### Work Information')
            st.selectbox('Internet Service', options=['DSL', 'Fiber optic', 'No'], key='InternetService')
            st.selectbox('Online Security', options=['No', 'Yes', 'No internet service'], key='OnlineSecurity')
            st.selectbox('Online Backup', options=['No', 'Yes', 'No internet service'], key='OnlineBackup')
            st.selectbox('Device Protection', options=['No', 'Yes', 'No internet service'], key='DeviceProtection')
            st.selectbox('Tech Support', options=['No', 'Yes', 'No internet service'], key='TechSupport')
            st.selectbox('Streaming TV', options=['No', 'Yes', 'No internet service'], key='StreamingTV')

        with col3:
            st.write('### Contract Information')
            st.selectbox('Streaming Movies', options=['No', 'Yes', 'No internet service'], key='StreamingMovies')
            st.selectbox('Contract', options=['Month-to-month', 'One year', 'Two year'], key='Contract')
            st.selectbox('Paperless Billing', options=[0, 1], key='PaperlessBilling')
            st.selectbox('Payment Method', options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], key='PaymentMethod')
            st.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=0.0, key='MonthlyCharges')
            st.number_input('Total Charges ($)', min_value=0.0, value=0.0, key='TotalCharges')
            #st.selectbox('Churn', options=['No', 'Yes'], key='Churn')
        
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            prediction, probability = make_prediction(pipeline, encoder)
            st.session_state['prediction'] = prediction
            st.session_state['probability'] = probability

if __name__ == "__main__":
    st.title("Make a Prediction")
    display_form()

    prediction = st.session_state['prediction']
    probability = st.session_state['probability']

    if prediction is None:
        st.markdown("### Predictions will show here")
    elif prediction == "Yes":
        probability_of_yes = probability[0][1] * 100
        st.markdown(f"### The employee will leave the company with a probability of {round(probability_of_yes, 2)}%")
    else:
        probability_of_no = probability[0][0] * 100
        st.markdown(f"### Employee will not leave the company with a probability of {round(probability_of_no, 2)}%")

    st.write(st.session_state)