import streamlit as st
#from credit.entity import artifact_entity,config_entity
#from credit.exception import CreditException
#from sklearn.pipeline import Pipeline
#from credit.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact,ModelPusherArtifact
#from credit.logger import logging
#from credit.predictor import ModelResolver
import pandas as pd
#from credit.utils import load_object,save_object
#from credit import utils
import os,sys
#from datetime import datetime
import numpy as np
import pickle

with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

st.title('Classifying Credit')
st.markdown('model to classify Credit into \
                (setosa, versicolor, virginica) based on their sepal/petal \
                and length/width.')

st.header("Credit data")
col1, col2 = st.columns(2)

with col1:
                #st.text("Sepal characteristics")
                status_options = {
    "... < 0 DM": 1,
    "0 <= ... < 200 DM": 2,
    "... >= 200 DM / salary assignments for at least 1 year": 3,
    "no checking account": 4
                                 }

                selected_status = st.selectbox("Select Status:", list(status_options.keys()), index=0)
                status = status_options[selected_status]

                duration = st.number_input("Enter duration in months:",step=60)

                credit_history_options = {
    "no credits taken/ all credits paid back duly": 0,
    "all credits at this bank paid back duly": 1,
    "existing credits paid back duly till now": 2,
    "delay in paying off in the past": 3,
    "critical account/ other credits existing (not at this bank)":4 
                                 }

                selected_credit_history = st.selectbox("Select Status:", list(credit_history_options.keys()), index=0)
                credit_history = credit_history_options[selected_credit_history]

                purpose_options = {
    "car (new)": 0,
    "car (used)": 1,
    "furniture/equipment": 2,
    "radio/television": 3,
    "domestic appliances": 4,
    "repairs": 5,
    "education": 6,
    "vacation ": 7,
    "retraining": 8,
    "business": 9,
    "others": 10

                                 }

                selected_purpose = st.selectbox("Select Purpose:", list(purpose_options.keys()), index=0)
                purpose = purpose_options[selected_purpose]


                amount = st.number_input("Enter Credit amount:",step=10000)

                savings_options = {
    "... < 100 DM": 1,
    "100 <= ... < 500 DM": 2,
    "500 <= ... < 1000 DM": 3,
    ".. >= 1000 DM": 4,
    "unknown/ no savings account": 5
                                  }
                selected_savings = st.selectbox("Select Savings account/bonds:", list(savings_options.keys()), index=0)
                savings = savings_options[selected_savings]

                employment_options = {
    "Unemployed": 1,
    "... < 1 year": 2,
    "1 <= ... < 4 years": 3,
    "4 <= ... < 7 years": 4,
    ".. >= 7 years": 5
                                  }
                selected_employment = st.selectbox("Select Savings account/bonds:", list(employment_options.keys()), index=0)
                employment_duration = employment_options[selected_employment]
    
            
                installment_rate = st.number_input("Enter Installment rate in percentage of disposable income",step=1)

                personal_status_sex_options = {
    "Male : divorced/separated": 1,
    "Female : divorced/separated/married": 2,
    "Male : single": 3,
    "Male : married/widowed": 4,
    "Female : single": 5
                                  }
                selected_personal_status = st.selectbox("Select Savings account/bonds:", list(personal_status_sex_options.keys()), index=0)
                personal_status_sex = personal_status_sex_options[selected_personal_status]
                
                
                
            
                
                age = st.slider('age', 2.0, 4.4, 0.5)
                number_credits = st.slider('number_credits', 2.0, 4.4, 0.5)
                job = st.slider('job', 2.0, 4.4, 0.5)

with col2:
                #st.text("Pepal characteristics")
                property_options = {
    "Real estate": 1,
    "if not option1 : building society savings agreement/ life insurance": 2,
    "if not option1/option2 : car or other, not in attribute 6": 3,
    "unknown / no property": 4
                                  }
                selected_property = st.selectbox("Select Property:", list(property_options.keys()), index=0)
                propertyl = property_options[selected_property]


                people_liable = st.slider('people_liable', 1.0, 7.0, 0.5)
                telephone = st.slider('telephone', 0.1, 2.5, 0.5)

st.text('')
if st.button("Predict type of Iris"):
    result = model.predict(
                    np.array([[status, duration, credit_history, purpose, amount, savings, employment_duration, installment_rate, personal_status_sex, propertyl, age, number_credits, job, people_liable, telephone]]))
  
 
    st.text(result[0])


st.text('')
st.text('')
