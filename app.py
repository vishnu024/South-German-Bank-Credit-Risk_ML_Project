import streamlit as st
import pandas as pd
import os,sys
import numpy as np
import pickle


with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

with open('transformer.pkl', 'rb') as t:
        transformer = pickle.load(t)       

st.title('Bank Credit Risk Prediction')
st.markdown('Created a Model to  predict whether the person, described by the attributes of the dataset \
                is a good (1) or a bad (0) credit risk.')

st.header("User Inputs:")
col1, col2 = st.columns(2)

with col1:
                status_options = {
    "no checking account": 1,
    "... < 0 DM": 2,
    "0 <= ... < 200 DM": 3,
    "... >= 200 DM / salary assignments for at least 1 year": 4
                                 }

                selected_status = st.selectbox("Select Status:", list(status_options.keys()), index=0)
                status = status_options[selected_status]

                duration = st.number_input("Enter duration in months:",step=60)

                credit_history_options = {
    "critical account/ other credits existing (not at this bank)": 0,
    "delay in paying off in the past": 1,
    "existing credits paid back duly till now": 2,
    "all credits at this bank paid back duly": 3,
    "no credits taken/ all credits paid back duly":4 
                                 }

                selected_credit_history = st.selectbox("Select Credit history:", list(credit_history_options.keys()), index=0)
                credit_history = credit_history_options[selected_credit_history]

                purpose_options = {
    "OTHERS": 0,
    "CAR (NEW)": 1,
    "CAR (USED)": 2,
    "FURNITURE/EQUIPMENT": 3,
    "RADIO/TELEVISION": 4,
    "DOMESTIC APPLIANCES": 5,
    "REPAIRS": 6,
    "EDUCATION ": 7,
    "VACATION": 8,
    "RETRAINING": 9,
    "BUSINESS": 10

                                 }

                selected_purpose = st.selectbox("Select Purpose:", list(purpose_options.keys()), index=0)
                purpose = purpose_options[selected_purpose]


                amount = st.number_input("Enter Credit amount:",step=10000)

                savings_options = {
    "UNKNOWN/NO SAVINGS ACCOUNT": 1,
    "... <  100 DM": 2,
    "100 <= ... <  500 DM": 3,
    "500 <= ... < 1000 DM": 4,
    "... >= 1000 DM": 5
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
                selected_employment = st.selectbox("Select Present employment since:", list(employment_options.keys()), index=0)
                employment_duration = employment_options[selected_employment]

                installment_options = {
    ">= 35": 1,
    "25 <= ...": 2,
    "20 <= ... ": 3,
    "< 20": 4
                                      }
                selected_installment = st.selectbox("Select Installment rate in percentage of disposable income:", list(installment_options.keys()), index=0)
                installment_rate = installment_options[selected_installment]

with col2:
                property_options = {
    "UNKNOWN / NO PROPERTY": 1,
    "CAR OR OTHER": 2,
    "BUILDING SOC. SAVINGS AGR./LIFE INSURANCE": 3,
    "REAL ESTATE": 4
                                  }
                selected_property = st.selectbox("Select Property:", list(property_options.keys()), index=0)
                propertyl = property_options[selected_property]
                

                age = st.number_input("Enter Age in years:",step=1)

                credits_options = {
    "1": 1,
    "2-3": 2,
    "4-5": 3,
    ">=6": 4
                                  }
                selected_credits = st.selectbox("Select Number of existing credits at this bank:", list(credits_options.keys()), index=0)
                number_credits = credits_options[selected_credits]

                

                job_options = {
    "unemployed/ unskilled - non-resident": 1,
    "unskilled - resident": 2,
    "skilled employee / official": 3,
    "management/ self-employed/highly qualified employee/ officer": 4
                                  }
                selected_job = st.selectbox("Select Job:", list(job_options.keys()), index=0)
                job = job_options[selected_job]


                people_liable = st.number_input("Enter Number of people being liable to provide maintenance for:",step=1)

                telephone_options = {
    "No": 1,
    "Yes, registered under the customers name": 2,
    
                                  }
                selected_telephone = st.selectbox("Select Telephone:", list(telephone_options.keys()), index=0)
                telephone = telephone_options[selected_telephone]

                personal_status_sex_options = {
    "Male : divorced/separated": 1,
    "Female : divorced/separated/married": 2,
    "Male : single": 3,
    "Male : married/widowed": 4,
    "Female : single": 5
                                  }
                selected_personal_status = st.selectbox("Select Gender and Personal status:", list(personal_status_sex_options.keys()), index=0)
                personal_status_sex = personal_status_sex_options[selected_personal_status]

st.text('')
if st.button("Predict the Credibility"):

    data = np.array([[status, duration, credit_history, purpose, amount, savings, employment_duration, installment_rate, personal_status_sex, propertyl, age, number_credits, job, people_liable, telephone]])
    datat = pd.DataFrame(data, columns=['status', 'duration', 'credit_history', 'purpose', 'amount', 'savings', 'employment_duration', 'installment_rate', 'personal_status_sex', 'propertyl', 'age', 'number_credits', 'job', 'people_liable', 'telephone'])
    datat = datat.rename(columns={'propertyl': 'property'})
    transformed = transformer.transform(datat)
    result= model.predict(np.array(transformed))

    if result == 0.0:
        st.subheader(':red[Bad Credit Risk] :sweat:')
    else:
        st.subheader(':green[Good Credit Risk] :blush:')
    
        

st.text("Note: The Deutsche mark (DM) was Germany's legal currency from 1948 to 2002.")
st.text("In 2002, Germany replaced the Deutsche mark with the Euro.")
st.markdown(
    '`Created by` [Vishnu Kumar](https://www.linkedin.com/in/vishnukumar007) | \
         `Code:` [GitHub](https://github.com/vishnu024/South-German-Bank-Credit-Risk_ML_Project)')
