import streamlit as st
from credit.predictor import ModelResolver
from credit.entity import config_entity,artifact_entity
from credit.exception import CreditException
from credit.logger import logging
from credit.utils import load_object
from sklearn.metrics import f1_score
import pandas  as pd
import sys,os
from credit.config import TARGET_COLUMN

model = load_object(file_path=model_path)
st.title('Classifying Credit')
st.markdown('model to classify Credit into \
     (setosa, versicolor, virginica) based on their sepal/petal \
    and length/width.')

st.header("Credit data")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    status = st.slider('status', 1.0, 8.0, 0.5)
    duration = st.slider('duration', 2.0, 4.4, 0.5)
    credit_history = st.slider('credit_history', 2.0, 4.4, 0.5)
    purpose = st.slider('purpose', 2.0, 4.4, 0.5)
    amount = st.slider('amount', 2.0, 4.4, 0.5)
    savings = st.slider('savings', 2.0, 4.4, 0.5)
    employment_duration = st.slider('employment_duration', 2.0, 4.4, 0.5)
    installment_rate = st.slider('installment_rate', 2.0, 4.4, 0.5)
    personal_status_sex = st.slider('personal_status_sex', 2.0, 4.4, 0.5)
    propertyl = st.slider('property', 2.0, 4.4, 0.5)
    age = st.slider('age', 2.0, 4.4, 0.5)
    number_credits = st.slider('number_credits', 2.0, 4.4, 0.5)
    job = st.slider('job', 2.0, 4.4, 0.5)

with col2:
    st.text("Pepal characteristics")
    people_liable = st.slider('people_liable', 1.0, 7.0, 0.5)
    telephone = st.slider('telephone', 0.1, 2.5, 0.5)

st.text('')
if st.button("Predict type of Iris"):
    result = model.predict(
        np.array([[status, duration, credit_history, purpose, amount, savings, employment_duration, installment_rate, personal_status_sex, propertyl, age, number_credits, job, people_liable, telephone, credit_risk]]))
    st.text(result[0])


st.text('')
st.text('')
st.markdown(
    '`Create by` [santiviquez](https://twitter.com/santiviquez) | \
         `Code:` [GitHub](https://github.com/santiviquez/iris-streamlit)')