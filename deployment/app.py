import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Choose Page :', ('Exploratory Data Analysis', 'Prediction Churn Customer'))

if navigation == 'Exploratory Data Analysis':
    eda.run()
else:
    prediction.run()