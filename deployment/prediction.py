import pandas as pd
import streamlit as st
import numpy as np
import pickle
import datetime
from tensorflow.keras.models import load_model

# Load model
with open('full_pipeline.pkl', 'rb') as file_1:
  full_pipeline = pickle.load(file_1)

# Load ANN Model
func_model = load_model('model.h5')
    
def run():
    # Membuat Form
    with st.form(key = 'churn_customer'):
        age = st.slider('Age', min_value= 0, max_value=100, value=35)
        last_login = st.slider('Days Last Login', min_value=0, max_value=50, value=16)
        time_spent = st.slider('Average of Time Spent', min_value=0, max_value=4000, value=3000)
        transaction = st.slider('Average of Transaction', min_value=0, max_value=100000, value=50000)
        freq_login = st.slider('Average Frequency Login Days', min_value=0, max_value=100, value=22)
        points_wallet = st.slider('Points in Wallet', min_value=0, max_value=3000, value=1000)
        gender = st.radio('Gender (M : Male, F: Female)', ('M', 'F'))
        region = st.radio('Region Category', ('City', 'Village', 'Town'))
        membership = st.radio('Membership Category', ('No Membership', 'Basic Membership', 'Silver Membership', 'Premium Membership', 'Gold Membership', 'Platinum Membership'))
        joined = st.radio('Joined Through Referral ?', ('Yes', 'No'))
        preferred = st.radio('Preferred Offer Types', ('Without Offers', 'Credit/Debit Card Offers', 'Gift Vouchers/Coupons'))
        medium = st.radio('Medium of Operation', ('Desktop', 'Smartphone', 'Both'))
        internet = st.radio('Internet Option', ('Mobile_Data', 'Fiber_optic', 'Wi-Fi'))
        discount = st.radio('Used Special Discount ?', ('Yes', 'No'))
        offer_app = st.radio('Offer Application Preference', ('Yes', 'No'))
        past_complaint = st.radio('Past Complaint', ('Yes', 'No'))
        complaint_status = st.radio('Complaint Status', ('No Information Available', 'Not Applicable', 'Unsolved', 'Solved', 'Solved in Follow-up'))
        feedback = st.radio('Feedback', ('Poor Website', 'Poor Customer Service', 'Too many ads', 'Poor Product Quality', 'No reason specified', 'Product always in Stock', 'Reasonable Price', 'Quality Customer Care', 'User Friendly Website'))
        joining_date = st.date_input('Joining Date', datetime.date(2019, 7, 6))
        last_visit = st.date_input('Last Visit', datetime.date(2019, 7, 6))
        submitted = st.form_submit_button('Predict')

    # Data Inference
    data_inf = {
        'age' : age,
        'days_since_last_login' : last_login,
        'avg_time_spent' : time_spent,
        'avg_transaction_value' : transaction,
        'avg_frequency_login_days' : freq_login,
        'points_in_wallet' : points_wallet,
        'joining_date' : joining_date,
        'last_visit_time' : last_visit,
        'gender' : gender,
        'region_category' : region,
        'membership_category' : membership,
        'joined_through_referral' : joined,
        'preferred_offer_types' : preferred,
        'medium_of_operation' : medium,
        'internet_option' : internet,
        'used_special_discount' : discount,
        'offer_application_preference' : offer_app,
        'past_complaint' : past_complaint,
        'complaint_status' : complaint_status,
        'feedback' : feedback
    }
    
    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)
    
    if submitted:
        # Feature Engineering & Modelling
        data_final = full_pipeline.transform(data_inf)
        
        y_pred_inf = func_model.predict(data_final)
        y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)
        y_pred_inf
        
        if y_pred_inf == 0:
            st.write('Not Churn')
        else:
            st.write('Churn')

if __name__ == '__main__':
    run()