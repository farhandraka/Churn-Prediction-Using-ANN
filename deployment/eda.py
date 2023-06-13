import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image


st.set_page_config(
    page_title = 'Churn Prediction - EDA',
    layout = 'wide')

def run():
    # Plot title
    st.title('Churn Customer Prediction')
    
    # Plot sub-header
    st.subheader('Exploratory Data Analysis Churn Dataset')
    
    # Add Image
    image = Image.open('churn.jpg')
    st.image(image)
    
    # Add description
    st.write('# Introduction')
    st.write('Name : Muhammad Farhan Darmawan')
    st.write('Batch : RMT 019')
    st.markdown('---')
    
    '''
    Dataset Description :

    | Column | Description |
    | --- | --- |
    | user_id	| ID of a customer |
    | age	| Age of a customer |
    | gender	| Gender of a customer |
    | region_category	| Region that a customer belongs to |
    | membership_category	| Category of the membership that a customer is using |
    | joining_date | Date when a customer became a member |
    | joined_through_referral	| Whether a customer joined using any referral code or ID |
    | preferred_offer_types | Type of offer that a customer prefers |
    | medium_of_operation	| Medium of operation that a customer uses for transactions |
    | internet_option	| Type of internet service a customer uses |
    | last_visit_time	| The last time a customer visited the website |
    | days_since_last_login	| Number of days since a customer last logged into the website |
    | avg_time_spent	| Average time spent by a customer on the website |
    | avg_transaction_value	| Average transaction value of a customer |
    | avg_frequency_login_days	| Number of times a customer has logged in to the website |
    | points_in_wallet	| Points awarded to a customer on each transaction |
    | used_special_discount	| Whether a customer uses special discounts offered |
    | offer_application_preference	| Whether a customer prefers offers |
    | past_complaint	| Whether a customer has raised any complaints |
    | complaint_status	| Whether the complaints raised by a customer was resolved |
    | feedback	| Feedback provided by a customer |
    | churn_risk_score	| Churn score (0 : Not churn, 1 : Churn) |
    '''
    st.write('# Dataset of Churn')
    #show dataframe
    data = pd.read_csv('churn.csv')
    st.dataframe(data)
    
    # data cleaning
    # drop column
    data = data.drop(['user_id'], axis=1)
    # change type to datetime
    data['joining_date'] = pd.to_datetime(data['joining_date'])
    data['last_visit_time'] = pd.to_datetime(data['last_visit_time'])
    # extract year
    data['joining_date_year'] = data['joining_date'].dt.year
    data['last_visit_year'] = data['last_visit_time'].dt.year
    # splitting data
    num_cols = data.select_dtypes(include=np.number).columns.tolist()
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    st.markdown('---')
    
    st.write('# Histogram Graph By Churn')
    # Show grafik age
    x1 = list(data[data['churn_risk_score'] == 1]['age'])
    x2 = list(data[data['churn_risk_score'] == 0]['age'])

    fig = plt.figure(figsize=(12,4))
    sns.set_context('notebook', font_scale=1.2)
    #sns.set_color_codes("pastel")
    plt.hist([x1, x2], bins = 40, density=False, color=['steelblue', 'lightblue'])
    plt.xlim([10,65])
    plt.legend(['Yes', 'No'], title = 'churn_risk_score', loc='upper right', facecolor='white')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age Histogram By Churn', size=15)
    plt.box(False)
    plt.savefig('ImageName', format='png', dpi=200, transparent=True)
    st.pyplot(fig)

    # Show grafik days last login
    x1 = list(data[data['churn_risk_score'] == 1]['days_since_last_login'])
    x2 = list(data[data['churn_risk_score'] == 0]['days_since_last_login'])

    fig = plt.figure(figsize=(12,4))
    sns.set_context('notebook', font_scale=1.2)
    #sns.set_color_codes("pastel")
    plt.hist([x1, x2], bins = 40, density=False, color=['steelblue', 'lightblue'])
    plt.xlim([0,100])
    plt.legend(['Yes', 'No'], title = 'churn_risk_score', loc='upper right', facecolor='white')
    plt.xlabel('Days Last Login')
    plt.ylabel('Frequency')
    plt.title('Days Last Login Histogram By Churn', size=15)
    plt.box(False)
    plt.savefig('ImageName', format='png', dpi=200, transparent=True)
    st.pyplot(fig)
    
    # Show grafik average time spent
    x1 = list(data[data['churn_risk_score'] == 1]['avg_time_spent'])
    x2 = list(data[data['churn_risk_score'] == 0]['avg_time_spent'])

    fig = plt.figure(figsize=(12,4))
    sns.set_context('notebook', font_scale=1.2)
    #sns.set_color_codes("pastel")
    plt.hist([x1, x2], bins = 40, density=False, color=['steelblue', 'lightblue'])
    plt.xlim([0,4000])
    plt.legend(['Yes', 'No'], title = 'churn_risk_score', loc='upper right', facecolor='white')
    plt.xlabel('Average Time Spent')
    plt.ylabel('Frequency')
    plt.title('Average Time Spent Histogram By Churn', size=15)
    plt.box(False)
    plt.savefig('ImageName', format='png', dpi=200, transparent=True)
    st.pyplot(fig)
    
    # Show grafik average transaction value
    x1 = list(data[data['churn_risk_score'] == 1]['avg_transaction_value'])
    x2 = list(data[data['churn_risk_score'] == 0]['avg_transaction_value'])

    fig = plt.figure(figsize=(12,4))
    sns.set_context('notebook', font_scale=1.2)
    #sns.set_color_codes("pastel")
    plt.hist([x1, x2], bins = 40, density=False, color=['steelblue', 'lightblue'])
    plt.xlim([0,100000])
    plt.legend(['Yes', 'No'], title = 'churn_risk_score', loc='upper right', facecolor='white')
    plt.xlabel('Average Transaction Value')
    plt.ylabel('Frequency')
    plt.title('Average Transaction Value Histogram By Churn', size=15)
    plt.box(False)
    plt.savefig('ImageName', format='png', dpi=200, transparent=True)
    st.pyplot(fig)
    
    # Show grafik average frequency login days
    x1 = list(data[data['churn_risk_score'] == 1]['avg_frequency_login_days'])
    x2 = list(data[data['churn_risk_score'] == 0]['avg_frequency_login_days'])

    fig = plt.figure(figsize=(12,4))
    sns.set_context('notebook', font_scale=1.2)
    #sns.set_color_codes("pastel")
    plt.hist([x1, x2], bins = 40, density=False, color=['steelblue', 'lightblue'])
    plt.xlim([0,75])
    plt.legend(['Yes', 'No'], title = 'churn_risk_score', loc='upper right', facecolor='white')
    plt.xlabel('Average Frequency Login Days')
    plt.ylabel('Frequency')
    plt.title('Average Frequency Login Days Histogram By Churn', size=15)
    plt.box(False)
    plt.savefig('ImageName', format='png', dpi=200, transparent=True)
    st.pyplot(fig)
    
    # Show grafik points in wallet
    x1 = list(data[data['churn_risk_score'] == 1]['points_in_wallet'])
    x2 = list(data[data['churn_risk_score'] == 0]['points_in_wallet'])

    fig = plt.figure(figsize=(12,4))
    sns.set_context('notebook', font_scale=1.2)
    #sns.set_color_codes("pastel")
    plt.hist([x1, x2], bins = 40, density=False, color=['steelblue', 'lightblue'])
    plt.xlim([0,2500])
    plt.legend(['Yes', 'No'], title = 'churn_risk_score', loc='upper right', facecolor='white')
    plt.xlabel('Points in Wallet')
    plt.ylabel('Frequency')
    plt.title('Points in Wallet Histogram By Churn', size=15)
    plt.box(False)
    plt.savefig('ImageName', format='png', dpi=200, transparent=True)
    st.pyplot(fig)
    
    # Show grafik joining date
    x1 = list(data[data['churn_risk_score'] == 1]['joining_date_year'])
    x2 = list(data[data['churn_risk_score'] == 0]['joining_date_year'])

    fig = plt.figure(figsize=(12,4))
    sns.set_context('notebook', font_scale=1.2)
    #sns.set_color_codes("pastel")
    plt.hist([x1, x2], bins = 20, density=False, color=['steelblue', 'lightblue'])
    plt.xlim([2015,2017])
    plt.legend(['Yes', 'No'], title = 'churn_risk_score', loc='upper right', facecolor='white')
    plt.xlabel('Joining Date')
    plt.ylabel('Frequency')
    plt.title('Joining Date Histogram By Churn', size=15)
    plt.box(False)
    plt.xticks([2015, 2016, 2017, 2018])
    plt.savefig('ImageName', format='png', dpi=200, transparent=True)
    st.pyplot(fig)
    
    # show grafik last visit
    x1 = list(data[data['churn_risk_score'] == 1]['last_visit_year'])
    x2 = list(data[data['churn_risk_score'] == 0]['last_visit_year'])

    fig = plt.figure(figsize=(12,4))
    sns.set_context('notebook', font_scale=1.2)
    #sns.set_color_codes("pastel")
    plt.hist([x1, x2], bins = 20, density=False, color=['steelblue', 'lightblue'])
    plt.xlim([2022,2024])
    plt.legend(['Yes', 'No'], title = 'churn_risk_score', loc='upper right', facecolor='white')
    plt.xlabel('Last Visit')
    plt.ylabel('Frequency')
    plt.title('Last Visit Histogram By Churn', size=15)
    plt.box(False)
    plt.xticks([2022, 2023, 2024])
    plt.savefig('ImageName', format='png', dpi=200, transparent=True)
    st.pyplot(fig)
    st.markdown('---')
    
    st.write('# Countplot Graph By Churn')
    # Daftar nama kolom
    cols = ['gender', 'region_category', 'membership_category',
        'joined_through_referral', 'preferred_offer_types',
        'medium_of_operation', 'internet_option',
        'used_special_discount', 'offer_application_preference',
        'past_complaint', 'complaint_status', 'feedback', 'churn_risk_score']

    # Membuat subplot
    f, axes = plt.subplots(7, 2, figsize=(40, 50), facecolor='white')
    f.suptitle('Frekuensi Data By Churn')

    # Membuat looping
    for i, column in enumerate(cols):
        row = i // 2  # Nomor baris subplot
        col = i % 2   # Nomor kolom subplot
        
        # Menampilkan countplot
        ax = sns.countplot(x=column, hue='churn_risk_score', data=data[cols], palette='Blues', ax=axes[row, col])
        ax.set_title(column)
        ax.legend(title='churn_risk_score', loc='upper right')

    # Menampilkan plot
    plt.tight_layout()
    plt.show()
    st.pyplot(f)
    st.markdown('---')
    
    st.write('# Distribution of Dataset')
    # Membuat subplot dengan ukuran 6 x 4
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # Mendapatkan daftar kolom dalam DataFrame
    columns = num_cols

    # Melakukan looping untuk membuat histogram pada setiap kolom
    for i, ax in enumerate(axes.flatten()):
        if i < len(columns):
            # Membuat histogram
            sns.histplot(data=data, x=columns[i], kde=True, color='steelblue', alpha=0.7, ax=ax)
            ax.set_xlabel(columns[i])
            ax.set_ylabel('frequency')

    # Menyusun tata letak subplot
    plt.tight_layout()

    # Menampilkan plot
    plt.show()
    st.pyplot(fig)
    st.markdown('---')
    
    st.write('# Scatterplot of Dataset')
    fig = sns.pairplot(data = data[num_cols], hue = 'churn_risk_score', palette='Blues')
    st.pyplot(fig)
    st.markdown('---')
    
    st.write(
        '''
        Statement : 

        - From 37010 customers The age of the customer distribute from 10 - 64 years old with a mean of 37 years old. The customer joining date is in 2015 - 2017 and the last visit is in 2023, the customer will log in 14 times a day and will spend 279 hours on the website. The longest last days of customer login is 26 days.

        - From the average transaction value, of more than 50000 transactions the risk of the customer will churn is no, opposite that the risk of the customer will churn is the possibility to churn. The customer with < 750 points in the wallet risks churn than  points in the wallet > 750

        - The customer churn is higher than the customer does not churn, both genders are equal, by region of city and town has a high possibility to churn, by membership category `no` and `basic` membership are high churn, and by the feedback, the customer who gives bad feedback is the high churn

        - There is no significant correlation of each columns and the distribution of data are skewed
        '''
    )
    
    
if __name__ == '__main__':
    run()