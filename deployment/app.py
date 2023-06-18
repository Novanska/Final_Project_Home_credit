import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title = 'HOMECREDIT DEBTORS PREDICTION' ,
    initial_sidebar_state= 'expanded',
)
st.title('PAY CREDIT PREDICTION')
st.subheader('EXPLORATORY DATA ANALYSIS(EDA)')
st.markdown('---')

image = Image.open('Home-Credit-logo.jpg')
st.image(image)

st.write('Dataset retrive from : <https://www.kaggle.com/competitions/home-credit-default-risk/data>')
st.write('In this EDA we just take application_train data from dataset.')
data = pd.read_csv('application_train.csv')

st.write('## Distribution of Target')
target_counts = data['TARGET'].value_counts()
fig, ax = plt.subplots()
labels = ['Pay', 'Not Pay']
pie = ax.pie(target_counts, labels=labels, autopct='%1.1f%%', startangle=90)
ax.set_title('Loan Repayment Status')
st.pyplot(fig)
fig = plt.figure(figsize=(10, 6))
data['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram')
plt.xlabel('Days Employment')
st.pyplot(fig)
fig = plt.figure(figsize=(10,6))
ax = sns.countplot(x='NAME_INCOME_TYPE', hue='TARGET', data=data)
ax.set_xlabel('Income Type')
ax.set_ylabel('Count')
ax.set_title('Distribution of Loan Applicants by Income Type and Payment Status')
plt.xticks(rotation=45)
plt.legend(title='Payment Status', labels=['Will pay', 'Will not pay'])
st.pyplot(fig)
fig = plt.figure(figsize=(10, 6))
ax = sns.countplot(x='CNT_CHILDREN', hue='TARGET', data=data)
ax.set_xlabel('Number of children')
ax.set_ylabel('Count')
ax.set_title('Number of Loan Approval and Rejection Based on Number of Children')
plt.legend(title='Repayment Status', labels=['Will pay', 'Will not pay'])
st.pyplot(fig)
fig = plt.figure(figsize=(10,6))
ax = sns.countplot(x='FLAG_OWN_CAR', hue='TARGET', data=data)
ax.set_xlabel('Own Car')
ax.set_ylabel('Count')
ax.set_title('Number of Loan Approval and Rejection Based on Car Ownership')
plt.legend(title='Repayment Status', labels=['Will pay', 'Will not pay'])
st.pyplot(fig)
fig = plt.figure(figsize=(10,6))
ax = sns.countplot(x='NAME_HOUSING_TYPE', hue='TARGET', data=data,order=data['NAME_HOUSING_TYPE'].value_counts().index)
ax.set_xlabel('House Type')
ax.set_ylabel('Count')
ax.set_title('Distribution of Loan Approval and Rejection by House Type and Payment Status')
plt.xticks(rotation=45)
plt.legend(title='Repayment Status', labels=['Will pay', 'Will not pay'])
st.pyplot(fig)
fig = plt.figure(figsize=(10, 6))
ax = sns.countplot(x='REGION_RATING_CLIENT', hue='TARGET', data=data,order=data['REGION_RATING_CLIENT'].value_counts().index)
ax.set_xlabel('Region Rating')
ax.set_ylabel('Count')
ax.set_title('Distribusi Pemohon Pinjaman Berdasarkan Peringkat Wilayah dan Status Pelunasan')
plt.legend(title='Status Pembayaran', labels=['Akan Melunasi', 'Tidak Akan Melunasi'])
st.pyplot(fig)
st.write('## Conclussion')
st.write('''Based on **EDA**:

1. Based on train data there are still around 8.1% of debtors who have not paid their installments
2. Most approved loans are for people who are employed, married, have 0 - 3 children, do not have a car and live in a house or apartment.
3. The region with a rating value of 2 is the region with the highest number of borrowers
''')