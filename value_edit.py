# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

page_title = "Daibetes_Prediction and checkup"

st.set_page_config(page_title = page_title)

data_set= pd.read_csv('diabetes_1.csv')

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
#name = st.sidebar.text_input('Enter the name')
st.write(data_set.head())
st.subheader('Training Data Stats')
st.write(data_set.describe())


# X AND Y DATA
x = data_set.drop(['Outcome'], axis = 1)
y = data_set.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
 
df= pd.read_csv('data_names.csv')

# FUNCTION
import streamlit as st
import pandas as pd

def user_report(df):
     name = st.sidebar.text_input('Name')
     pregnancies = st.sidebar.slider('Pregnancies', 0, 17)
     glucose = st.sidebar.slider('Glucose', 0, 200)
     bp = st.sidebar.slider('Blood Pressure', 0, 122)
     skinthickness = st.sidebar.slider('Skin Thickness', 0, 100)
     insulin = st.sidebar.slider('Insulin', 0, 300)
     bmi = st.sidebar.slider('BMI', 0, 67)
     dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4)
     age = st.sidebar.slider('Age', 0, 100)

     submit_button = st.sidebar.button('Submit')

     user_report_data = {
        'name': name,
        'pregnancies': pregnancies,
        'glucose': glucose,
        'bp': bp,
        'skinthickness': skinthickness,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age
    }

     if submit_button:
        df = df.append(user_report_data, ignore_index=True)
    
     return df



# PATIENT DATA
user_data_with_name = user_report(df)
#user_data_with_name = user_report()
#user_data_with_name.insert(0,"name",name,True)
st.subheader('Patient Data')
st.write(user_data_with_name)

user_data = user_data_with_name.drop('name',axis=1)



# MODEL
rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)




# OUTPUT
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'
st.title(output)
