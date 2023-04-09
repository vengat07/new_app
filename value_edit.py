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

df= pd.read_csv('diabetes_1.csv')

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
#name = st.sidebar.text_input('Enter the name')
st.write(df.head())
st.subheader('Training Data Stats')
st.write(df.describe())


# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
 
data_names = pd.read_csv('data_names.csv')

# FUNCTION
import streamlit as st
import pandas as pd

def user_report(df):
    name = st.text_input('Name')
    pregnancies = st.slider('Pregnancies', 0,17, )
    glucose = st.slider('Glucose', 0,200, )
    bp = st.slider('Blood Pressure', 0,122, )
    skinthickness = st.slider('Skin Thickness', 0,100,)
    insulin = st.slider('Insulin', 0,300,)
    bmi = st.slider('BMI', 0,67, )
    dpf = st.slider('Diabetes Pedigree Function', 0.0,2.4,  )
    age = st.slider('Age', 0,100, )

    submit_button = st.sidebar.button('Submit')
    if not submit_button:
        return df
    
    user_report_data = {
        'Name': name,
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    data_names = df.append(user_report_data, ignore_index=True)
    
    return data_names



# PATIENT DATA
user_data_with_name = user_report(data_names)
#user_data_with_name = user_report()
#user_data_with_name.insert(0,"name",name,True)
st.subheader('Patient Data')
st.write(user_data_with_name)

user_data = user_data_with_name.drop('Name',axis=1)



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
