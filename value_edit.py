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
 
#df= pd.read_csv('data_names.csv')

# FUNCTION
import streamlit as st
import pandas as pd

def user_report():
   name = st.sidebar.text_input('Enter the name')
   pregnancies = int(st.sidebar.slider('Pregnancies', 0,17,))
   glucose = int(st.sidebar.slider('Glucose', 0,200,))
   bp = int(st.sidebar.slider('Blood Pressure', 0,122, ))
   skinthickness = 29
   insulin = int(st.sidebar.slider('Insulin', 0,150,  ))
   bmi = int(st.sidebar.slider('BMI', 0,67,))
   dpf = 0.47
   age = int(st.sidebar.slider('Age', 21,88,))

   if st.sidebar.button('Submit'):
      user_report_data = {
         'name':name,
         'pregnancies':pregnancies,
         'glucose':glucose,
         'bp':bp,
         'skinthickness':skinthickness,
         'insulin':insulin,
         'bmi':bmi,
         'dpf':dpf,
         'age':age
      }
      report_data = pd.DataFrame(user_report_data, index=[0])
      return report_data
   else:
      return None




# PATIENT DATA
user_data_with_name = user_report()
#user_data_with_name = user_report()
#user_data_with_name.insert(0,"name",name,True)
st.subheader('Patient Data')
st.write(user_data_with_name)

user_data = user_data_with_name.drop('name',axis=1)

#st.write(user_data)

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

st.subheader('Type Result')
output=''
if glucose > 125 and insulin < 125 and bmi < 25 and age > 40:
  output = 'You are Type I Diabetes'
else:
  output = 'You are Type II Diabetes'
st.title(output)

#if glucose > 125 and insulin < 70 and bmi < 25 and age > 40:
