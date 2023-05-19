# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

page_title = "SUGAR SPOT"

st.set_page_config(page_title = page_title)

df= pd.read_csv('diabetes_1.csv')

# HEADINGS
st.title('SUGAR SPOT')
st.sidebar.header('Patient Data')
st.write(df.head())
st.subheader('Training Data Stats')
st.write(df.describe())


# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
 


# FUNCTION

def user_report():
   name = st.sidebar.text_input('Enter the name')
   pregnancies = int(st.sidebar.slider('Pregnancies', 0,17,))
   glucose = int(st.sidebar.slider('Glucose', 0,200,))
   bp = int(st.sidebar.slider('Blood Pressure', 0,122, ))
   skinthickness = 29
   insulin = int(st.sidebar.slider('Insulin', 0,150,  ))
   bmi = int(st.sidebar.slider('BMI', 0,67,))
   dpf = 0.47
   age = int(st.sidebar.slider('Age', 0,100,))

   if st.sidebar.button('Predict'):
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
      
      if glucose > 125 and insulin < 70 and bmi < 25 and age > 40:
         outresult = 'You are Type I Diabetes'
        
      else:
         outresult = 'You are Type II Diabetes'
      return report_data,outresult
   else:
      return None,None


# PATIENT DATA
user_data_with_name, outresult = user_report()
st.subheader('Patient Data')
st.write(user_data_with_name)

user_data = user_data_with_name.drop('name',axis=1)

# MODEL
rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)


# VISUALISATIONS
st.title('Visualised Patient Report')

# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'


# Age vs Pregnancies
st.header('Pregnancy count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'age', y = 'pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['age'], y = user_data['pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)



# Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['age'], y = user_data['glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)



# Age vs Bp
st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['bp'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)



# Age vs Insulin
st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['age'], y = user_data['insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)


# Age vs BMI
st.header('BMI Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = user_data['age'], y = user_data['bmi'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)



# OUTPUT
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You are not Diabetic'
  st.title(output)
else:
  output = 'You are Diabetic'
  if outresult:
     st.subheader('Type Result')
     st.title(output)
     st.title(outresult)
  


