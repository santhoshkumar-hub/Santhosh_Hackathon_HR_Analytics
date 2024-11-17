# import necessary libraries
import streamlit as st
import pandas as pd
import joblib

st.title("Promotion Prediction")

# read the dataset to fill list values
df = pd.read_csv('E:/Santhosh Repo/Hackathon/train_LZdllcl.csv')

# create input fields 
department = st.selectbox("department", pd.unique(df['department']))
region = st.selectbox("region", pd.unique(df['region']))
education = st.selectbox("education", pd.unique(df['education']))
gender = st.selectbox("gender", pd.unique(df['gender']))
recruitment_channel = st.selectbox("recruitment_channel", pd.unique(df['recruitment_channel']))
no_of_trainings = st.number_input("no_of_trainings")
age = st.number_input("age")
previous_year_rating = st.selectbox("previous_year_rating", pd.unique(df['previous_year_rating']))
length_of_service = st.number_input("length_of_service")
KPIs_met = st.number_input("KPIs_met >80%")
awards_won = st.number_input("awards_won?")
avg_training_score = st.number_input("avg_training_score")

 
# convert the input values to dict
inputs = {
  "department": department,
  "region": region,
  "education": education,
  "gender": gender,
  "recruitment_channel": recruitment_channel,
  "no_of_trainings": no_of_trainings,
  "age": age,
  "previous_year_rating": previous_year_rating,
  "length_of_service": length_of_service,
  "KPIs_met >80%": KPIs_met,
  "awards_won?": awards_won,
  "avg_training_score": avg_training_score
}

 
# on click
if st.button("Predict"):
    # load the pickle model 
    model = joblib.load('E:/Santhosh Repo/Hackathon/APIDeployment/jobchg_pipeline_model.pkl')

    X_input = pd.DataFrame(inputs,index=[0])
    # predict the target using the loaded model
    prediction = model.predict(X_input)
    # display the result
    st.write(prediction)

