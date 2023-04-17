# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:27:50 2023

@author: rohan
"""

import streamlit as st
from sklearn import sklearn
import pickle

pickle_file_in = open("regressor.pkl","rb")
model = pickle.load(pickle_file_in)

def welcome():
    return "Hello How are you?"

def predict_shop_customer(Gender, Age, Spending_Score, Profession, 
                          Work_Experience, 
                          Family_Size):
    prediction = model.predict([[Gender,Age,Spending_Score,Profession,
                                     Work_Experience, Family_Size]])
    print(prediction)
    return prediction

def main():
    
    st.title("Annual Income Prediction")
    html_temp = """
    <div style="background-color:black;padding:10px">
    <h2 style="color:white;text-align:center;">Shop Customer ML WebApp </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Gender = st.radio("Gender:",("Male","Female"))#Male = 0,Female=1
    Age = st.slider("Input Age:",18,80)
    Spending_Score = st.slider("Input Spending Score:",0,100)
    #heathcare=0,engineer=1,lawyer=2,entertainment=3,artist=4,executive=5,
    #docter=6,homemaker=7,marketing=8
    Profession = st.radio("Profession:",("Healthcare","Engineer","Lawyer",
                                         "Entertainment","Artist","Executive",
                                         "Doctor","Homemaker","Marketing"))
    Work_Experience = st.slider("Work_Experience:",0,17)
    Family_Size = st.slider("Family Size:",1,9)
    result = ""
    if Gender == "Male":
        Gender = 1
    else:
        Gender = 0
        
    if Profession == "Healthcare":
        Profession = 0
    elif Profession == "Engineer":
        Profession = 1
    elif Profession == "Lawyer":
        Profession = 2
    elif Profession == "Entertainment":
        Profession = 3
    elif Profession == "Artist":
        Profession = 4
    elif Profession == "Executive":
        Profession = 5
    elif Profession == "Doctor":
        Profession = 6
    elif Profession == "Homemaker":
        Profession = 7
    else:
        Profession = 8
    if st.button("Predict"):
        result = predict_shop_customer(Gender, Age, Spending_Score, Profession, 
                                  Work_Experience, 
                                  Family_Size)
    st.success("Annual Income is :- {}".format(result))
    if st.button("About"):
        st.text("Powered by Streamlit")
        st.text("Developed by Prateek Singh")
        
if __name__=="__main__":
    main()
