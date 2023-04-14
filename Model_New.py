# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:18:51 2023

@author: rohan
"""

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.feature_selection import f_regression, mutual_info_regression,chi2, SelectKBest

plt.rcParams['figure.figsize'] = [10,8]

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("Customers.csv")

data.rename(columns={'Annual Income ($)': 'Annual_Income', 
                     'Spending Score (1-100)':'Spending_Score', 
                    'Work Experience':'Work_Experience', 
                    'Family Size':'Family_Size'}, 
            inplace=True)

data.head()

data.info()

df = data.copy(deep=True)
df.drop("CustomerID", axis=1, inplace=True)

df.dropna(inplace=True)
df.drop(df.loc[df["Annual_Income"]==0].index, inplace=True)
df.drop(df.loc[df["Age"]<18].index, inplace=True)


encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])
print(encoder.classes_)

print(df['Gender'][0:6])

df["Profession"] = encoder.fit_transform(df["Profession"])

x = df.drop("Annual_Income", axis=1)
y = df["Annual_Income"]

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.20,
                                                    random_state=42)

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)


# Fitting teh model and predicting the values.
model = LinearRegression()
regressor = model.fit(x_train, y_train)
y_pred = regressor.predict(x_test)


# Dumping the pickle file into our disk memory.
pickle_out = open("regressor.pkl","wb")
pickle.dump(regressor, pickle_out)
pickle_out.close()


regressor.predict([[1,1,0,0,0,1]])


