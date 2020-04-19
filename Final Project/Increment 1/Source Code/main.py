from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_submission = pd.read_csv("submission.csv")

# EDA on train data
df_train['Date'] = pd.to_datetime(df_train.Date)
df_train = df_train.fillna({"Province_State": "None"})
df_train["Country_Region"] = df_train["Country_Region"].astype('category')
df_train["Province_State"] = df_train["Province_State"].astype('category')
df_train["Country_cat"] = df_train["Country_Region"].cat.codes

# EDA on test data
df_test['Date'] = pd.to_datetime(df_test.Date)
df_test = df_test.fillna({"Province_State": "None"})

df_train.info()

df_sample = df_train.loc[df_train['Country_Region'] == 'Algeria']

print(df_sample.head())

Y = df_train["Date"]
X = df_train["ConfirmedCases"]

sns.set(style="whitegrid")
sns.barplot(x="ConfirmedCases", y="Country_Region", data=df_train)
plt.show()
