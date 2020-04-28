from keras.models import Sequential
from keras.layers import Dense
from keras.layers import *

# load dataset
from pandas import DataFrame, concat
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()

df_train = pd.read_csv("train2.csv")
df_test = pd.read_csv("test2.csv")

df_submission = pd.read_csv("submission.csv")

# Get numerical values for the dates
all_dates = df_train.Date.append(df_test.Date)
all_dates = all_dates.astype('category')
date_dict_num_key = dict(enumerate(all_dates.cat.categories))
date_dict = {y: x for x, y in date_dict_num_key.items()}


    # EDA on train data
# Date attribute is changed to numerical value.
num_date_col = []
for date in df_train["Date"]:
    num_date_col.append(date_dict[date])

df_train["Date"] = num_date_col

# Province_State attributes null values are changed to None
df_train = df_train.fillna({"Province_State": "None"})

# Country_Region and Province_State attributes are changed to category object,
# then are changed to numerical values. A dictionary is made to be able to
# get the label for the numerical value in the future.
# First for Country_Region
df_train["Country_Region"] = df_train["Country_Region"].astype('category')
country_region_dict_num_key = dict(enumerate(df_train["Country_Region"].cat.categories))
country_region_dict = {y: x for x, y in country_region_dict_num_key.items()}
df_train["Country_Region"] = df_train["Country_Region"].cat.codes
# Now for Province_State
df_train["Province_State"] = df_train["Province_State"].astype('category')
province_state_dict_num_key = dict(enumerate(df_train["Province_State"].cat.categories))
province_state_dict = {y: x for x, y in province_state_dict_num_key.items()}
df_train["Province_State"] = df_train["Province_State"].cat.codes

df_train = df_train.fillna({"HumanDevIndex": 7})
df_train = df_train.fillna({"GovtType": "None"})
df_train["GovtType"] = df_train["GovtType"].astype('category')
df_train = df_train.fillna({"HospitalBeds": 2})
df_train = df_train.fillna({"Continent": "None"})
df_train["Continent"] = df_train["Continent"].astype('category')
df_train = df_train.fillna({"PopDensity": 50})
df_train = df_train.fillna({"AvgMarchTemp": 40})
# df_train["LockdownDate"] = pd.to_datetime(df_train.LockdownDate) # need to update this attribute's values

    # EDA on test data
# Date attribute is changed to numerical value.
num_date_col = []
for date in df_test["Date"]:
    num_date_col.append(date_dict[date])

df_test["Date"] = num_date_col

# Province_State attributes null values are changed to None
df_test = df_test.fillna({"Province_State": "None"})

# First for Country_Region
df_test["Country_Region"] = df_test["Country_Region"].astype('category')
df_test["Country_Region"] = df_test["Country_Region"].cat.codes
# Now for Province_State
df_test["Province_State"] = df_test["Province_State"].astype('category')
df_test["Province_State"] = df_test["Province_State"].cat.codes

df_test = df_test.fillna({"HumanDevIndex": 7})
df_test = df_test.fillna({"GovtType": "None"})
df_test["GovtType"] = df_test["GovtType"].astype('category')
df_test = df_test.fillna({"HospitalBeds": 2})
df_test = df_test.fillna({"Continent": "None"})
df_test["Continent"] = df_test["Continent"].astype('category')
df_test = df_test.fillna({"PopDensity": 50})
df_test = df_test.fillna({"AvgMarchTemp": 40})
# df_test["LockdownDate"] = pd.to_datetime(df_train.LockdownDate) # need to update this attribute's values

##############################################################################################
# Method for model

varis = 2 # amount of output sequences


def build_model():
    model = Sequential()
    model.add(LSTM(units=4, return_sequences=True, activation='tanh', input_shape=(None, varis)))
    #model.add(Dense(2))
    model.add(LSTM(units=varis, return_sequences=True))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


def build_stateful_model():
    stateful_model = Sequential()
    stateful_model.add(LSTM(4, return_sequences=True, stateful=True, batch_input_shape=(1, None, 2)))
    #stateful_model.add(Dense(2))
    stateful_model.add(LSTM(varis, return_sequences=False, stateful=True))
    return stateful_model


def fit_model(model, t_len=70):

    for x in range(0, int(len(df_train) / 70)):
        # Find at what data point Confirmed Cases are not all 0's that will be the beginning of the training set
        train = df_train[x * 70:x * 70 + t_len]
        train = train.loc[train['ConfirmedCases'] != 0]

        print(country_region_dict_num_key[train['Country_Region'].iloc[0]],
              province_state_dict_num_key[train['Province_State'].iloc[0]])

        norm = train.drop(['Date'], 1, inplace=True)
        norm = train.drop(['Province_State'], 1, inplace=True)
        norm = train.drop(['Id'], 1, inplace=True)
        norm = train.drop(['Country_Region'], 1, inplace=True)

        train = train.values
        train = min_max_scaler.fit_transform(train)

        trainx = train[0:len(train) - 1]
        trainy = train[1:len(train)]
        trainx = np.reshape(trainx, (1, len(trainx), varis))
        trainy = np.reshape(trainy, (1, len(trainy), varis))

        model.fit(trainx, trainy, epochs=20, batch_size=10, shuffle=False)

    model.save_weights('LSTMBasic1.h5')

    return model


def create_test_set(country, province):
    df_sample = df_test.loc[df_test['Country_Region'] == country_region_dict[country]]
    df_sample = df_sample.loc[df_sample["Province_State"] == province_state_dict[province]]
    df_sample = df_submission.loc[df_submission["ForecastId"].isin(df_sample["ForecastId"])]
    norm = df_sample.drop(['ForecastId'], 1, inplace=True)

    return df_sample


def model_predict(model, t_len, sample_country, sample_province):

    df_sample = df_train.loc[df_train['Country_Region'] == country_region_dict[sample_country]]
    df_sample = df_sample.loc[df_sample["Province_State"] == province_state_dict[sample_province]]

    x = df_sample['Date'][t_len:]

    #df_sample = create_test_set(sample_country, sample_province)

    norm = df_sample.drop(['Date'], 1, inplace=True)
    norm = df_sample.drop(['Province_State'], 1, inplace=True)
    norm = df_sample.drop(['Id'], 1, inplace=True)
    norm = df_sample.drop(['Country_Region'], 1, inplace=True)

    train = df_sample[:t_len]
    min_max_scaler.fit_transform(train)

    test = df_sample[t_len:]
    test = test.values

    inputs = np.reshape(test, (len(test), varis))
    inputs = min_max_scaler.transform(inputs)
    inputs = np.reshape(inputs, (1, len(inputs), varis))

    predict = model.predict(inputs)

    predict = np.reshape(predict, (len(predict[0]), varis))
    predict = min_max_scaler.inverse_transform(predict)

    plt.figure(figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

    print(test)
    print(predict)

    plt.plot(x, test[:, 0], color='yellow', label='Real Confirmed Cases')
    plt.plot(x, test[:, 1], color='red', label='Real Confirmed Fatalities')
    plt.plot(x, predict[:, 0], color='blue', label='Predicted Confirmed Cases')
    plt.plot(x, predict[:, 1], color='black', label='Predicted Confirmed Fatalities')

    plt.title('Confirmed Cases Prediction for ' + sample_country + " " + sample_province, fontsize=40)
    plt.xlabel('Time', fontsize=40)
    plt.ylabel('Confirmed Cases', fontsize=40)
    plt.legend(loc='best')
    plt.show()


################################################################################
    # manipulate datasets
#### Current do not have non-sequential data train implemented for the model so ignoring newly added columns
df_train = df_train[['Id','Province_State','Country_Region','Date','ConfirmedCases','Fatalities']]
df_test = df_test[['ForecastId','Province_State','Country_Region','Date']]


################################################################################
# Run functions and methods

# amount of training dataset to train the model with
training_percentage = 1
training_len = int(70 * training_percentage)

model_build = build_model()
#fit_model = fit_model(model_build, training_len)

# Comment fit_model line if predicting
model_build.load_weights('LSTMBasic1.h5')
model_predict(model_build, -5, "US", "Colorado")

