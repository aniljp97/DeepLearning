import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

df_train = pd.read_csv("train2.csv")
df_test = pd.read_csv("test.csv")

df_submission = pd.read_csv("submission.csv")

# EDA on train data
# Province_State attributes null values are changed to None
df_train = df_train.fillna({"Province_State": "None"})

# Remove outliers data
df_train = df_train[df_train["Country_Region"] != "Diamond Princess"]
df_train = df_train[df_train["Country_Region"] != "Holy See"]
df_train = df_train[df_train["Country_Region"] != "Taiwan*"]

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
df_train["GovtType"] = df_train["GovtType"].cat.codes

df_train = df_train.fillna({"HospitalBeds": 2})

df_train = df_train.fillna({"Continent": "None"})
df_train["Continent"] = df_train["Continent"].astype('category')
df_train["Continent"] = df_train["Continent"].cat.codes

df_train = df_train.fillna({"PopDensity": 50})
new_popdensity_col = []
for i in df_train["PopDensity"]:
    new_popdensity_col.append(i.replace(',', ''))
df_train["PopDensity"] = new_popdensity_col

df_train = df_train.fillna({"AvgMarchTemp": 40})

df_train["Lockdown"] = df_train["Lockdown"].astype('category')
df_train["Lockdown"] = df_train["Lockdown"].cat.codes

new_population_col = []
for i in df_train["Population"]:
    new_population_col.append(i.replace(',', ''))
df_train["Population"] = new_population_col


# EDA on test data
# Province_State attributes null values are changed to None
df_test = df_test.fillna({"Province_State": "None"})

# Remove outliers data
df_test = df_test[df_test["Country_Region"] != "Diamond Princess"]
df_test = df_test[df_test["Country_Region"] != "Holy See"]
df_test = df_test[df_test["Country_Region"] != "Taiwan*"]

# First for Country_Region
df_test["Country_Region"] = df_test["Country_Region"].astype('category')
df_test["Country_Region"] = df_test["Country_Region"].cat.codes
# Now for Province_State
df_test["Province_State"] = df_test["Province_State"].astype('category')
df_test["Province_State"] = df_test["Province_State"].cat.codes


# EDA on submission data
# Keep ForecastID only of those that are also in df_test
df_submission = df_submission[df_submission["ForecastId"].isin(df_test["ForecastId"])]

################################################################################
# manipulate datasets
# selecting columns are relevant columns
df_train = df_train[['HumanDevIndex', 'GovtType', 'HospitalBeds', 'Continent', 'PopDensity', 'AvgMarchTemp', 'Lockdown',
                     'Province_State', 'Country_Region', 'ConfirmedCases', 'Fatalities', 'Date', 'Population']]
# df_test = df_test[['Province_State', 'Country_Region', 'Date']]

#########################################################################################
# Attempt with multi-input multi-step forecasting using CNN; METHODS

# training and testing data
t_len = 60  # length of x training for section of sequential data (the rest is for y training).  For prediction, may
# want to keep (seq_len - t_len) equal to a factor of 30 so that will get the extra 30 days of data wanted
# for submission.  NOTE:: using timeStepOptimize() method, found 60 to be best for training.
seq_len = 70  # length of each sequence of data

n_features_in = 11  # number of features
n_features_out = 2

# the number of time steps as decided by t_len variable
n_steps_in, n_steps_out = t_len, (seq_len - t_len)

n_output = 0  # number of outputs determined by y training input for the model (defined in getTrainingInput())


def getTrainingInput():
    """
    Get input that will be used training the CNN model from train2.csv

    :return: X and Y arrays for training the model.
    Sizes of X and Y depend on set t_len and n_features_in variables from above.
    """
    X, y = list(), list()  # x and y training for the CNN model

    # put sequences of each area into respective x and y lists for training the model
    for i in range(0, int(len(df_train) / seq_len)):
        x_train = df_train[i * seq_len:i * seq_len + n_steps_in]
        y_train = df_train[i * seq_len + n_steps_in:i * seq_len + seq_len]

        # print(country_region_dict_num_key[y_train['Country_Region'].iloc[0]],
        #       province_state_dict_num_key[y_train['Province_State'].iloc[0]])

        # get population form the sample for scaling
        population = int(x_train['Population'].iloc[0])

        x_train.drop(['Date'], 1, inplace=True)
        x_train.drop(['Population'], 1, inplace=True)

        y_train = y_train[['ConfirmedCases', 'Fatalities']]

        x_train = array(x_train).reshape(len(x_train), n_features_in)
        y_train = array(y_train).reshape(n_steps_out, n_features_out)

        X.append(x_train)
        y.append(y_train)

    y = array(y)
    # get number of outputs, assign it to global variable
    global n_output
    n_output = y.shape[1] * y.shape[2]

    return array(X), y


def createModel():
    """
    Create the CNN model.  Hyperparamters have been changed according to optimization finding from running tests with
    hyperParameterOptimize(x, y).

    :return: the created model
    """
    # define CNN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features_in)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_output))
    # compile model
    model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])
    return model


def fitSaveModel():
    """
    Fits/trains the CNN model with the input of getTrainingInput(). Saves the model as 'CNNModel.h5'.

    :return: the fitted model and the history object for that fitting of the model
    """
    X, y = getTrainingInput()
    y = y.reshape((y.shape[0], n_output))  # flatten output
    # create model
    model = createModel()
    # train and save model
    history = model.fit(X, y, batch_size=40, epochs=5000, verbose=2)
    model.save('./CNNModel.h5')
    return model, history


def hyperParameterOptimize(x, y):
    """
    Gives insight on the optimal hyperparameters to use for the CNN model.

    :param x: x input going into training the CNN model. Create with getTrainingInput().
    :param y: y input going into training the CNN model. Create with getTrainingInput().
    :return: None, prints insight to the Run output.
    """
    model = KerasClassifier(build_fn=createModel, verbose=0)
    # WHAT WE ARE TESTING
    # for batch_size and epochs # ::batch_size=40, epochs=5000
    batch_size = [40, 60, 80, 100]
    epochs = [3000, 4000, 5000, 6000, 7000, 8000]
    param_grid = dict(batch_size=batch_size, epochs=epochs)

    # for optimizers # ::adam
    # optimizer = ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # param_grid = dict(optimizer=optimizer)

    # for activation functions # ::relu
    # activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    # param_grid = dict(activation=activation)

    # for neurons in hidden layer # ::50
    # neurons = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    # param_grid = dict(neurons=neurons)

    # for loss functions # ::mae
    # losses = ['mse', 'mae']
    # param_grid = dict(loss=losses)

    # run search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    y = y.reshape((y.shape[0], n_output))  # flatten output
    grid_result = grid.fit(x, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def submissionPredictAll(model):
    """
    Predicts all data needed to go into the submission.csv file. The submission file is written to with this data.

    :param model: the model that will be used to predict the future Covid-19 data needed for submission.
    :return: array of all the submission values. ['Confirmed Cases', 'Fatalities']
    """
    submission = array([[0,0]])  # initialize and remove placeholder first row later.

    for i in range(0, int(len(df_train) / seq_len)):
        df_sample = df_train[i * seq_len:i * seq_len + seq_len]

        # Drop data irrelevant to model fitting from the sample
        df_sample.drop(['Date'], 1, inplace=True)
        df_sample.drop(['Population'], 1, inplace=True)

        # reshape df_sample data to proper input for model prediction
        x_input = array(df_sample[:t_len]).reshape(t_len, n_features_in)
        x_input = x_input.reshape((1, n_steps_in, n_features_in))

        # get only output columns
        df_sample = df_sample[["ConfirmedCases", "Fatalities"]]
        # get expected values of the first timestep
        expected = array(df_sample[t_len:]).reshape(n_steps_out, n_features_out)  # expected values

        # predict future values with prediction stepping
        submission_values = array(df_sample[57:]).reshape(13, n_features_out)

        steps = int(30 / n_steps_out)  # steps until needed future data is predicted
        new_seq = expected  # the next piece of data to be added to the next prediction
        prev_seq = x_input

        # predict needed future data in steps
        for i in range(steps):
            # concatenate the previous sequence with the new sequence and remove the first values at length of new seq
            prev_seq = prev_seq.reshape(n_steps_in, n_features_in)
            categorical_columns = prev_seq[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]  # get categorical data
            prev_seq = prev_seq[:, [-2, -1]]  # make prev_seq equal only to the sequential columns of data
            step_seq = np.vstack((prev_seq, new_seq))[len(new_seq):]

            # predict the curr sequence
            step_seq = np.hstack((categorical_columns, step_seq))
            step_seq = step_seq.reshape((1, n_steps_in, n_features_in))
            pred = model.predict(step_seq, verbose=0)

            # prepare sequences for next step of iteration
            new_seq = pred.reshape(n_steps_out, n_features_out).astype(int)
            prev_seq = step_seq

            # concatenate new found data to the submission values
            submission_values = np.vstack((submission_values, new_seq))

        submission = np.vstack((submission,submission_values))

    submission = submission[1:]  # removed first row placeholder made when initializing
    df_submission["ConfirmedCases"] = submission[:, 0]
    df_submission["Fatalities"] = submission[:, -1]

    # write new data to my_submission.csv
    df_submission.to_csv('submission.csv', index=False)

    return submission


def getPredictFromSubmission(country, province):
    """
    Pulls the predicted Confirmed Cases and Fatalities of a specific province/state, country/region directly from the
    submission.csv file.  Plots the data, labeling known values and predicted values.

    :param country: country/region within 'Country_Region' attribute.
    :param province: province/state within 'Province_State' attribute.
    :return: None, only plots specified data.
    """
    # get sample of data for area
    df_sample = df_test.loc[df_test['Country_Region'] == country_region_dict[country]]
    df_sample = df_sample.loc[df_sample["Province_State"] == province_state_dict[province]]

    # get dates from the sample for plotting
    dates = df_sample['Date'][:]

    prediction = array(df_submission[df_submission["ForecastId"].isin(df_sample["ForecastId"])])
    prediction = prediction[:, [-2,-1]]

    plt.figure(figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    # The known data
    plt.plot(dates[:13], prediction[:13, 0], color='yellow', label='Real Confirmed Cases')
    plt.plot(dates[:13], prediction[:13, 1], color='red', label='Real Confirmed Fatalities')
    # The predicted data
    plt.plot(dates[12:], prediction[12:, 0], color='blue', label='Predicted Confirmed Cases')
    plt.plot(dates[12:], prediction[12:, 1], color='black', label='Predicted Confirmed Fatalities')
    # Labeling
    plt.title('Future Prediction for ' + country + ", " + province, fontsize=40)
    plt.xlabel('Time', fontsize=40)
    plt.xticks(rotation=30)
    plt.ylabel('Confirmed Cases', fontsize=40)
    plt.legend(loc='best')
    plt.show()


def sampleSubmissionPredict(model, country, province):
    """
    Gets future predictions of data for the dates specified by test.csv of a specific province/state, country/region.
    First plot shows the first timestep comparing the data to the actual data given. (Size of timestep in global var).
    Predicts future data by the defined timestep size and combined to be need test data. Plots the data, labeling known
    values and predicted values.

    :param model: CNN model used for predictions
    :param country: country/region within 'Country_Region' attribute.
    :param province: province/state within 'Province_State' attribute.
    :return: array of values of the specified submission data.
    """
    # get sample of data for area
    df_sample = df_train.loc[df_train['Country_Region'] == country_region_dict[country]]
    df_sample = df_sample.loc[df_sample["Province_State"] == province_state_dict[province]]
    # get population form the sample for scaling
    population = int(df_sample['Population'].iloc[0])
    # get dates from the sample for plotting
    dates = df_sample['Date'][n_steps_in:]
    # Drop data irrelevant to model fitting from the sample
    df_sample.drop(['Date'], 1, inplace=True)
    df_sample.drop(['Population'], 1, inplace=True)

    # reshape df_sample data to proper input for model prediction
    x_input = array(df_sample[:t_len]).reshape(t_len, n_features_in)
    x_input = x_input.reshape((1, n_steps_in, n_features_in))

    # get only output columns
    df_sample = df_sample[["ConfirmedCases", "Fatalities"]]

    # demonstrate prediction on n_steps_out of data
    yhat = model.predict(x_input, verbose=0)
    expected = array(df_sample[t_len:]).reshape(n_steps_out, n_features_out)  # expected values
    predicted = yhat.reshape(n_steps_out, n_features_out).astype(int)  # predicted values

    # plot predicted data that compares the accuracy of last few days of predict data to the real data
    plt.figure(figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(dates, expected[:, 0], color='yellow', label='Real Confirmed Cases')
    plt.plot(dates, expected[:, 1], color='red', label='Real Confirmed Fatalities')
    plt.plot(dates, predicted[:, 0], color='blue', label='Predicted Confirmed Cases')
    plt.plot(dates, predicted[:, 1], color='black', label='Predicted Confirmed Fatalities')
    plt.title("Last " + str(n_steps_out) + " Days of " + country + ", " + province + " Data", fontsize=40)
    plt.xlabel('Time', fontsize=40)
    plt.ylabel('Cases/Fatalities', fontsize=40)
    plt.legend(loc='best')
    plt.show()

    ####################################################################################################################

    # predict future values with prediction stepping
    submission_values = array(df_sample[57:]).reshape(13, n_features_out)

    steps = int(30 / n_steps_out)  # steps until needed future data is predicted
    new_seq = expected  # the next piece of data to be added to the next prediction
    prev_seq = x_input

    # predict needed future data in steps
    for i in range(steps):
        # concatenate the previous sequence with the new sequence and remove the first values at length of new sequence
        prev_seq = prev_seq.reshape(n_steps_in, n_features_in)
        categorical_columns = prev_seq[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]  # get categorical data
        prev_seq = prev_seq[:, [-2, -1]]  # make prev_seq equal only to the sequential columns of data
        step_seq = np.vstack((prev_seq, new_seq))[len(new_seq):]

        # predict the curr sequence
        step_seq = np.hstack((categorical_columns, step_seq))
        step_seq = step_seq.reshape((1, n_steps_in, n_features_in))
        pred = model.predict(step_seq, verbose=0)

        # prepare sequences for next step of iteration
        new_seq = pred.reshape(n_steps_out, n_features_out).astype(int)
        prev_seq = step_seq

        # concatenate new found data to the submission values
        submission_values = np.vstack((submission_values, new_seq))

    # plot predicted data
    date_sample = df_test.loc[df_test['Country_Region'] == country_region_dict[country]]
    date_sample = date_sample.loc[date_sample["Province_State"] == province_state_dict[province]]
    dates = date_sample['Date']  # get dates from the sample for plotting

    plt.figure(figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    # The known data
    plt.plot(dates[:13], submission_values[:13, 0], color='yellow', label='Real Confirmed Cases')
    plt.plot(dates[:13], submission_values[:13, 1], color='red', label='Real Confirmed Fatalities')
    # The predicted data
    plt.plot(dates[12:], submission_values[12:, 0], color='blue', label='Predicted Confirmed Cases')
    plt.plot(dates[12:], submission_values[12:, 1], color='black', label='Predicted Confirmed Fatalities')
    # Labeling
    plt.title('Future Prediction for ' + country + ", " + province, fontsize=40)
    plt.xlabel('Time', fontsize=40)
    plt.xticks(rotation=30)
    plt.ylabel('Confirmed Cases', fontsize=40)
    plt.legend(loc='best')
    plt.show()

    return submission_values


def timeStepOptimize():  # best found was 60
    """
    Get insight of CNN model on different sizes of timesteps to optimize used timestep size. Note: testing asks for
    30 days past given data. To get proper number of predicted days, the size of the timestep must be a factor of 30.

    :return: None, prints insight to the Run output.
    """
    accuracies = list()
    losses = list()

    # We want our y training timesteps to be factors of 30, so make the training lengths we are testing accordingly
    train_lens = [40, 55, 60, 64, 65, 67, 69]  # 30,15,10,6,5,3,1
    for t in train_lens:
        print("Testing " + str(t) + "...")
        # change the global value for the training timestep length
        global t_len
        t_len = t
        model, history = fitSaveModel()
        accuracies.append(history.history['accuracy'][-1])
        losses.append(history.history['loss'][-1])

    best_index = accuracies.index(max(accuracies))
    print("Best: " + str(accuracies[best_index]) + " using " + str(train_lens[best_index]))
    for t in range(len(train_lens)):
        print(str(accuracies[t]) + " (" + str(losses[t]) + ") with: " + str(train_lens[t]))


########################################################################################################################
# Run methods for model training and prediction

# running method to optimize hyper parameters of the model
# X, Y = getTrainingInput()
# hyperParameterOptimize(X, Y)

# running method to optimize the timestep size for the input going into the model # ::60
# timeStepOptimize()

# train and save the model as 'CNNModel.h5'
fitSaveModel()  # comment when model has been saved

# load model
cnn_model = load_model('./CNNModel.h5')

# demonstrate prediction on an area #
# select an area. If the country/region has no states/provinces, put "None" for province_state variable
country_region = "United Kingdom"
province_state = "None"

# apply model.predict on a specific country/region, province/state
print(sampleSubmissionPredict(cnn_model, country_region, province_state))

# write to the submission.csv file, and get a prediction from a specific country/region, province/state
# submissionPredictAll(cnn_model)
# getPredictFromSubmission(country_region, province_state)
