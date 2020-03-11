"""
Part 5:
Run-through of exploratory data analysis, evaluation, comments, and conclusions printed to the output.
Please follow through with the output.
Dataset: https://www.kaggle.com/jeevannagaraj/indian-liver-patient-dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats
from sklearn.model_selection import train_test_split


print("We are using a dataset of patient data targeted at the presence of cardiovascular disease.\n")
df = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv')

print("First lets see what the data looks like:")
print(df.head(), end='\n\n')
print("And now some information:")
df.info()
print()


print("Looks like column alkphos has a few missing values. Lets look closer at those:")
print(df[df.isnull().any(axis=1)])
print("Since its only 4 rows, quite insignificant to the full size, lets go ahead and remove them:")
df = df.drop([209,241,253,312])
df.info()
print("No null values now.", end='\n\n')


print("We also see that we have one attribute (gender) that is not a numerical value and is instead an object.\n"
      "We'll change that:")
gender_dict = {"Male": 0, "Female": 1}
new_gender_column = []
for g in df["gender"]:
    new_gender_column.append(gender_dict[g])

df["gender"] = new_gender_column
df.info()
print("Gender is now an int type and binary value.", end='\n\n')


print("Lets look at correlations of attributes to the target and see if anything stands out:")
print(abs(df[df.columns[:]].corr()['is_patient'][:-1]))
print("Looks like every attribute has reasonable correlation other than 'sgpt' and 'gender'\n"
      "who are somewhat outliers both under 0.1. We will remove them.", end='\n\n')
df = df.drop("sgpt", axis=1)
df = df.drop("gender", axis=1)


print("Now lets check for outliers in the attributes:")
plt.title("Checking for outlier data...")
df.boxplot(column=list(df.columns))
plt.show()

print("We can see there are some drastic outliers that need to be removed but the outliers of some of these\n"
      "columns are still quite large in quantity and bunched up so we will cut off data around 1000.", end='\n\n')

z_scores = stats.zscore(df)

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = df[filtered_entries]

plt.title("Data cut off around 1000")
df.boxplot(column=list(df.columns))
plt.show()


print("Now that we have done a good amount of exploratory data analysis, we can apply classification algorithms:")
# get training and testing data
x_train = df.drop("is_patient", axis=1)
y_train = df["is_patient"]
x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x_train,y_train, test_size=0.2)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train_data, y_train_data)
print("Gaussian Naive Bayes model accuracy:", round(gnb.score(x_test_data, y_test_data) * 100, 2), end='%\n')

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train_data, y_train_data)
print("SVM model accuracy:", round(svc.score(x_test_data, y_test_data) * 100, 2), end='%\n')

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_data, y_train_data)
print("K-NN model accuracy:", round(knn.score(x_test_data, y_test_data) * 100, 2), end='%\n\n')

print("It looks like the SVM model was generally the best model for our dataset.\n"
      "Naive Bayes is consistently the worst out of the bunch probably because the things\n"
      "that make it good for multiple class problems like text classification weren't relevant\n"
      "to our datasets binary classification problem. K-NN accuracy is quite varying and dependent\n"
      "on the k value so it could be possible to get K-NN accuracy up more with more extensive\n"
      "evaluation of the model.", end='\n\n')
