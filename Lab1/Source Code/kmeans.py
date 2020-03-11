"""
Part 6:
Run-through of exploratory data analysis, evaluation, comments, and conclusions printed to the output.
Please follow through with the output.
Dataset: https://www.kaggle.com/yersever/500-person-gender-height-weight-bodymassindex
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')

print("Lets take a look at our dataset:")
print(df.head(), end='\n\n')

print("Before we do anything let's go through some EDA.")

print("First change the Gender attribute to an int:")
gender_dict = {"Male": 0, "Female": 1}
new_gender_column = []
for g in df["Gender"]:
    new_gender_column.append(gender_dict[g])
df["Gender"] = new_gender_column
df.info()
print()

print("We can see that there is no null values present so we are good there.", end='\n\n')

print("Lets look at correlations of attritubes to Index and see if anything stands out:")
print(abs(df[df.columns[:]].corr()["Index"][:-1]))
print("Expectedly, gender has practically no correlation and as clustering goes, we have no use of a binary attribute\n"
      "so lets remove it.", end='\n\n')
df = df.drop("Gender", axis=1)

print("Now lets check for outliers in the attributes:")
plt.title("Checking for outlier data...")
df.boxplot(column=list(df.columns))
plt.show()
print("Looks like there is no outliers to worry about.", end='\n\n')


print("Lets visualize clusters with Height and Weight to our target Index.")
sns.FacetGrid(df, hue="Index", height=4).map(plt.scatter, "Height", "Weight").add_legend()
plt.title("Cluster with Height and Weight")
plt.show()

print("No need to visualize Gender since it is a binary value.", end='\n\n')
print("We are done with EDA and can move on the KMeans.", end='\n\n')

print("Now lets use the elbow method to find the best value for K in the KMeans model:")
x = df.iloc[:,[0,1]]

wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 15), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

print("Its not super definitive where the optimal number lands but we will go with 3.", end='\n\n')

print("Lets see what we get when we apply KMeans:")
k = 3
km = KMeans(n_clusters=k)
km.fit(x)

y_clusters = km.predict(x)
from sklearn import metrics
print("3Means silhouette score:", metrics.silhouette_score(x, y_clusters))

