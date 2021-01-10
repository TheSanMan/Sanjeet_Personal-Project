import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt #from matplotlib import style #import seaborn as sns

data = pd.read_csv("data.csv")

print(len(data))

data = data[
    ["danceability", "energy", "acousticness", "duration_ms", "popularity", "tempo", "instrumentalness",
     "liveness", "year", "loudness", "speechiness"]]

predict = "popularity"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)

acc = linear.score(x_test, y_test)
print(acc)

print(linear.coef_)

#Lplt.show(sns.jointplot(x="speechiness", y="popularity", data=data, kind='reg', joint_kws={'line_kws':{'color':'red'}}))




#print("Coefficient: \n", linear.coef_)
#print("Intercept \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#print(linear.score(x, y))
