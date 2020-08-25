#import tensorflow
#import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from matplotlib import style
import matplotlib.pyplot as pyplot
import pickle
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
# Since our data is seperated by semicolons we need to do sep=";"
print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())
predict = "G3"

X = np.array(data.drop([predict], 1)) # Features
y = np.array(data[predict]) # Labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

best =0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test) # acc stands for accuracy
    print(acc)
    if(acc>best):
        best=acc
        with open("linear_regression_student.pickle", "wb") as f:
            pickle.dump(linear, f)



pickle_in= open("linear_regression_student.pickle", "rb")

linear2=pickle.load(pickle_in)


print('Coefficient: \n', linear2.coef_) # These are each slope value
print('Intercept: \n', linear2.intercept_) # This is the intercept

predictions = linear2.predict(x_test) # Gets a list of all predictions

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p="G2"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("actual score")
pyplot.show()
