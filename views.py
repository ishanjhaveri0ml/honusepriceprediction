from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def home(request):
    return render(request, "home.html")


def predict(request):
    return render(request, "predict.html")


def result(request):
    data = pd.read_csv("D:/USA_Housing.csv")
    data = data.drop(['Address'], axis=1)
    data.shape
    X = data.drop('Price', axis=1)
    Y = data['Price']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=30, random_state=2)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    var1 = float(request.GET.get('n1'))
    var2 = float(request.GET.get('n2'))
    var3 = float(request.GET.get('n3'))
    var4 = float(request.GET.get('n4'))
    var5 = float(request.GET.get('n5'))

    input_data = np.array([var1, var2, var3, var4, var5]).reshape(1, -1)



    pred = model.predict(input_data)
    pred = round(pred[0])

    price = "the predicted price is $" + str(pred)

    return render(request, "predict.html", {"result2": price})
