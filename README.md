# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages required. 
2.Read the dataset. 
3.Define X and Y array. 
4.Define a function for costFunction,cost and gradient. 
5.Define a function to plot the decision boundary and predict the Regression value. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: S.PARTHSASARATHI
RegisterNumber:  212223040144
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1) 
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```

## Output:
DATASET
![370538442-f749cb30-9114-408a-8bbf-e42d50803a67](https://github.com/user-attachments/assets/909ff800-20bd-47c1-8963-5173193f831c)
Datatypes of Dataset
![370538745-72cece4b-4155-4968-8dfd-c437bb655ee8](https://github.com/user-attachments/assets/77ce16a3-dd2c-4216-bbf3-6b94cb126457)
Labeled Dataset
![370538860-f82bacdb-3d21-4f6f-b24d-1aa7acf9a46a](https://github.com/user-attachments/assets/da0e3c44-ac5e-4af0-ba68-5b83227f01c6)
Y value (dependent variable)
![370538989-aa738acc-4f9c-4bee-b5fb-e339c36c47da](https://github.com/user-attachments/assets/01a98839-11b5-491b-a27b-009a97d30547)
Accuracy
![370539377-e157c4fa-ed30-4c1c-ac92-2ed48ade3882](https://github.com/user-attachments/assets/584e21fc-836e-4c95-8845-19665bae0ec2)
Predicted Y value
![370539506-b03123a8-b7b9-4595-9b34-d718fec6697c](https://github.com/user-attachments/assets/0a24b5a4-f13e-4bad-9eb0-942ed60e9606)
Y value
![370540074-5f16737f-f6bf-4eed-b9e0-6872fb31fb39](https://github.com/user-attachments/assets/bdd718c0-c028-4eac-9c44-1f295baa9357)
New Y predictions
![370540160-eb52d01c-0eed-4157-96cf-8a656dc5e1cb](https://github.com/user-attachments/assets/e4dbcc20-485c-4a02-8ee6-b822204c5db0)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

