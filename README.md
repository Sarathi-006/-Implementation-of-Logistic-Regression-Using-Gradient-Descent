# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load pandas, numpy, and matplotlib for data handling, computation, and visualization.

2.Read the dataset (Placement_Data.csv).Drop unnecessary columns (sl_no, salary).

3.Convert categorical variables to numerical codes using .astype('category') and .cat.codes.

4.Sigmoid Function: Compute probabilities using ℎ(𝑧)=11+𝑒−𝑧h(z)= 1/1+e −z.

5.Initialize random weights (theta) and use gradient_descent() to optimize weights over multiple iterations.

6.Compute accuracy by comparing predicted values (y_pred) with actual values (Y).

7.Print accuracy, predictions for the dataset, and predictions for new inputs.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: S.PARTHSASARATHI
RegisterNumber:  212223040144
*/
```
```

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
y = Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
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
Dataset
![329310796-5bfd1bab-c917-439a-b704-ccc834b3c6bb](https://github.com/user-attachments/assets/252ac870-60fe-46f6-82fe-d1832469f28a)
Data types
![329310891-673238d6-78f7-4f11-ae12-7a4d37373d34](https://github.com/user-attachments/assets/3678b54f-2416-4658-92a4-2386efb53232)
New dataset
![329310981-2f3befdc-e483-4622-b528-d094fd91bcf4](https://github.com/user-attachments/assets/18b7ce78-66a3-4e18-8240-05eb1dc96b86)
Y values
![329311112-62bcf3a1-a6d6-4105-852d-31b05f5cab2d](https://github.com/user-attachments/assets/482998ef-74de-4d6e-8ce6-e30b069323b8)
Accuracy
![329311917-4b336b75-ee0a-488d-898a-ec7c66288674](https://github.com/user-attachments/assets/f04529db-40a1-47da-9e1d-987ed59f202e)
Y pred
![329311479-ede9b220-c365-4f22-914c-5d0b62c1c720](https://github.com/user-attachments/assets/19a44d43-8209-4299-9e5d-a2109a8fed56)
New Y
![329311539-ea7536cc-4ec7-443b-8c69-82a311110908](https://github.com/user-attachments/assets/e78ceb8b-54f7-4925-bd5a-f0f2c43670ff)
![329311579-a1e09cc3-458a-45a3-ba60-deaf87e554e8](https://github.com/user-attachments/assets/01e3a889-882e-4cd6-ae5c-f4aaf02b56b9)
![329311628-e5795b77-f293-4244-b6aa-40a4414635cc](https://github.com/user-attachments/assets/12ea5da2-8f88-4ca5-bffe-d46c340ad547)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

