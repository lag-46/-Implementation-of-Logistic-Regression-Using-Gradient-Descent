# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1. Load the placement dataset.

2. Remove unwanted columns (sl_no and salary).

3. Convert categorical columns into category type.

4. Convert categorical values into numerical codes.

5. Separate Independent variables (X) and Dependent variable (Y).

6. Initialize model parameters (theta) randomly.

7. Compute linear combination using formula:
   Z = X * theta

8. Apply Sigmoid function to get probability:
   h = 1 / (1 + e^(-Z))

9. Calculate Loss using Log Loss formula:
   Loss = - Σ [ y log(h) + (1 - y) log(1 - h) ]

10. Calculate Gradient using formula:
    Gradient = (1 / m) * Xᵀ * (h - y)

11. Update parameters using Gradient Descent formula:
    theta = theta - alpha * Gradient

12. Repeat prediction and update steps for given iterations.

13. Predict output using:
    h = sigmoid(X * theta)

14. Convert probability to class label using rule:
    If h >= 0.5 → Class = 1
    If h < 0.5 → Class = 0

15. Calculate Accuracy using formula:
    Accuracy = Number of Correct Predictions / Total Predictions

16. Predict output for new input data using trained theta.






## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: PANDEESWARAN N
RegisterNumber:  212224230191
*/
```
```
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
data=pd.read_csv("C:\\Users\\admin\\Downloads\\Placement_Data (1).csv") 
data

data.head()


data1=data.copy()
data1.head()
data1=data.drop(['sl_no','salary'],axis=1)
data1

data=data.drop('sl_no',axis=1) 
data=data.drop('salary',axis=1) 

data["gender"]=data["gender"].astype('category') 
data["ssc_b"]=data["ssc_b"].astype('category') 
data["hsc_b"]=data["hsc_b"].astype('category') 
data["degree_t"]=data["degree_t"].astype('category') 
data["workex"]=data["workex"].astype('category') 
data["specialisation"]=data["specialisation"].astype('category') 
data["status"]=data["status"].astype('category') 
data["hsc_s"]=data["hsc_s"].astype('category') 
data.dtypes


data["gender"]=data["gender"].cat.codes 
data["ssc_b"]=data["ssc_b"].cat.codes 
data["hsc_b"]=data["hsc_b"].cat. codes
data["degree_t"]=data["degree_t"].cat.codes 
data["workex"]=data["workex"].cat.codes 
data["specialisation"]=data["specialisation"].cat.codes 
data["status"]=data["status"].cat.codes 
data["hsc_s"]=data["hsc_s"].cat.codes 
data


x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

theta = np.random.randn(x.shape[1])

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h) + (1-y)*np.log(1-h))

def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)

def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred = predict(theta, x)

accuracy = np.mean(y_pred.flatten() == y)

print("Accuracy:", accuracy)
print("Predicted:\n", y_pred)

print("Actual:\n", y)



xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
print("Predicted Result:",y_prednew)


```

## Output:

## Read the file and display :

<img width="1250" height="530" alt="image" src="https://github.com/user-attachments/assets/cb3010e1-0e64-46cf-a061-802aace14fd9" />


## HEAD :

<img width="1253" height="255" alt="image" src="https://github.com/user-attachments/assets/c7e159bc-8d56-4ec4-b8e4-8a764420bb27" />

## Cleaned Dataset :

<img width="1248" height="501" alt="image" src="https://github.com/user-attachments/assets/dc2ef211-470c-4822-af36-bf21a913cf14" />

## Categorizing columns :

<img width="955" height="525" alt="image" src="https://github.com/user-attachments/assets/761cd640-bba5-4467-b8fe-eaa2084f0c7e" />

## Labelling columns and displaying dataset :

<img width="1253" height="607" alt="image" src="https://github.com/user-attachments/assets/8e20657a-a99e-460b-9f5b-533f09451d3f" />

## Model Training :

<img width="930" height="787" alt="image" src="https://github.com/user-attachments/assets/119fa310-cff6-4aab-828a-293dcac36d34" />

## Model Prediction :

<img width="826" height="161" alt="image" src="https://github.com/user-attachments/assets/9256be3c-f8c9-41ab-9ecd-62ba49a6f79d" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

