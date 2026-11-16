# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. 6.Compare the graphs and hence we obtained the linear regression for the given data.
## Program:
```
*/
Developed by: MEGANATHAN R
RegisterNumber:212224230156


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create Dataset
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10],
    "Marks_Scored": [35,40,50,55,60,65,70,80,85,95]
}

df = pd.DataFrame(data)

# Display Dataset
print("Dataset:\n", df.head())

# Independent and Dependent Variables
X = df[["Hours_Studied"]]
y = df["Marks_Scored"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model Parameters
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

# Evaluation
print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red',
         linewidth=2, label="Regression Line")

plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

# Predict Marks
hours = 7.5
predicted_marks = model.predict(
    pd.DataFrame({"Hours_Studied": [hours]})
)

print("\nPredicted Marks for", hours,
      "hours =", round(predicted_marks[0], 2))

*/
```

## Output:
<img width="835" height="732" alt="image" src="https://github.com/user-attachments/assets/d60f74e3-99f6-45c3-b6d8-3496647fcc1f" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
