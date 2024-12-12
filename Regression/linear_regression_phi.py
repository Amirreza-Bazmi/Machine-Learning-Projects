import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
from sklearn.metrics import r2_score
import pandas as pd


# Reading file
data = pd.read_csv('data.csv')

landa=0.001
# Separating input and output
X = data[['Mileage']]
y = data['Price']

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=10)

phi_train=np.concatenate((np.ones((len(x_train),1)),x_train),axis=1)

# Training Step
weight=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi_train),phi_train)+landa*np.eye(np.shape(phi_train)[-1])),
                     np.transpose(phi_train)),y_train)

# Test Phase
phi_test=np.concatenate((np.ones((len(x_test),1)),x_test),axis=1)

# Predicting
y_pred = np.matmul(phi_test,weight)

r_squared = r2_score(y_test, y_pred)
print('R-squared Score:', r_squared)

# Plot outputs
plt.scatter(x_train, y_train, color= "black", label= "Data")
# plt.scatter(x_test, y_pred, color= "red", label= "Test data")
plt.plot(x_test, y_pred, color= "blue", linewidth= 3, label= "Linear Regression: Mileage")
# plt.plot(x_test, y_test, color='green')
plt.title("Linear Regression: Price vs Mileage  R^2 = %.4f" % r_squared)
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.legend()
plt.savefig("Linear Regression with phi function (mileage).png")
plt.show()