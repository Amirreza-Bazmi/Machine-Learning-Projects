# Linear Regression Project

This repository contains code and plots for performing **linear regression** on a car pricing dataset. It explores the relationships between the car's price and various features such as mileage, cylinder, cruise control, and more.

---

## 📂 Files in the Repository

1. **`data.csv`**: The dataset containing car prices and their corresponding features.
2. **`linear_regression.py`**: Implements linear regression using the `scikit-learn` library.
3. **`linear_regression_phi.py`**: Implements linear regression manually with a custom feature (phi) transformation and regularization.
4. **Plots**:
   - [Linear Regression with Mileage]()
   - [Linear Regression with Cylinder](https://github.com/Amirreza-Bazmi/Machine-Learning-Projects/blob/main/Regression/2-Linear%20Regression-Cylinder.png)
   - [Linear Regression with Cruise](https://github.com/Amirreza-Bazmi/Machine-Learning-Projects/blob/main/Regression/2-Linear%20Regression-Cruise.png)
   - [Linear Regression with Doors](https://github.com/Amirreza-Bazmi/Machine-Learning-Projects/blob/main/Regression/2-Linear%20Regression-Doors.png)
   - [Linear Regression with Leather](https://github.com/Amirreza-Bazmi/Machine-Learning-Projects/blob/main/Regression/2-Linear%20Regression-Leather.png)

---

## 📊 Features and Workflow

1. **Data Preparation**:
   - Reads the dataset (`data.csv`).
   - Splits data into training and test sets.
   
2. **Linear Regression Implementation**:
   - **With Scikit-learn** (`linear_regression.py`):
     - Trains and evaluates a linear regression model.
     - Supports multi-feature input.
   - **Manual Approach** (`linear_regression_phi.py`):
     - Applies a custom transformation (phi function).
     - Includes regularization with a lambda parameter.

3. **Visualization**:
   - Generates scatter plots of the data alongside the regression line.
   - Saves the plots as PNG files for features like mileage, cruise control, cylinders, and more.

---

## 🔧 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Amirreza-Bazmi/Machine-Learning-Projects.git
   cd Machine-Learning-Projects/Regression
