# Simple and Weighted Linear Regression

This repository contains Python implementations for **Simple Linear Regression** and **Weighted Linear Regression**. The project analyzes the relationship between input features and output labels, visualizes the results, and demonstrates the effect of weighted regression using a Gaussian kernel.

---

## 📂 Files in the Repository

1. **`Simple and Weighted Linear Regression.py`**: Python script implementing:
   - **Simple Linear Regression**
   - **Weighted Linear Regression** (Gaussian kernel-based).
2. **`features.txt`**: Input feature values (independent variable \( x \)).
3. **`labels.txt`**: Output label values (dependent variable \( y \)).
4. **Output Plots**:
   - Simple Linear Regression: A straight-line fit.
   - Weighted Linear Regression: A smooth curve fit based on Gaussian weights.

---

### Input Parameters: **Understanding \( \tau \) (Tau)**

When running the script, you will be prompted to input the **\( \tau \)** value, which controls the **bandwidth parameter** for the Gaussian kernel in Weighted Linear Regression.

#### What is \( \tau \)?
- \( \tau \) determines how much influence each data point has during regression based on its distance to the test point.
- The Gaussian kernel function is defined as:
   \[
   K(x_1, x_2) = \exp\left(-\frac{(x_1 - x_2)^2}{2 \cdot \tau^2}\right)
   \]
   - **Nearby points** have larger kernel values and contribute more.
   - **Distant points** have smaller kernel values and contribute less.

#### How \( \tau \) Affects the Model:
- **Small \( \tau \) (e.g., 0.1)**:
   - The model gives high weight to nearby points and less to distant points.
   - Results in a **highly localized** fit, capturing small variations in data.
   - May lead to **overfitting**.

- **Large \( \tau \) (e.g., 1.0)**:
   - The model considers distant points more evenly.
   - Produces a **smoother curve** that generalizes better to the overall trend.
   - May lead to **underfitting**.

---

#### Example of Prompting \( \tau \):
When you execute the script, you’ll see the following prompt in your terminal:
```bash
Enter value of tau: 0.8
```

---

## 🚀 How to Run

### 1. Clone the Repository
To get started, clone the repository using the following command:
```bash
git clone https://github.com/Amirreza-Bazmi/Machine-Learning-Projects.git
cd Machine-Learning-Projects/Simple and Weighted Linear Regression






