# Handwritten Digit Recognition with Naive Bayes and SVM

This project implements **Naive Bayes** and **Support Vector Machines (SVM)** for handwritten digit classification using the MNIST dataset. Both algorithms are evaluated on their performance with varying parameters, showcasing their strengths and trade-offs in accuracy and computational efficiency.

---

## 📂 Files in the Repository

1. **`naive-bayes-mnist.py`**:
   - Implements Naive Bayes with Laplace smoothing for digit classification.
   - Visualizes the effect of different Laplace parameters on accuracy.

2. **`svm-mnist.py`**:
   - Implements SVM for digit classification using two approaches:
     - **One-vs-One (OvO)**
     - **One-vs-All (OvA)**
   - Explores different kernels (`linear`, `rbf`, `poly`) and penalty parameters (C).

3. **Visualization Outputs**:
   - Bar charts for Naive Bayes accuracy.
   - Results and metrics for SVM approaches.

---

## 📊 Overview of the Project

### Objective
To classify handwritten digits (0-9) using two traditional machine learning algorithms and compare their performance on the MNIST dataset.

### Dataset
- **MNIST**: A benchmark dataset of handwritten digits with 60,000 training samples and 10,000 testing samples.

### Methodology

#### 1. Naive Bayes
- Utilizes **Gaussian Naive Bayes** with Laplace smoothing.
- Evaluates accuracy for different Laplace values (`0.1`, `10`, `1000`).
- **Key Finding**:
  - Best accuracy achieved with **Laplace = 0.1** (Accuracy: 80.6%).

#### 2. Support Vector Machines (SVM)
- Implements two multi-class classification strategies:
  - **One-vs-One (OvO)**: Trains one classifier for every pair of classes.
  - **One-vs-All (OvA)**: Trains one classifier per class against all others.
- Explores three kernel types:
  - **Linear**
  - **Radial Basis Function (RBF)**
  - **Polynomial**
- **Key Observations**:
  - Best accuracy achieved with **C=7**, **RBF kernel**, and **OvA** strategy (Accuracy: 96.6%).
  - OvA strategy is faster compared to OvO while achieving similar results.

---

## 🚀 How to Run

### 1. Clone the Repository
To get started, clone this repository using the following command:
```bash
git clone https://github.com/Amirreza-Bazmi/Machine-Learning-Projects.git
cd Machine-Learning-Projects/MNIST-NaiveBayes-SVM
```

### 2. Run Naive Bayes
Execute the Naive Bayes implementation:
```bash
python naive-bayes-mnist.py
```

### 3. Run SVM
Execute the SVM implementation:
```bash
python svm-mnist.py
```

---

## 🛠 Customization

- **Adjust Parameters**:
  - Modify Laplace smoothing in `naive-bayes-mnist.py`.
  - Experiment with different `C` values and kernels in `svm-mnist.py`.

- **Change Dataset**:
  - Replace MNIST with a custom dataset by updating the data loading section.

---

## 📬 Contact
For questions or suggestions, feel free to reach out:
- **Email**: [bazmi.ce@gmail.com](mailto:bazmi.ce@gmail.com)
- **GitHub**: [Amirreza-Bazmi](https://github.com/Amirreza-Bazmi)

---

Happy Coding! 🚀
