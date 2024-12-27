import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def one_vs_one_svm(X_train, X_test, y_train, y_test, kernels, C):
    # Model
    for c in C:
        for kernel in kernels:
            # Create a model
            model = SVC(kernel=kernel, C=c, decision_function_shape="ovo")

            # Training model
            print("Start training...")
            model.fit(X_train, y_train)

            # Prediction
            print("Predict...")
            y_pred = model.predict(X_test)

            # Calculate Accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Number of support vectors
            number_sv = np.sum(model.n_support_)

            # Print
            print(f"Parameters: C={c}, Kernel={kernel}")
            print("One-vs-One:")
            print("Accuracy:", accuracy)
            print("Number of support vectors:", number_sv)
        print('\n' + '--' * 40 + '\n')

def one_vs_all_svm(X_train, X_test, y_train, y_test, kernels, C):
    # Model
    for c in C:
        for kernel in kernels:
            # Create a model
            model = SVC(kernel=kernel, C=c, decision_function_shape="ovr")

            # Training model
            print("Start training...")
            model.fit(X_train, y_train)

            # Prediction
            print("Predict...")
            y_pred = model.predict(X_test)

            # Calculate Accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Number of support vectors
            number_sv = np.sum(model.n_support_)

            # Print
            print(f"Parameters: C={c}, Kernel={kernel}")
            print("One-vs-All:")
            print("Accuracy:", accuracy)
            print("Number of support vectors:", number_sv)
        print('\n' + '--' * 40 + '\n')

if __name__ == "__main__":
    # Fetching data from the MNIST dataset
    mnist = fetch_openml('mnist_784', parser="auto")

    # Extracting features and labels
    X, y = mnist.data, mnist.target.astype(int)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1/7, random_state=42)

    # Kernels type
    kernels = ["linear", "rbf", "poly"]

    # Parameter C
    C_values = [0.05, 1, 7]

    # One-vs-one SVM
    one_vs_one_svm(X_train, X_test, y_train, y_test, kernels, C_values)

    # One-vs-all SVM
    one_vs_all_svm(X_train, X_test, y_train, y_test, kernels, C_values)




