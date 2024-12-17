import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def simple_linear_regression(feature, label):
    # Adding bias
    plot_line = True
    X = np.column_stack((np.ones(len(feature)), feature))
    # plot_line = False
    # X = np.column_stack((np.ones(len(feature)), feature,feature**2,feature**3,feature**4))
    # Theta
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(label)

    # Linear equation
    print("Linear Equation: y = {:.2f} + {:.2f} * x".format(theta[0], theta[1]))

    # Plotting training data and the regression line
    plt.scatter(feature, label, color= "red", label= "Data")
    if plot_line:
        plt.plot(feature, X.dot(theta), color= "black", linewidth= 3, label= "Line")
    else:
        plt.scatter(feature, X.dot(theta), color= "black", linewidth= 3, label= "Line")

    plt.xlabel("Feature")
    plt.ylabel("Label")
    plt.title("Simple Linear Regression: y = {:.2f} + {:.2f} * x".format(theta[0], theta[1]))
    plt.legend()
    plt.grid(True)

    plt.savefig("Simple Linear Regression.png")
    plt.show()

def kernel(x1, x2, tau):
    return np.exp(-np.sum((x1 - x2) ** 2) / (2 * tau ** 2))

def weighted_linear_regression(feature, label, tau= 0.8):
    # Sorting the data
    sorted_indices = np.argsort(feature)
    feature_sorted = feature[sorted_indices]
    label_sorted = label[sorted_indices]

    # Split the sorted data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(feature_sorted, label_sorted, train_size=0.8, random_state=42)
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    x_test = np.column_stack((np.ones(len(x_test)), x_test))

    # Thetas and predictions
    thetas = []
    predictions = []

    # Calculate theta and prediction for each test data point
    for i in range(len(y_test)):
        # Calculate W for any test data point
        W = np.zeros((len(y_train), len(y_train)))
        for j in range(len(y_train)):
            W[j, j] = kernel(x_test[i, 1], x_train[j, 1], tau)

        # Calculate theta for any test data point
        theta = np.linalg.inv(x_train.T.dot(W).dot(x_train)).dot(x_train.T).dot(W).dot(y_train)

        # Prediction for any test data point
        prediction = x_test[i].dot(theta)

        # Store theta and prediction
        thetas.append(theta)
        predictions.append(prediction)

    # Plotting training data and test data
    plt.scatter(x_train[:, 1], y_train, label= "Training Data", color= "black")
    plt.scatter(x_test[:, 1], y_test, label= "Test Data", color= "red")

    # Sorting the predicted points
    sorted_predictions_indices = np.argsort(x_test[:, 1])
    sorted_x_test = x_test[sorted_predictions_indices]
    sorted_predictions = np.array(predictions)[sorted_predictions_indices]

    # Plotting sorted predictions
    plt.plot(sorted_x_test[:, 1], sorted_predictions, label= "Predictions", linewidth= 4, color= "blue")

    plt.xlabel("Features")
    plt.ylabel("Labels")
    plt.title(fr"Weighted Linear Regression based on $\tau$= {tau}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Weighted Linear Regression based on tau({tau}).png")
    plt.show()

if __name__ == "__main__":
    # Reading features and labels
    label = np.loadtxt("labels.txt")
    feature = np.loadtxt("features.txt")

    # Simple linear regression
    simple_linear_regression(feature, label)

    # Weighted linear regression
    # tau = 0.8
    tau = eval(input("Enter value of tau: "))
    weighted_linear_regression(feature, label, tau)

    print("Done.")