import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def linear_regresion(dataset, features):
    # x = Features & y = Price
    # Split the features string and remove leading/trailing spaces
    feature_list = [feature.strip() for feature in features.split(',')]

    # Ensure all features exist in the dataset
    if not all(feature in dataset.columns for feature in feature_list):
        print("Some features are not present in the dataset.")
        return

    x = dataset[feature_list]
    y = dataset["Price"]

    # Create a linear regression model and Training the model
    model = LinearRegression().fit(x, y)

    # Prediction
    y_predited = model.predict(x)

    # R-squared score
    r_squared = model.score(x, y)
    print(f"R-squared Score: {r_squared}")

    if len(feature_list) == 1:
        # Ploting
        plt.scatter(x, y, color="gray", label="Data")
        plt.plot(x, y_predited, color="red", label=f"Linear Regression: {feature_list[0]}", linewidth=6)
        plt.xlabel(f"Input: {feature_list[0]}")
        plt.ylabel("Price")
        plt.title(f"Linear Regression: Price vs {feature_list[0]}  R^2 = %.4f" % r_squared)
        plt.legend()
        plt.savefig(f"Linear Regression-{feature_list[0]}.png")
        plt.show()


if __name__ == "__main__":
    # Reading file
    data = pd.read_csv("data.csv")

    # Features
    features = input("Enter the features "
                     "\n(if to be more than one feature : example1, example2, example3, ...)"
                     "\n(if to be one feature : example1):\n")
    linear_regresion(data, features)