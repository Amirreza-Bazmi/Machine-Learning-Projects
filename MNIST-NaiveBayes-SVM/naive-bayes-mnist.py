from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

# Fetching data from the MNIST dataset
mnist = fetch_openml('mnist_784', parser= "auto")


# Extracting features and labels
X, y = mnist.data, mnist.target.astype(int)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=42)

# Initializing Naive Bayes classifier
laplace = 0.1
model = GaussianNB(var_smoothing= laplace)

# Training the model
model.fit(X_train, y_train)

# Predicting labels for test data
y_pred = model.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot accuracy
plt.figure(figsize=(8, 6))
plt.bar('Test', accuracy, color= 'orange')
plt.title(f'Accuracy of Naive Bayes Model based on laplace({laplace})')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig(f"Naive bayes based on laplace-{laplace}.jpg")
plt.show()


