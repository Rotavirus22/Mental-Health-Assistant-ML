import pandas as pd
import numpy as np
import pickle
from collections import Counter

# Load the data
data = pd.read_csv("mental_health.csv")

# Define feature columns and target variable
feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
X = data[feature_cols].to_numpy()
y = data['treatment'].to_numpy()

# Manual train-test split
def manual_train_test_split(X, y, test_size=0.30, random_state=0):
    np.random.seed(random_state)  # for reproducibility
    indices = np.arange(X.shape[0])  # array of indices
    np.random.shuffle(indices)  # shuffle indices randomly

    split_index = int(X.shape[0] * (1 - test_size))  # calculate the split index

    # Use array indexing instead of .iloc
    X_train = X[indices[:split_index]]  # training data
    X_test = X[indices[split_index:]]   # testing data
    y_train = y[indices[:split_index]]  # training labels
    y_test = y[indices[split_index:]]   # testing labels

    return X_train, X_test, y_train, y_test

# Perform the manual train-test split
X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=0.30, random_state=0)

# Evaluate Classification Model
def evalClassModel(y_test, y_pred_class, y_pred_prob=None, plot=False):
    accuracy = np.mean(y_test == y_pred_class)
    print('Accuracy:', accuracy)

    # Confusion matrix
    TP = np.sum((y_test == 1) & (y_pred_class == 1))
    TN = np.sum((y_test == 0) & (y_pred_class == 0))
    FP = np.sum((y_test == 0) & (y_pred_class == 1))
    FN = np.sum((y_test == 1) & (y_pred_class == 0))

    confusion = np.array([[TN, FP], [FN, TP]])

    print('Classification Accuracy:', accuracy)
    classification_error = 1 - accuracy
    print('Classification Error:', classification_error)

    # Precision
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    print('Precision:', precision)

    # AUC Score
    roc_auc = compute_auc(y_test, y_pred_prob)
    print('AUC Score:', roc_auc)

    return accuracy


def compute_auc(y_true, y_prob):
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]

    fpr = []
    tpr = []
    thresholds = np.unique(y_prob_sorted)
    total_positives = np.sum(y_true)
    total_negatives = len(y_true) - total_positives

    for threshold in thresholds:
        y_pred_at_threshold = (y_prob_sorted >= threshold).astype(int)
        TP = np.sum((y_true_sorted == 1) & (y_pred_at_threshold == 1))
        FP = np.sum((y_true_sorted == 0) & (y_pred_at_threshold == 1))
        tpr.append(TP / total_positives)
        fpr.append(FP / total_negatives)

    auc = np.trapezoid(tpr, fpr)
    return auc


# Logistic Regression
class LogisticRegressionManual:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_pred_class)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# defining the logistic regression function
logreg = LogisticRegressionManual(learning_rate=0.01, num_iterations=1000)

# Train the stacking model
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_class = logreg.predict(X_test)

# Evaluate the model
accuracy = evalClassModel(y_test, y_pred_class, y_pred_class)
print(f'Accuracy of the Model: {accuracy}')

# Save the model to a file
with open("manual_model.pkl", "wb") as file:
    pickle.dump(logreg, file)

print("Model trained and saved as manual_model.pkl")
