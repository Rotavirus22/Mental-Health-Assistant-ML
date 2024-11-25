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

    auc = np.trapz(tpr, fpr)
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


# KNN Classifier
class KNeighborsClassifierManual:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X.astype(float)
        self.y_train = y

    def predict(self, X):
        X = X.astype(float)
        predictions = []
        for x_test in X:
            distances = np.linalg.norm(self.X_train - x_test, axis=1)
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            predictions.append(np.bincount(k_nearest_labels).argmax())
        return np.array(predictions)


# Decision Tree Classifier
class DecisionTreeClassifierManual:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1:
            return y[0]

        if X.shape[1] == 0 or X.shape[0] == 0:
            return np.bincount(y).argmax() if len(y) > 0 else 0

        if self.max_depth is not None and depth >= self.max_depth:
            return np.bincount(y).argmax() if len(y) > 0 else 0

        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            return np.bincount(y).argmax() if len(y) > 0 else 0

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_tree, right_tree)

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_index, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold

    def _information_gain(self, X, y, feature_index, threshold):
        parent_entropy = self._entropy(y)
        left_indices = X[:, feature_index] < threshold
        right_indices = X[:, feature_index] >= threshold

        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])

        left_weight = len(y[left_indices]) / len(y)
        right_weight = len(y[right_indices]) / len(y)

        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        return -np.sum(proportions * np.log2(proportions + 1e-6))

    def _predict_single(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature_index, threshold, left_tree, right_tree = tree
        if x[feature_index] < threshold:
            return self._predict_single(x, left_tree)
        else:
            return self._predict_single(x, right_tree)

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])


# Random Forest Classifier
class RandomForestClassifierManual:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifierManual(max_depth=self.max_depth)
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(tree_pred).most_common(1)[0][0] for tree_pred in tree_preds.T])


# Gaussian Naive Bayes
class GaussianNBManual:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.params = {}
        for c in self.classes:
            X_c = X[y == c]
            self.params[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0),
                'prior': len(X_c) / len(y)
            }

    def predict(self, X):
        probs = []
        for x in X:
            class_probs = {}
            for c in self.classes:
                prior = np.log(self.params[c]['prior'])
                likelihood = np.sum(np.log(self.gaussian_pdf(x, self.params[c]['mean'], self.params[c]['var'])))
                class_probs[c] = prior + likelihood
            probs.append(max(class_probs, key=class_probs.get))
        return np.array(probs)

    def gaussian_pdf(self, x, mean, var):
        coefficient = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-(x - mean) ** 2 / (2 * var))
        return coefficient * exponent


# Stacking Classifier
class StackingClassifierManual:
    def __init__(self, classifiers, meta_classifier):
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier

    def fit(self, X, y):
        base_preds = np.zeros((X.shape[0], len(self.classifiers)))
        for i, clf in enumerate(self.classifiers):
            clf.fit(X, y)
            base_preds[:, i] = clf.predict(X)
        self.meta_classifier.fit(base_preds, y)

    def predict(self, X):
        base_preds = np.zeros((X.shape[0], len(self.classifiers)))
        for i, clf in enumerate(self.classifiers):
            base_preds[:, i] = clf.predict(X)
        return self.meta_classifier.predict(base_preds)


# Initialize classifiers
clf1 = KNeighborsClassifierManual(n_neighbors=3)
clf2 = DecisionTreeClassifierManual(max_depth=5)
clf3 = RandomForestClassifierManual(n_estimators=10, max_depth=5)
clf4 = GaussianNBManual()
meta_clf = LogisticRegressionManual(learning_rate=0.01, num_iterations=1000)

# Create Stacking Classifier
stack = StackingClassifierManual(classifiers=[clf1, clf2, clf3, clf4], meta_classifier=meta_clf)

# Train the stacking model
stack.fit(X_train, y_train)

# Make predictions on the test set
y_pred_class = stack.predict(X_test)

# Evaluate the model
accuracy = evalClassModel(y_test, y_pred_class, y_pred_class)
print(f'Accuracy of the Stacking Model: {accuracy}')

# Save the model to a file
with open("manual_model.pkl", "wb") as file:
    pickle.dump(stack, file)

print("Model trained and saved as manual_model.pkl")
