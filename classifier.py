# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd
from sklearn import metrics
import os
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

## Introduction to Breast Cancer Biopsy Classification
# In this project, we will build a classifier to determine whether a breast cancer tumor sample is malignant (dangerous) or benign (not dangerous).
# The dataset contains various metrics recorded from tumor biopsies, including: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

# Download and prepare the data

data = pd.read_csv('cancer.csv')
data['diagnosis'].replace({'M':1, 'B':0}, inplace = True)
data.to_csv('cancer.csv')
del data

## Loading the annotated dataset
data_path = 'cancer.csv'

# Read the data into a pandas DataFrame
dataframe = pd.read_csv(data_path)

# Select relevant columns and create a categorical diagnosis column for visualization
dataframe = dataframe[['diagnosis', 'perimeter_mean', 'radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean']]
dataframe['diagnosis_cat'] = dataframe['diagnosis'].astype('category').map({1: '1 (malignant)', 0: '0 (benign)'})

# ---
# # Exploring our data
# ---

## Looking at our dataset
# Display the first five rows to understand its structure.
dataframe.head()

# Column meanings:
# * `diagnosis`: Malignant (1) or benign (0).
# * `perimeter_mean`: Average perimeter of cells.
# * `radius_mean`: Average radius of cells.
# * `texture_mean`: Average texture of cells.
# * `area_mean`: Average area of cells.
# * `smoothness_mean`: Average smoothness of cells.
# * `concavity_mean`: Average concavity of cells.
# * `symmetry_mean`: Average symmetry of cells.

# Get information about the data types of each column
dataframe.info()

## Visualizing our dataset
# We can graph features to see how they vary between diagnoses.
sns.catplot(x='radius_mean', y='diagnosis_cat', data=dataframe, order=['1 (malignant)', '0 (benign)'])
plt.show()


# ---
# # Predicting Diagnosis
# Let's start by predicting a diagnosis using a single feature: radius mean.
# ---

## Approach 1: Can we use linear regression to classify these cells?
# Fit and visualize a linear regression model
X, y = dataframe[['radius_mean']], dataframe[['diagnosis']]

model = linear_model.LinearRegression()
model.fit(X, y)
preds = model.predict(X)

sns.scatterplot(x='radius_mean', y='diagnosis', data=dataframe)
plt.plot(X, preds, color='r')
plt.legend(['Data', 'Linear Regression Fit'])
plt.show()

## Approach 2: Simple Boundary Classifier
# Since the target variable is categorical, a classifier is more appropriate.
# The simplest approach is to draw a boundary.

boundary = 15

sns.catplot(x='radius_mean', y='diagnosis_cat', data=dataframe, order=['1 (malignant)', '0 (benign)'])
plt.plot([boundary - 6, boundary + 4], [-.2, 1.2], 'g', linewidth=2)
plt.show()

### Building the boundary classifier
# This function classifies data points based on whether they are above or below a given boundary.
def boundary_classifier(target_boundary, radius_mean_series):
    result = []
    for i in radius_mean_series:
        if i > target_boundary:
            result.append(1)
        else:
            result.append(0)
    return result

# Choose a boundary and run the classifier
chosen_boundary = 15

y_pred = boundary_classifier(chosen_boundary, dataframe['radius_mean'])
dataframe['predicted'] = y_pred

y_true = dataframe['diagnosis']

sns.catplot(x='radius_mean', y='diagnosis_cat', hue='predicted', data=dataframe, order=['1 (malignant)', '0 (benign)'])
plt.plot([chosen_boundary, chosen_boundary], [-.2, 1.2], 'g', linewidth=2)
plt.show()

# Calculate the accuracy of the boundary classifier
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy with boundary at {chosen_boundary}: {accuracy}")


# ---
# ## Approach 3: Logistic Regression - using machine learning to determine the optimal boundary
# ---

# Split data into training and testing sets
train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=1)

print('Number of rows in training dataframe:', train_df.shape[0])
print('Number of rows in test dataframe:', test_df.shape[0])


### Single Variable Logistic Regression
# Set input feature (X) and output variable (y)
X_feature_name = ['radius_mean']
y_target_name = 'diagnosis'

X_train = train_df[X_feature_name]
y_train = train_df[y_target_name]

# Initialize and train the logistic regression model
logreg_model = linear_model.LogisticRegression()
logreg_model.fit(X_train, y_train)

### Testing our model
X_test = test_df[X_feature_name]
y_test = test_df[y_target_name]

# Make predictions on the test set
y_pred = logreg_model.predict(X_test)

# Visualize the results
test_df['predicted'] = y_pred
sns.catplot(x='radius_mean', y='diagnosis_cat', hue='predicted', data=test_df, order=['1 (malignant)', '0 (benign)'])
plt.show()

# Calculate the accuracy of the logistic regression model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with logistic regression on 'radius_mean': {accuracy}")


### What is logistic regression doing? It's giving 'soft' predictions!
# Let's visualize the probabilities for `X_test`
y_prob = logreg_model.predict_proba(X_test)
X_test_view = X_test[X_feature_name].values.squeeze()
plt.xlabel('radius_mean')
plt.ylabel('Predicted Probability of Malignancy')
sns.scatterplot(x=X_test_view, y=y_prob[:, 1], hue=y_test, palette=['purple', 'green'])
plt.show()

# The Y-axis is the probability of being 'malignant' and the X-axis is the radius mean.
# The colors show the true diagnosis.

# ---
# # Approach 4: Multiple Feature Logistic Regression
# ---

### Experimenting with Single-Variable Logistic Regression
# Try another feature to see how well it predicts diagnosis
X_feature_name = ['texture_mean']
y_target_name = 'diagnosis'

train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=1)

X_train = train_df[X_feature_name]
y_train = train_df[y_target_name]

logreg_model = linear_model.LogisticRegression()
logreg_model.fit(X_train, y_train)

y_pred = logreg_model.predict(test_df[X_feature_name])
print(f"Accuracy with logistic regression on '{X_feature_name[0]}': {accuracy_score(test_df[y_target_name], y_pred)}")

### Logistic Regression with Multiple Features
# Using multiple features together can improve prediction accuracy.
X_features = ["texture_mean", "concavity_mean", "area_mean", "radius_mean"]
y_target = 'diagnosis'

train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=1)

X_train = train_df[X_features]
y_train = train_df[y_target]
X_test = test_df[X_features]
y_test = test_df[y_target]

logreg_model = linear_model.LogisticRegression()
logreg_model.fit(X_train, y_train)

y_pred = logreg_model.predict(X_test)

print(f"Accuracy with multiple features: {accuracy_score(y_test, y_pred)}")


# ---
# # What makes a separation good?
# ---
# Errors come in two kinds:
# **False positives:** The model predicts malignant, but it's actually benign.
# **False negatives:** The model predicts benign, but it's actually malignant.

### Confusion Matrices
# We can visualize the number of correct vs. incorrect predictions.
# * **TP (True Positive)**: Predicted positive (malignant), and it’s true.
# * **TN (True Negative)**: Predicted negative (benign), and it’s true.
# * **FP (False Positive)**: Predicted positive, but it’s false.
# * **FN (False Negative)**: Predicted negative, but it’s false.

# Create and visualize the Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual diagnosis')
plt.xlabel('Predicted diagnosis')
plt.show()

### Choosing a Metric
# We might measure success in different ways:
# **Accuracy:** What portion of our predictions are right? `(TP + TN) / Total`
# **Precision:** What portion of our positive predictions are actually positive? `TP / (TP + FP)`
# **Recall:** What portion of the actual positives did we identify? `TP / (TP + FN)`

(tn, fp), (fn, tp) = cnf_matrix
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

precision = tp / (tp + fp)
recall = tp / (tp + fn)
print(f"Precision: {precision}")
print(f"Recall: {recall}")


# ---
# # Optional: Decision Trees Walkthrough
# ---
from sklearn import tree

# Create and train the decision tree model
class_dt = tree.DecisionTreeClassifier(max_depth=3)
class_dt.fit(X_train, y_train)

# Visualize the tree
plt.figure(figsize=(15, 10))
tree.plot_tree(class_dt, feature_names=X_features, class_names=['Benign', 'Malignant'], filled=True, fontsize=10)
plt.show()

# Make predictions and evaluate performance
y_pred_dt = class_dt.predict(X_test)
print("Decision Tree Performance:")
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred_dt))
print("Precision: ", metrics.precision_score(y_test, y_pred_dt))
print("Recall: ", metrics.recall_score(y_test, y_pred_dt))


# ---
# # Advanced (Optional): Choosing a Classifier
# ---
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

classifier_names = [
    "K-Neighbors", "Linear SVM", "Gaussian Process", "Decision Tree",
    "Random Forest", "MLP", "AdaBoost", "Gaussian NB", "QDA"
]

results = []

for model in classifiers:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    pre = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    results.append([acc, pre, rec])

results_df = pd.DataFrame(results, index=classifier_names, columns=['Accuracy', 'Precision', 'Recall'])
print(results_df)

# Plotting the results for comparison
results_df.plot(kind='bar', figsize=(14, 7))
plt.title("Classifier Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()