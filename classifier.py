# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import metrics
import os
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

data = pd.read_csv('cancer.csv')
data['diagnosis'] = data['diagnosis'].replace({'M': 1, 'B': 0})
data.to_csv('cancer.csv', index=False)
del data

data_path = 'cancer.csv'
dataframe = pd.read_csv(data_path)
dataframe = dataframe[['diagnosis', 'perimeter_mean', 'radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean']]
dataframe['diagnosis_cat'] = dataframe['diagnosis'].astype('category').map({1: '1 (malignant)', 0: '0 (benign)'})

dataframe.head()
dataframe.info()

sns.catplot(x='radius_mean', y='diagnosis_cat', data=dataframe, order=['1 (malignant)', '0 (benign)'])
plt.show()

X, y = dataframe[['radius_mean']], dataframe[['diagnosis']]
model = linear_model.LinearRegression()
model.fit(X, y)
preds = model.predict(X)

sns.scatterplot(x='radius_mean', y='diagnosis', data=dataframe)
plt.plot(X, preds, color='r')
plt.legend(['Data', 'Linear Regression Fit'])
plt.show()

boundary = 15
sns.catplot(x='radius_mean', y='diagnosis_cat', data=dataframe, order=['1 (malignant)', '0 (benign)'])
plt.plot([boundary - 6, boundary + 4], [-.2, 1.2], 'g', linewidth=2)
plt.show()

def boundary_classifier(target_boundary, radius_mean_series):
    result = []
    for i in radius_mean_series:
        if i > target_boundary:
            result.append(1)
        else:
            result.append(0)
    return result

chosen_boundary = 15
y_pred = boundary_classifier(chosen_boundary, dataframe['radius_mean'])
dataframe['predicted'] = y_pred
y_true = dataframe['diagnosis']

sns.catplot(x='radius_mean', y='diagnosis_cat', hue='predicted', data=dataframe, order=['1 (malignant)', '0 (benign)'])
plt.plot([chosen_boundary, chosen_boundary], [-.2, 1.2], 'g', linewidth=2)
plt.show()

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy with boundary at {chosen_boundary}: {accuracy}")

train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=1)
print('Number of rows in training dataframe:', train_df.shape[0])
print('Number of rows in test dataframe:', test_df.shape[0])

X_feature_name = ['radius_mean']
y_target_name = 'diagnosis'
X_train = train_df[X_feature_name]
y_train = train_df[y_target_name]
logreg_model = linear_model.LogisticRegression()
logreg_model.fit(X_train, y_train)

X_test = test_df[X_feature_name]
y_test = test_df[y_target_name]
y_pred = logreg_model.predict(X_test)
test_df['predicted'] = y_pred

sns.catplot(x='radius_mean', y='diagnosis_cat', hue='predicted', data=test_df, order=['1 (malignant)', '0 (benign)'])
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with logistic regression on 'radius_mean': {accuracy}")

y_prob = logreg_model.predict_proba(X_test)
X_test_view = X_test[X_feature_name].values.squeeze()
plt.xlabel('radius_mean')
plt.ylabel('Predicted Probability of Malignancy')
sns.scatterplot(x=X_test_view, y=y_prob[:, 1], hue=y_test, palette=['purple', 'green'])
plt.show()

X_feature_name = ['texture_mean']
y_target_name = 'diagnosis'
train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=1)
X_train = train_df[X_feature_name]
y_train = train_df[y_target_name]
logreg_model = linear_model.LogisticRegression()
logreg_model.fit(X_train, y_train)
y_pred = logreg_model.predict(test_df[X_feature_name])
print(f"Accuracy with logistic regression on '{X_feature_name[0]}': {accuracy_score(test_df[y_target_name], y_pred)}")

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

(tn, fp), (fn, tp) = cnf_matrix
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print(f"Precision: {precision}")
print(f"Recall: {recall}")

from sklearn import tree
class_dt = tree.DecisionTreeClassifier(max_depth=3)
class_dt.fit(X_train, y_train)
plt.figure(figsize=(15, 10))
tree.plot_tree(class_dt, feature_names=X_features, class_names=['Benign', 'Malignant'], filled=True, fontsize=10)
plt.show()

y_pred_dt = class_dt.predict(X_test)
print("Decision Tree Performance:")
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred_dt))
print("Precision: ", metrics.precision_score(y_test, y_pred_dt))
print("Recall: ", metrics.recall_score(y_test, y_pred_dt))

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
results_df.plot(kind='bar', figsize=(14, 7))
plt.title("Classifier Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()
