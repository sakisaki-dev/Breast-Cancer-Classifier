# Breast Cancer Classifier

A walkthrough of classic machine learning classifiers that predict whether a breast tumor is malignant or benign from cell-nucleus measurements in the Breast Cancer Wisconsin (Diagnostic) dataset. Everything is in one script, `classifier.py`, using pandas, scikit-learn, matplotlib and seaborn.

## How it works

The script builds up from a single hand-set threshold to a comparison of nine scikit-learn models:

1. Recodes `diagnosis` (M = 1, B = 0) and plots `radius_mean` against diagnosis.
2. Fits a linear regression line to show why regression is a poor fit for a 0/1 label.
3. Classifies with a manual threshold (`radius_mean > 15`) and reports accuracy.
4. Splits the data 80/20 (`random_state=1`) and trains logistic regression on `radius_mean` alone, then `texture_mean` alone, and plots predicted malignancy probabilities.
5. Trains logistic regression on four features (`texture_mean`, `concavity_mean`, `area_mean`, `radius_mean`) and reports a confusion matrix, precision and recall.
6. Trains a depth-3 decision tree on the same features and plots the tree.
7. Compares accuracy, precision and recall for k-nearest neighbors, linear SVM, Gaussian process, decision tree, random forest, MLP, AdaBoost, Gaussian naive Bayes and QDA, and draws a bar chart.

## Data

The dataset is not included. Download the CSV version with named columns (`diagnosis`, `radius_mean`, `texture_mean`, ...) from [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) and save it as `cancer.csv` in the repo root. The original source is the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic); the raw UCI file has no header row and will not work as-is.

Note: the script overwrites `cancer.csv` with the recoded diagnosis column. Keep a copy of the original.

## Setup and run

```bash
git clone https://github.com/sakisaki-dev/Breast-Cancer-Classifier.git
cd Breast-Cancer-Classifier
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt "pandas<3"
python classifier.py
```

Each plot opens in a window; close it to continue. Metrics print to the terminal.

`pandas<3` is needed because the diagnosis recoding uses `replace(..., inplace=True)` on a column, which pandas 3 no longer applies, and the script then fails at the first model fit.

## Limitations

* Models are evaluated on a single 80/20 split with no cross-validation, and features are not scaled.
* Random forest and MLP have no fixed random seed, so their scores change between runs.
* The script is a learning exercise, not a clinical tool.
