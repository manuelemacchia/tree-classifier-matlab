# Tree classifier in MATLAB
## Overview
MATLAB implementation of a decision tree based on ID3 capable of binary classification and handling of continuous features.

## Usage
Open `classifier.m`, insert your training and test data, and run it. Data entry instructions are described in the script file. Datasets with both continuous and categorical features are supported.

## Structure
- `classifier.m` contains training and test data, as well as fit and predict function calls.

- `tree_fit.m` builds a decision tree classifier from the provided training set. It returns a tree in the form of a cell array.

- `tree_predict.m` predicts the classes of the test set. It returns a vector that contains the class predictions.
