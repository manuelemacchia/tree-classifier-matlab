%%% Input
% The training set is split into two matrices, X_train and y_train.
% X_train contains a list of instances (vectors), each one with a constant
% number of features. y_train contains the class associated to each
% instance. Therefore, X_train and y_train have the same number of rows.
X_train = [
    30   0   10;
    30   0   70;
    30   1   20;
    30   1   80;
    60   0   40;
    60   0   60;
    60   1   50;
    60   1   60
];

% X_type gives information about the type of features contained in X_train.
% Type 1 features are categorical, while type 2 features are numerical.
% This information is useful when building the tree classifier, as it needs
% to handle splits for numerical features differently.
X_type = [2 1 2];

% We only consider binary classes, so we can store a class as a one or a
% zero. In our case, 0="no" and 1="yes".
y_train = [0 0 0 1 0 1 0 1]';

% The test set has the same structure of the training set, but is used to
% verify the accuracy of the tree modeled on the training set.
X_test = [
    30   0   70;
    60   1   55;
    30   0   13;
    90   1   10;
    40   1   80;
    30   0   10
];

% These are the classes that the tree should predict, that can be used to
% compute the accuracy (or any other metric) of the classifier.
y_test = [0 1 0 0 1 0]';

%%% Script
tree = tree_fit(X_train, X_type, y_train);
y_pred = tree_predict(tree, X_test, X_type);

disp("Test set:");
disp(X_test);

fprintf("Class predictions for the %d instances of the test set:\n", size(X_test, 1));
disp(y_pred');
