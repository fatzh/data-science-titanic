# -*- coding: utf-8 -*-
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(estimator, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
    """
    source http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title("Learning Curves")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Error")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    # compute error
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores
    train_errors_mean = np.mean(train_errors, axis=1)
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_mean = np.mean(test_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_errors_mean - train_errors_std,
                     train_errors_mean + train_errors_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_errors_mean - test_errors_std,
                     test_errors_mean + test_errors_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_errors_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_errors_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_features_curves(estimator, X, y, feature_list, split=10):
    """
    Split the features using the split parameters.

    To make sense, the feature_list must be ordered in order of importance.
    """
    train_scores, test_scores = [], []
    plt.figure()
    plt.title("Features Curves")
    plt.xlabel("Number of features")
    plt.ylabel("Error")
    # reduce features
    for f in range(split, len(feature_list), split):
        X_sub = X[feature_list[0:f]]
        # create a test set
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_sub, y, test_size=0.2, random_state=0)
        estimator.fit(X_train, y_train)
        train_scores.append(estimator.score(X_train, y_train))
        test_scores.append(estimator.score(X_test, y_test))

    import ipdb; ipdb.set_trace() ## BREAKPOINT
    # compute error
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores
    train_errors_mean = np.mean(train_errors, axis=1)
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_mean = np.mean(test_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_errors_mean - train_errors_std,
                     train_errors_mean + train_errors_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_errors_mean - test_errors_std,
                     test_errors_mean + test_errors_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_errors_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_errors_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



