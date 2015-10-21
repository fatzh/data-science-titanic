# -*- coding: utf-8 -*-
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curves(train_scores, test_scores, title="Learning Curves", ylim=None, with_variance=False, x_label="Training samples"):
    """
    source http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(x_label)
    plt.ylabel("Error")
    # compute error
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores
    train_errors_mean = np.mean(train_errors, axis=0)
    train_errors_std = np.std(train_errors, axis=0)
    test_errors_mean = np.mean(test_errors, axis=0)
    test_errors_std = np.std(train_errors, axis=0)
    print "Small test error mean %f obtained for %f" % (test_errors_mean.min(), test_errors_mean.idxmin())
    # that's the change along the x axis.. can be train size, components, etc..
    train_x_axis = train_errors.columns.tolist()
    plt.grid()

    if with_variance:
        plt.fill_between(train_x_axis, train_errors_mean - train_errors_std,
                         train_errors_mean + train_errors_std.tolist(), alpha=0.1,
                         color="r")
        plt.fill_between(train_x_axis, test_errors_mean - test_errors_std,
                         test_errors_mean + test_errors_std, alpha=0.1, color="g")

    plt.plot(train_x_axis, train_errors_mean, 'o-', color="r",
             label="Training error")
    plt.plot(train_x_axis, test_errors_mean, 'o-', color="g",
             label="Test error")

    plt.legend(loc="best")
    return plt



def rf_accuracy_by_sample_size(model, X, y, n_iter=3, sample_sizes=np.linspace(.1, 1, 10)):
    """
    Create shufflesplits of the train set of different sizes and compare the result with
    the test set.

    Each sample size is ran 3 times.

    SAve scores in a DataFrame
    """
    print "Calculating accuracy score for different sample sizes."
    train_scores = pd.DataFrame()
    test_scores = pd.DataFrame()
    for size in sample_sizes:
        print "Size = ", size
        train_score = []
        test_score = []
        # for the 100%, we can't split
        if size == 1.0:
            for i in range(n_iter):
                model.fit(X, y)
                train_score.append(model.score(X, y))
                test_score.append(model.oob_score_)
        else:
            sss = cross_validation.StratifiedShuffleSplit(y, n_iter=n_iter, train_size=size, random_state=123)
            for train_index, test_index in sss:
                model.fit(X.iloc[train_index], y.iloc[train_index])
                train_score.append(model.score(X.iloc[train_index], y.iloc[train_index]))
                test_score.append(model.oob_score_)
        train_scores[sss.n_train] = train_score
        test_scores[sss.n_train] = test_score
    return train_scores, test_scores

def rf_accuracy_by_pca(model, X, y, variance=np.linspace(.1, .999, 20), to_scale=None):
    """
    reduces the features dimensions using PCA and different number of components.
    """
    print "Calculating accuracy for different number of components"
    train_scores = pd.DataFrame()
    test_scores = pd.DataFrame()
    # need to scale first
    scaler = StandardScaler(copy=False)
    scaler.fit(X[to_scale])
    X.loc[:, to_scale] = scaler.transform(X[to_scale])
    for v in variance:
        print "Variance : ", v
        pca = PCA(n_components=v)
        pca.fit(X)
        colname = round(v*100, 2)
        X_reduced = pca.transform(X)
        model.fit(X_reduced, y)
        train_scores[colname] = [model.score(X_reduced, y)]
        test_scores[colname] = [model.oob_score_]
    return train_scores, test_scores

def rf_accuracy_by_feature_importance(model, X, y, thresholds=np.linspace(.0005, 0.13, 100)):
    """
    reduces the features dimensions based on feature importance
    """
    print "Calculating accuracy for different number of features"
    train_scores = pd.DataFrame()
    test_scores = pd.DataFrame()
    for t in thresholds:
        print "Treshold : ", t
        model.fit(X, y)
        colname = round(t*100, 2)
        X_reduced = model.transform(X, t)
        model.fit(X_reduced, y)
        train_scores[colname] = [model.score(X_reduced, y)]
        test_scores[colname] = [model.oob_score_]
    return train_scores, test_scores


def rf_accuracy_by_n_estimator(model, X, y, n_estimator=np.arange(10, 200)):
    """
    Returns the accuracy score for different number of trees
    """
    print "Calculating accuracy for different number of trees"
    train_scores = pd.DataFrame()
    test_scores = pd.DataFrame()
    # save the initial params
    n = model.n_estimators
    for t in n_estimator:
        print "Numer of trees : %d" % (t)
        model.set_params(n_estimators=t)
        model.fit(X, y)
        colname = t
        train_scores[colname] = [model.score(X, y)]
        test_scores[colname] = [model.oob_score_]
    # reset n_estimators
    model.set_params(n_estimators=n)
    return train_scores, test_scores

def rf_accuracy_by_max_depth(model, X, y, max_depths=np.arange(1, 50)):
    """
    Returns the accuracy score for different max depth
    """
    print "Calculating accuracy for different number max depth"
    train_scores = pd.DataFrame()
    test_scores = pd.DataFrame()
    for md in max_depths:
        print "Max depth : %d" % (md)
        model.set_params(max_depth=md)
        model.fit(X, y)
        colname = md
        train_scores[colname] = [model.score(X, y)]
        test_scores[colname] = [model.oob_score_]
    return train_scores, test_scores

def plot_learning_curves_cv(estimator, X, y, title="Learning Curves", ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Plot learning curves using cross validation
    source http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Observations")
    plt.ylabel("Error")
    # compute error
    train_sizes, train_scores, test_scores = learning_curve(
                estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores
    train_errors_mean = np.mean(train_errors, axis=1)
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_mean = np.mean(test_errors, axis=1)
    test_errors_std = np.std(train_errors, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_errors_mean - train_errors_std,
                     train_errors_mean + train_errors_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_errors_mean - test_errors_std,
                     test_errors_mean + test_errors_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_errors_mean, 'o-', color="r",
             label="Training error")
    plt.plot(train_sizes, test_errors_mean, 'o-', color="g",
             label="Cross validation error")

    plt.legend(loc="best")
    return plt

def f1_average(y_true, y_pred):
    """
    returns average of f1 score for both classes.
    """
    f1_survived = f1_score(y_true, y_pred, pos_label=1, average="binary")
    f1_died = f1_score(y_true, y_pred, pos_label=0, average="binary")
    return np.mean([f1_survived, f1_died])


def f1_scores_by_sample_size(model, X_train, y_train, X_test, y_test, n_iter=3, sample_sizes=np.linspace(.1, 1, 10)):
    """
    Create shufflesplits of the train set of different sizes and compare the result with
    the test set.

    Each sample size is ran 3 times.

    Save scores in a DataFrame
    """
    print "Calculating F1 score for different sample sizes."
    train_scores = pd.DataFrame()
    test_scores = pd.DataFrame()
    for size in sample_sizes:
        print "Size = ", size
        f1_train_score = []
        f1_test_score = []
        # for the 100%, we can't split
        if size == 1.0:
            for i in range(n_iter):
                model.fit(X_train, y_train)
                f1_train_score.append(f1_average(y_train, model.predict(X_train)))
                f1_test_score.append(f1_average(y_test, model.predict(X_test)))
        else:
            sss = cross_validation.StratifiedShuffleSplit(y_train, n_iter=n_iter, train_size=size, random_state=123)
            for train_index, test_index in sss:
                model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
                f1_train_score.append(f1_average(y_train.iloc[train_index], model.predict(X_train.iloc[train_index])))
                f1_test_score.append(f1_average(y_test, model.predict(X_test)))
        train_scores[sss.n_train] = f1_train_score
        test_scores[sss.n_train] = f1_test_score
    return train_scores, test_scores


def f1_scores_by_pca(model, X_train, y_train, X_test, y_test, variance=np.linspace(.1, .999, 20), to_scale=None):
    """
    reduces the features dimensions using PCA and different number of components.
    """
    print "Calculating F1 score for different number of components"
    train_scores = pd.DataFrame()
    test_scores = pd.DataFrame()
    for v in variance:
        print "Variance : ", v
        pca = PCA(n_components=v)
        pca.fit(X_train)
        colname = round(v*100, 2)
        X_train_reduced = pca.transform(X_train)
        model.fit(X_train_reduced, y_train)
        X_test_reduced = pca.transform(X_test)
        train_scores[colname] = [f1_average(y_train, model.predict(X_train_reduced))]
        test_scores[colname] = [f1_average(y_test, model.predict(X_test_reduced))]
    return train_scores, test_scores
