# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from helpers import helpers


def main():
    # widen pd output for debugging
    pd.set_option('display.width', 1000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file to parse (CSV)")
    parser.add_argument("--learningcurves", help="Calculate and plot learning curves (takes time...)", action="store_true")
    parser.add_argument("--featurescurves", help="Calcualte and plot errors by features (takes time...)", action="store_true")
    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        sys.exit()
    input_file = args.input

    do_learning_curves = False
    if args.learningcurves:
        do_learning_curves = True

    do_features_curves = False
    if args.featurescurves:
        do_features_curves = True

    # load data
    df = pd.read_csv(input_file, sep=',')


    # split X and y
    y = df['Survived']
    X = df.drop('Survived', 1)

    # store the feature list
    features_list = X.columns.values

    # set a random seed
    np.random.seed = 123

    # random forest
    model_random_forest = RandomForestClassifier(n_estimators=1000)

    # 1. most important features

    # fit model
    model_random_forest.fit(X, y)
    # check feature importance.
    # http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    importances = model_random_forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(20):
            print("%d. feature %s (%f)" % (f + 1, features_list[indices[f]], importances[indices[f]]))

    # that's the features, ordered by importance
    important_features = features_list[indices]

    # Learning curves
    if do_learning_curves:

        # initialise cross-validation
        cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=3, test_size=0.2, random_state=0)
        helpers.plot_learning_curve(model_random_forest, X, y, n_jobs=-1, cv=cv)
        plt.savefig('./figures/learning_curves.png')

    # 3. Errors by number of features
    if do_features_curves:

        # split by groups of 10, in order of importance
        model_random_forest = RandomForestClassifier(n_estimators=1000)
        # initialise cross-validation
        cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=3, test_size=0.2, random_state=0)
        helpers.plot_features_curves(model_random_forest, X, y, important_features)






if __name__ == '__main__':
    main()

