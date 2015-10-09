# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.externals import joblib
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
    parser.add_argument("--tweaklearningcurves", help="Calculate and plot tweakd learning curves (takes time...)", action="store_true")
    parser.add_argument("--featurescurves", help="Calcualte and plot errors by features (takes time...)", action="store_true")
    parser.add_argument("--featuresreduce", help="Use PCA to reduce the number of features", action="store_true")
    parser.add_argument("--savemodel", help="Save model", action="store_true")
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

    do_tweaked_learning_curves = False
    if args.tweaklearningcurves:
        do_tweaked_learning_curves = True

    do_feature_reduction = False
    if args.featuresreduce:
        do_feature_reduction = True

    save_model = False
    if args.savemodel:
        save_model = True

    # load data
    df = pd.read_csv(input_file, sep=',')


    # split X and y
    y = df['Survived']
    X = df.drop('Survived', axis=1)

    # set a random seed
    np.random.seed = 123

    # 3. Feature reduction
    if do_feature_reduction:

        # using pca, set threshold to 90%

        # Minimum percentage of variance we want to be described by the resulting transformed components
        variance_pct = .9

        # Create PCA object
        pca = PCA(n_components=variance_pct)

        # Transform the initial features
        X_transformed = pca.fit_transform(X,y)

        # Create a data frame from the PCA'd data
        pcaDataFrame = pd.DataFrame(X_transformed)

        print pcaDataFrame.shape[1], " components describe ", str(variance_pct)[1:], "% of the variance"

        # now using reduced features
        X = pd.DataFrame(X_transformed)

    # store the feature list
    features_list = X.columns.values

    # 1. most important features
    model_random_forest = RandomForestClassifier(n_estimators=1000)

    # fit model
    model_random_forest.fit(X, y)
    # check feature importance.
    # http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    importances = model_random_forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(20 if X.shape[1] >= 20 else X.shape[1]):
            print("%d. feature %s (%f)" % (f + 1, features_list[indices[f]], importances[indices[f]]))

    # that's the features, ordered by importance
    important_features = features_list[indices]

    # Learning curves
    if do_learning_curves:

        # initialise cross-validation
        # random forest
        model_random_forest = RandomForestClassifier(n_estimators=1000)
        cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10, test_size=0.4, random_state=0)
        helpers.plot_learning_curve(model_random_forest, X, y, n_jobs=-1, cv=cv)
        plt.savefig('./figures/learning_curves.png')

        # adjust hyperparameters

    # with feature reduction, we must reduce the max number of features
    if do_feature_reduction:
        max_features = X.shape[1]
    else:
        max_features = 5
    if do_tweaked_learning_curves:
        # Learning curves with tweaked parameters
        model_random_forest = RandomForestClassifier(n_estimators=1000, max_features=max_features, min_samples_split=0.01)
        cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10, test_size=0.4, random_state=0)
        helpers.plot_learning_curve(model_random_forest, X, y, n_jobs=-1, cv=cv)
        plt.savefig('./figures/learning_curves_tweaked_features.png')

    if save_model:
        model_random_forest = RandomForestClassifier(n_estimators=1000, max_features=max_features, min_samples_split=0.01)
        model_random_forest.fit(X, y)
        joblib.dump(model_random_forest, './model/random_forest.pkl')



    # 3. Errors by number of features
    if do_features_curves:

        # split by groups of 10, in order of importance
        model_random_forest = RandomForestClassifier(n_estimators=1000)
        # initialise cross-validation
        cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=3, test_size=0.2, random_state=0)
        helpers.plot_features_curves(model_random_forest, X, y, important_features)






if __name__ == '__main__':
    main()

