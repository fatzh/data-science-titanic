# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from operator import itemgetter
from time import time
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from helpers import helpers

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)
        ))
        print("Parameters: {0}".format(score.parameters))
        print("")

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
    parser.add_argument("--hyperparamsgrid", help="Test hyperparameters using grid search", action="store_true")
    parser.add_argument("--hyperparamstestmaxfeatures", help="Test max features hyperparameters", action="store_true")
    parser.add_argument("--hyperparamstestmaxdepth", help="Test max depth hyperparameters", action="store_true")
    parser.add_argument("--hyperparamstestcriterion", help="Test criterion hyperparameters", action="store_true")
    parser.add_argument("--extratrees", help="Calculate and plot learning curves for an ExtraTree model", action="store_true")
    parser.add_argument("--addfeatures", help="Add artificial features", action="store_true")
    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        sys.exit()
    input_file = args.input

    do_learning_curves = args.learningcurves
    do_features_curves = args.featurescurves
    do_tweaked_learning_curves = args.tweaklearningcurves
    do_feature_reduction = args.featuresreduce
    save_model = args.savemodel
    do_hyperparams_grid = args.hyperparamsgrid
    do_extratrees = args.extratrees
    do_test_hyperparams_max_features = args.hyperparamstestmaxfeatures
    do_test_hyperparams_max_depth = args.hyperparamstestmaxdepth
    do_test_hyperparams_criterion = args.hyperparamstestcriterion
    do_extra_features = args.addfeatures

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
        X_transformed = pca.fit_transform(X, y)

        # Create a data frame from the PCA'd data
        pcaDataFrame = pd.DataFrame(X_transformed)

        print pcaDataFrame.shape[1], " components describe ", str(variance_pct)[1:], "% of the variance"

        # now using reduced features
        X = pcaDataFrame

    import ipdb; ipdb.set_trace() ## BREAKPOINT
    if do_extra_features:

        numerics = X.loc[:, ['age', 'fare', 'family_size', 'cabin_count']]

        # for each pair of variables, determine which mathmatical operators to use based on redundancy
        for i in range(0, numerics.columns.size-1):
            for j in range(0, numerics.columns.size-1):
                col1 = str(numerics.columns.values[i])
                col2 = str(numerics.columns.values[j])
                # multiply fields together (we allow values to be squared)
                if i <= j:
                    name = col1 + "*" + col2
                    X = pd.concat([X, pd.Series(numerics.iloc[:,i] * numerics.iloc[:,j], name=name)], axis=1)
                # add fields together
                if i < j:
                    name = col1 + "+" + col2
                    X = pd.concat([X, pd.Series(numerics.iloc[:,i] + numerics.iloc[:,j], name=name)], axis=1)
                # divide and subtract fields from each other
                if not i == j:
                    name = col1 + "/" + col2
                    X = pd.concat([X, pd.Series(numerics.iloc[:,i] / numerics.iloc[:,j], name=name)], axis=1)
                    name = col1 + "-" + col2
                    X = pd.concat([X, pd.Series(numerics.iloc[:,i] - numerics.iloc[:,j], name=name)], axis=1)
        import ipdb; ipdb.set_trace() ## BREAKPOINT

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
        model_random_forest = RandomForestClassifier(n_estimators=1000, oob_score=True)
        cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10, test_size=0.4, random_state=0)
        helpers.plot_learning_curve(model_random_forest, X, y, n_jobs=-1, cv=cv)
        plt.savefig('./figures/learning_curves.png')


    # ExtraTrees
    if do_extratrees:
        model_extra_trees = ExtraTreesClassifier(n_estimators=1000)
        cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10, test_size=0.4, random_state=0)
        helpers.plot_learning_curve(model_extra_trees, X, y, n_jobs=-1, cv=cv)
        plt.savefig('./figures/learning_curves_extra_trees.png')



    # with feature reduction, we must reduce the max number of features
    if do_feature_reduction:
        max_features = X.shape[1]
    else:
        max_features = 'auto'
    if do_tweaked_learning_curves:
        # Learning curves with tweaked parameters
        model_random_forest = RandomForestClassifier(
            n_estimators=1000,
            max_features='auto',
            max_depth=10,
            criterion='entropy',
            bootstrap=False)
        cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10, test_size=0.4, random_state=0)
        helpers.plot_learning_curve(model_random_forest, X, y, n_jobs=-1, cv=cv)
        plt.savefig('./figures/learning_curves_tweaked_features.png')

    if save_model:
        model_random_forest = RandomForestClassifier(
            n_estimators=10000,
            max_features='auto',
            max_depth=10,
            criterion='gini',
            bootstrap=False)
        model_random_forest.fit(X, y)
        joblib.dump(model_random_forest, './model/max_depth/random_forest.pkl')

    if do_test_hyperparams_max_features:
        mf = [.05, .25, .50, None, 'sqrt', 'log2']

        for i in mf:
            model_random_forest = RandomForestClassifier(
                n_estimators = 1000,
                max_features = i,
                max_depth = None,
                criterion = 'gini',
                bootstrap = False
            )
            cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10, test_size=0.4, random_state=0)
            helpers.plot_learning_curve(model_random_forest, X, y, n_jobs=-1, cv=cv)
            plt.savefig('./figures/learning_curves_hyperparams_max_feature_'+str(i)+'.png')

    if do_test_hyperparams_max_depth:
        md = [5, 10, 50, 100, None]

        for i in md:
            model_random_forest = RandomForestClassifier(
                n_estimators = 1000,
                max_features = 'auto',
                max_depth = i,
                criterion = 'gini',
                bootstrap = False
            )
            cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10, test_size=0.4, random_state=0)
            helpers.plot_learning_curve(model_random_forest, X, y, n_jobs=-1, cv=cv, title="Learning Curves, max_depth="+str(i))
            plt.savefig('./figures/learning_curves_hyperparams_max_depth_'+str(i)+'.png')

    if do_test_hyperparams_criterion:
        md = ['gini', 'entropy']

        for i in md:
            model_random_forest = RandomForestClassifier(
                n_estimators = 10000,
                max_features = 'auto',
                max_depth = 10,
                criterion = i,
                bootstrap = False
            )
            cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10, test_size=0.4, random_state=0)
            helpers.plot_learning_curve(model_random_forest, X, y, n_jobs=-1, cv=cv, title="Learning Curves, criterion="+str(i))
            plt.savefig('./figures/learning_curves_hyperparams_criterion_'+str(i)+'.png')

    # 3. Errors by number of features
    if do_features_curves:

        # split by groups of 10, in order of importance
        model_random_forest = RandomForestClassifier(n_estimators=1000)
        # initialise cross-validation
        cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=3, test_size=0.2, random_state=0)
        helpers.plot_features_curves(model_random_forest, X, y, important_features)



    # 4. hyperparameters
    if do_hyperparams_grid:

        # let's start with a basic model
        model_random_forest = RandomForestClassifier(n_estimators=10)
        # use a full grid over all parameters
        param_grid = {"max_depth": [3, 5, 10, 20, 50, None],
                      "max_features": [3, 10, .20, .50, 50, 'auto'],
                      "min_samples_split": [1, 3, 10, 50],
                      "min_samples_leaf": [1, 3, 10],
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}

        # run grid search
        # see
        # http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html#example-model-selection-randomized-search-py
        grid_search = GridSearchCV(model_random_forest, param_grid=param_grid)
        start = time()
        grid_search.fit(X, y)

        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
                    % (time() - start, len(grid_search.grid_scores_)))
        report(grid_search.grid_scores_)






if __name__ == '__main__':
    main()

