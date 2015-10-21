# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter
import argparse
import sys
import os
import numpy as np
from helpers import helpers


def main():
    # widen pd output for debugging
    pd.set_option('display.width', 1000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file to parse (CSV)")
    parser.add_argument("--kaggle-test-set", help="Kaggle test set preprocessed (CSV)")
    parser.add_argument("--export-test", help="Export baseline predictions to CSV", action="store_true")
    parser.add_argument("--baseline", help="Train a baseline model and report error", action="store_true")
    parser.add_argument("--baseline-learning-curves", help="Build baseline learning curves with f1 score", action="store_true")
    parser.add_argument("--pca-learning-curves", help="Build learning curves based on pca", action="store_true")
    parser.add_argument("--feature-importance-learning-curves", help="Build learning curves based on feature importance", action="store_true")
    parser.add_argument("--estimators-learning-curves", help="Build learning curves based on the number of trees", action="store_true")
    parser.add_argument("--grid-search", help="Grid search for best model", action="store_true")
    parser.add_argument("--best-model-to-kaggle", help="Predict Kaggle test set using best model. Kaggle test set must be provided", action="store_true")
    args = parser.parse_args()



    if args.input is None:
        parser.print_help()
        sys.exit()
    input_file = args.input

    # set a random seed
    np.random.seed = 123

    # load data
    df = pd.read_csv(input_file, sep=',')


    # split X and y
    y = df['Survived']
    X = df.drop('Survived', axis=1)

    # we will use this model for our analysis
    model = RandomForestClassifier(oob_score=True, random_state=123)

    # 1. Establish a baseline
    if args.baseline:
        """
        train a simple random forest model and get the output
        """
        model.fit(X, y)
        print "Out of bag error : %f " % (model.oob_score_)
        print "Train error : %f " % (model.score(X, y))
        # run on the kaggle test set if provided
        if args.kaggle_test_set:
            print "Generating Kaggle baseline"
            kaggle_set = pd.read_csv(args.kaggle_test_set, sep=',')
            # store the passengers Ids
            passengers_ids = pd.Series(kaggle_set['PassengerId'])
            kaggle_set.drop('PassengerId', axis=1, inplace=1)
            kaggle_pred = pd.Series(model.predict(kaggle_set), name='Survived')
            result = pd.concat([passengers_ids, kaggle_pred], axis=1)
            # save to csv
            result.to_csv(os.path.join('.', 'kaggle', 'baseline.csv'), sep=',', encoding='utf-8', index=False)

        # check feature importance.
        # http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

        # store the feature list
        features_list = X.columns.values

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(20 if X.shape[1] >= 20 else X.shape[1]):
                print("%d. feature %s (%f)" % (f + 1, features_list[indices[f]], importances[indices[f]]))

    # 2. Baseline learning curves
    if args.baseline_learning_curves:
        train_scores, test_scores = helpers.rf_accuracy_by_sample_size(model, X, y, n_iter=10)
        plot = helpers.plot_learning_curves(title="Baseline Errors", train_scores=train_scores, test_scores=test_scores, with_variance=True, x_label="Observations")
        plot.savefig('./figures/Accuracy_baseline.png')

    # 3. PCA learning curves
    if args.pca_learning_curves:
        train_scores, test_scores = helpers.rf_accuracy_by_pca(model, X, y, to_scale=['cabin_count', 'age', 'fare', 'family_size'])
        plot = helpers.plot_learning_curves(title="PCA and Error", train_scores=train_scores, test_scores=test_scores, with_variance=False, x_label="Variance (in %)")
        plot.savefig('./figures/Accuracy_PCA_learning_curves.png')

    # 4. Feature importance learning curves
    if args.feature_importance_learning_curves:
        train_scores, test_scores = helpers.rf_accuracy_by_feature_importance(model, X, y)
        plot = helpers.plot_learning_curves(title="Feature importance and Error", train_scores=train_scores, test_scores=test_scores, with_variance=False, x_label="Feature importance (in %)")
        plot.savefig('./figures/Accuracy_feature_importance_learning_curves.png')


    # 5. Numer of trees learning curves
    if args.estimators_learning_curves:
        train_scores, test_scores = helpers.rf_accuracy_by_n_estimator(model, X, y)
        plot = helpers.plot_learning_curves(title="Number of trees and Error", train_scores=train_scores, test_scores=test_scores, with_variance=False, x_label="Number of trees")
        plot.savefig('./figures/Accuracy_n_estimator_learning_curves.png')

    # 6. Grid search parameters
    if args.grid_search:
        test_model = RandomForestClassifier(n_estimators=80, random_state=123)
        parameters = {'criterion': ['gini', 'entropy'],
                      'max_features': [.2, .5, .8, 'auto', None],
                      'max_depth': [3, 5, 10, 15, 20, None],
                      'min_samples_leaf': [1, 2, 5]
                      }
        grid_search = GridSearchCV(test_model, parameters, verbose=1)
        grid_search.fit(X, y)
        # print report
        # http://scikit-learn.org/stable/auto_examples/randomized_search.html
        top_scores = sorted(grid_search.grid_scores_, key=itemgetter(1), reverse=True)[:10]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                score.mean_validation_score,
                np.std(score.cv_validation_scores)
            ))
            print("Parameters: {0}".format(score.parameters))
            print("")

        # print learning curves for the best 5 models
        for i, score in enumerate(top_scores):
            best_model = RandomForestClassifier(n_estimators=80,
                                                oob_score=True,
                                                random_state=123,
                                                criterion=score.parameters['criterion'],
                                                max_features=score.parameters['max_features'],
                                                max_depth=score.parameters['max_depth'],
                                                min_samples_leaf=score.parameters['min_samples_leaf']
                                                )
            train_scores, test_scores = helpers.rf_accuracy_by_sample_size(best_model, X, y, n_iter=10)
            plot = helpers.plot_learning_curves(title="Model " + str(i+1) + " Errors", train_scores=train_scores, test_scores=test_scores, with_variance=True, x_label="Observations")
            plot.savefig('./figures/Model_' + str(i + 1) + '_Accuracy_baseline.png')

    # 7. Kaggle submission for best model as determined by the grid search
    if args.best_model_to_kaggle:

        # this is our best model
        best_model = RandomForestClassifier(n_estimators=80, max_features=None, criterion="entropy", max_depth=10, min_samples_leaf=1, random_state=123)
        best_model.fit(X, y)
        print "Generating Kaggle submission"
        kaggle_set = pd.read_csv(args.kaggle_test_set, sep=',')
        # store the passengers Ids
        passengers_ids = pd.Series(kaggle_set['PassengerId'])
        kaggle_set.drop('PassengerId', axis=1, inplace=1)
        kaggle_pred = pd.Series(best_model.predict(kaggle_set), name='Survived')
        result = pd.concat([passengers_ids, kaggle_pred], axis=1)
        # save to csv
        result.to_csv(os.path.join('.', 'kaggle', 'best_model.csv'), sep=',', encoding='utf-8', index=False)



if __name__ == '__main__':
    main()

