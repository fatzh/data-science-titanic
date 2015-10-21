# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report
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
    parser.add_argument("--kaggle-test-file", help="Kaggle test set preprocessed (CSV)")
    parser.add_argument("--baseline", help="Train a baseline model and report error", action="store_true")
    parser.add_argument("--baseline-learning-curves", help="Build baseline learning curves with f1 score", action="store_true")
    parser.add_argument("--export-test", help="Export baseline predictions to CSV", action="store_true")
    parser.add_argument("--grid-search", help="Execute grid search on SVM parameters", action="store_true")
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


    # prepare a test set with 1/4 of the data
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=123)

    # scale features
    scaler = StandardScaler()
    to_scale = ['age', 'cabin_count', 'family_size', 'fare']
    scaler.fit(X_train[to_scale])
    X_train.loc[:, to_scale] = scaler.transform(X_train[to_scale])
    X_test.loc[:, to_scale] = scaler.transform(X_test[to_scale])

    # we will use this model for our analysis
    model = svm.SVC(random_state=123, verbose=True)

    # 1. Establish a baseline
    if args.baseline:
        """
        train a simple SVM model and get the scores
        """
        model.fit(X_train, y_train)
        y_pred = pd.Series(model.predict(X_test), name='Survived')
        print model.score(X_test, y_test)
        print classification_report(y_test, y_pred)
        # run on the kaggle test set if provided
        if args.kaggle_test_file:
            kaggle_set = pd.read_csv(args.kaggle_test_file, sep=',')
            # store the passengers Ids
            passengers_ids = pd.Series(kaggle_set['PassengerId'])
            kaggle_set.drop('PassengerId', axis=1, inplace=1)
            # scale
            kaggle_set.loc[:, to_scale] = scaler.transform(kaggle_set[to_scale])
            kaggle_pred = pd.Series(model.predict(kaggle_set), name='Survived')
            result = pd.concat([passengers_ids, kaggle_pred], axis=1)
            # save to csv
            result.to_csv(os.path.join('.', 'kaggle', 'baseline_svm.csv'), sep=',', encoding='utf-8', index=False)

        if args.export_test:
            # change the name of the new column
            y_pred.name = "Predicted"
            result = pd.concat([X_test.reset_index(drop=True),
                                y_test.reset_index(drop=True),
                                y_pred.reset_index(drop=True)], axis=1)
            # save to cv
            result.to_csv(os.path.join('.', 'predictions', 'baseline_svm_predictions.csv'), sep=',', encoding='utf-8', index=False)


    # 2. Baseline learning curves
    if args.baseline_learning_curves:
        model = svm.SVC(random_state=123, verbose=True, C=1.4, kernel='poly')
        cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=100, test_size=0.2, random_state=123)
        plot = helpers.plot_learning_curves_cv(model, X_train, y_train, cv=cv, n_jobs=4)
        plot.savefig('./figures/svm_cv_F1_baseline.png')

    # 3. Grid search parameters
    if args.grid_search:
        test_model = svm.SVC(random_state=123)
        parameters = {'C': np.linspace(.1, 2, 10),
                      'kernel': ['rbf', 'poly'],
                      'class_weight': ['auto', None]
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
            best_model = svm.SVC(C=score.parameters['C'],
                                 kernel=score.parameters['kernel'],
                                 class_weight=score.parameters['class_weight'],
                                 random_state=123,
                                 )
            train_scores, test_scores = helpers.f1_scores_by_sample_size(best_model, X_train, y_train, X_test, y_test, n_iter=10)
            plot = helpers.plot_learning_curves(title="Model " + str(i+1) + " Errors", train_scores=train_scores, test_scores=test_scores, with_variance=True, x_label="Observations")
            plot.savefig('./figures/Model_SVM_' + str(i + 1) + '_Accuracy_baseline.png')




if __name__ == '__main__':
    main()

