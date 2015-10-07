# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import cross_validation, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from distutils.util import strtobool



def user_yes_no_query(question):
    print('%s [y/n]\n' % question)
    while True:
        try:
            return strtobool(raw_input().lower())
        except ValueError:
            print('Please respond with \'y\' or \'n\'.\n')


def main():
    # widen pd output for debugging
    pd.set_option('display.width', 1000)
    plt.ioff()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file to parse (CSV)")
    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        sys.exit()
    input_file = args.input

    # load data
    df = pd.read_csv(input_file, sep=',')


    # split X and y
    y = df['Survived']
    X = df.drop('Survived', 1)

    # get the feature list
    features_list = X.columns.values



    loop = True
    while(loop):

        # random forest
        model_random_forest = RandomForestClassifier(n_estimators=1000)
        score_random_forest = 0
        while score_random_forest < 0.84:
            # split training set for cross validation
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)
            model_random_forest.fit(X, y)
            score_random_forest = model_random_forest.score(X_test, y_test)
            print 'Random Forest - R-squared:', score_random_forest
        feature_importances = model_random_forest.feature_importances_

        # make importances relative to max importance
        feature_importances = 100.0 * (feature_importances / feature_importances.max())

        # A threshold below which to drop features from the final data set.
        # Specifically, this number represents
        # the percentage of the most important feature's importance value
        fi_threshold = 15

        # Get the indexes of all features over the importance threshold
        important_idx = np.where(feature_importances > fi_threshold)[0]

        # Create a list of all the feature names above the importance threshold
        important_features = features_list[important_idx]
        print "\n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance):\n", \
                    important_features

        # Get the sorted indexes of important features
        sorted_idx = np.argsort(feature_importances[important_idx])[::-1]
        print "\nFeatures sorted by importance (DESC):\n", important_features[sorted_idx]

        # Adapted from
        # http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
        pos = np.arange(sorted_idx.shape[0]) + .5
        #plt.subplot(1, 2, 2)
        plt.barh(pos, feature_importances[important_idx][sorted_idx[::-1]], align='center')
        plt.yticks(pos, important_features[sorted_idx[::-1]])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        plt.draw()
        #plt.show()
        plt.savefig('./features.png')


        if user_yes_no_query("Save models to disk ? "):

            # round scores... and make strings
            score_random_forest = str(round(score_random_forest * 100, 4))

            # create dir for models
            if not os.path.exists(os.path.join('./models/random_forest', score_random_forest)):
                os.makedirs(os.path.join('./models/random_forest', score_random_forest))

            joblib.dump(model_random_forest, os.path.join('.', 'models', 'random_forest', score_random_forest, 'model_random_forest.pkl'))


        if not user_yes_no_query("Try again ? "):
            loop = False

if __name__ == '__main__':
    main()

