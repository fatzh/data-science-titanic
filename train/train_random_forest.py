# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import cross_validation, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import argparse
import sys
import os
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

    while(True):
        # split training set for cross validation
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)

        # SVM
        score_forest = 0
        while score_forest < 0.80:
            model_random_forest = RandomForestClassifier(
                n_estimators=10000,
                max_features='auto',
                max_depth=10,
                criterion='entropy',
                bootstrap=False)
            model_random_forest.fit(X_train, y_train)
            score_forest = model_random_forest.score(X_test, y_test)
            print 'Random forest score : ', score_forest

        if user_yes_no_query("Save models to disk ? "):

            # round scores... and make strings
            score_forest = str(round(score_forest * 100, 4))

            # create dir for models
            if not os.path.exists(os.path.join('./models/random_forest', score_forest)):
                os.makedirs(os.path.join('./models/random_forest', score_forest))

            joblib.dump(model_random_forest, os.path.join('.', 'models', 'random_forest', score_forest, 'model_forest.pkl'))

if __name__ == '__main__':
    main()

