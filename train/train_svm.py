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

    loop = True
    while(loop):
        # split training set for cross validation
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)

        # SVM
        score_svm = 0
        while score_svm < 0.84:
            model_svm = svm.SVC(kernel='poly', C=15)
            model_svm = model_svm.fit(X, y)
            score_svm = model_svm.score(X_test, y_test)
            print 'SVM - R-squared:', score_svm

        if user_yes_no_query("Save models to disk ? "):

            # round scores... and make strings
            score_svm = str(round(score_svm * 100, 4))

            # create dir for models
            if not os.path.exists(os.path.join('./models/svm', score_svm)):
                os.makedirs(os.path.join('./models/svm', score_svm))

            joblib.dump(model_svm, os.path.join('.', 'models', 'svm', score_svm, 'model_svm.pkl'))


        if not user_yes_no_query("Try again ? "):
            loop = False

if __name__ == '__main__':
    main()

