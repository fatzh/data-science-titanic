# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import cross_validation, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import argparse
import sys
import os




def main():
    # widen pd output for debugging
    pd.set_option('display.width', 1000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file to parse (CSV)")
    parser.add_argument("--model", help="Model to use(Pickle)")
    parser.add_argument("--output", help="Filename for the output file")
    args = parser.parse_args()

    if (args.input or args.model) is None:
        parser.print_help()
        sys.exit()
    input_file = args.input
    model_file = args.model

    output_file = os.path.basename(input_file)

    if args.output is not None:
        output_file = args.output

    # load model
    model = joblib.load(model_file)

    # load data
    df = pd.read_csv(input_file, sep=',')

    # save passenger ids
    passenger_ids = pd.Series(df['PassengerId'])

    # and drop for prediction
    df.drop('PassengerId', axis=1, inplace=True)

    # get prediction
    predictions = pd.Series(model.predict(df), name='Survived')
    probas = model.predict_proba(df)
    import ipdb; ipdb.set_trace() ## BREAKPOINT

    # create a new dataframe for the results
    result = pd.concat([passenger_ids, predictions], axis=1)

    # save data
    result.to_csv(os.path.join('.', 'output', output_file), sep=',', encoding='utf-8', index=False)


if __name__ == '__main__':
    main()

