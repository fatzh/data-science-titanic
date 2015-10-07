# -*- coding: utf-8 -*-
import pandas as pd
import argparse
import sys
import os
from module.titanic import Titanic


def main():
    # widen pd output for debugging
    pd.set_option('display.width', 1000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file to parse (CSV)")
    parser.add_argument("--train", help="Train action, this will create models and encoders.", action="store_true")
    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        sys.exit()
    input_file = args.input
    train = False
    if args.train:
        train = True

    # create output dir
    if not os.path.exists('./output'):
        os.mkdir('./output')

    # load data
    df = pd.read_csv(input_file)

    # initialise preprocessor
    tp = Titanic(df, train)

    # create a new dataframe with the engineered features
    new_df = pd.concat([
        tp.preprocess_classes(),
        tp.preprocess_brackets(),
        tp.preprocess_quotes(),
        tp.preprocess_title(),
        tp.preprocess_firstname(),
        tp.preprocess_sex(),
        tp.preprocess_age(),
        tp.preprocess_family_size(),
        tp.preprocess_families(),
        tp.preprocess_first_ticket_numbers(),
        tp.preprocess_fares(),
        tp.preprocess_cabin_deck(),
        tp.preprocess_cabin_position(),
        tp.preprocess_cabin_count(),
        tp.preprocess_port()
    ], axis=1)

    if train:
        # add the survived feature for training
        new_df = pd.concat([
            df['Survived'],
            new_df
        ], axis=1)
    else:
        # passenger ID is needed to build the submission file
        new_df = pd.concat([
            df['PassengerId'],
            new_df
            ], axis=1)

    # save data
    new_df.to_csv(os.path.join('.', 'output', os.path.basename(input_file)), sep=',', encoding='utf-8', index=False)


if __name__ == '__main__':
    main()

