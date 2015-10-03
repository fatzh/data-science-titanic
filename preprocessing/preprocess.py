# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import argparse
import sys
import os
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

def flatten_feature(col):
    '''
    This function takes all the values in a column
    and returns a dataframe with boolean values
    for each possible values in the column
    '''
    # create a dataframe and convert to integer
    toReturn = pd.get_dummies(col, prefix=col.name).applymap(int)
    return toReturn

def fill_missing_ages(df, train):
    '''
    fills the age using a linear regression model
    '''
    # for the age, we need to set a value when Nan...
    if train:
        age_estimator = build_age_regression_model(df)
    else:
        age_estimator = joblib.load('./models/age_regressor.pkl')
    # use the estimator to build an array of missing ages
    ages = age_estimator.predict(df[['Sex', 'Fare', 'Pclass']][df['Age'].isnull()])
    indexes = df[['Sex', 'Fare', 'Pclass']][df['Age'].isnull()].index
    # convert ages array to panda serie
    fill_ages = pd.Series(np.array(ages), np.array(indexes))
    # and fill the missing values
    Age = pd.Series(df['Age'], name='age')
    Age.fillna(fill_ages, inplace=True)

    return Age


def build_age_regression_model(df):
    '''
    this builds a simple linear regression model for the age,
    based on the sex/class and fare paid by the
    passenger
    '''
    df_age = pd.DataFrame(df[['Sex', 'Fare', 'Pclass', 'Age']][df.Age.notnull()])
    df_age_data = df_age[['Sex', 'Fare', 'Pclass']]
    df_age_target = df_age['Age']
    regressor = LinearRegression()
    regressor.fit(df_age_data, df_age_target)
    # save regression model to disk
    joblib.dump(regressor, './models/age_regressor.pkl')
    return regressor

def names_to_title(df):
    '''
    We extract the title from the name here
    '''
    # create a new dataframe
    Title = pd.Series(df['Name'].str.extract('.*,\s?\w*?\s?(\w+)\..*'), name='title')
    # group titles as per the observations
    Title[Title.isin(['Lady', 'Countess', 'Dona'])] = 'Lady'
    Title[Title.isin(['Rev', 'Dr', 'Jonkheer', 'Major', 'Master', 'Capt', 'Col', 'Don'])] = 'Sir'
    Title[Title.isin(['Mlle', 'Ms', 'Miss'])] = 'Miss'
    Title[Title.isin(['Mrs', 'Mme'])] = 'Mrs'
    return Title


def names_to_bracket(df):
    '''
    We noticed that having a braket in your name
    has an impact on your survival rate...
    '''
    # create a new dataframe
    df2 = pd.DataFrame(df['Name'])
    df2['Bracket'] = df2.apply(lambda row: fill_bracket(row), axis=1)
    return pd.Series(df2['Bracket'], name='braket')


def fill_bracket(row):
    '''
    returns 1 if there's a braket in the name,
    0 otherwise
    '''
    if '(' in row['Name']:
        return 1
    else:
        return 0


def names_to_family(df, train):
    '''
    set the family name with the size of the family
    '''
    df2 = pd.DataFrame(df[['Name', 'SibSp', 'Parch']])
    df2['Family'] = df2.apply(lambda row: fill_family(row), axis=1)
    if train:
        # we're training the model, so we'll save the family we found
        # and use it when preprocessing the test data
        train_families = pd.unique(df2['Family'])
        joblib.dump(train_families, './vars/train_families.pkl')
    else:
        train_families = joblib.load('./vars/train_families.pkl')
        # we replace the family we don't know about by "unknown"
        df2['Family'][~df2['Family'].isin(train_families)] = 'unknown'
        pass
    return pd.Series(df2['Family'], name='family')


def fill_family(row):
    '''
    returns the family name and the size of the family
    Also returns "none" if travelling alone
    '''
    family_size = row['Parch'] + row['SibSp'] + 1
    family_name = row['Name'].split(',')[0]
    if family_size > 1:
        return family_name + '_' + str(family_size)
    else:
        return 'none'



def fill_fares(df, train):
    '''
    The fare is an important parameter,
    We fill the value with the mean of the class
    Save the values for future use
    '''
    Fare = pd.Series(df['Fare'], name='fare')
    if train:
        # build list of mean prices
        t = pd.concat([df['Pclass'], Fare], axis=1)
        t = t.loc[(t['fare'] != 0) & (pd.notnull(t['fare']))]
        grouped = t.groupby(['Pclass'])
        mean_fares = {}
        for name, group in grouped:
            mean_fares[name] = np.mean(group).fare
        joblib.dump(mean_fares, './vars/mean_fares.pkl')
    else:
        mean_fares = joblib.load('./vars/mean_fares.pkl')

    t = pd.concat([df['Pclass'], Fare], axis=1)
    for index, row in t[(Fare == 0) | (pd.isnull(Fare))].iterrows():
        Fare.loc[index] = mean_fares[t.loc[index, 'Pclass']]
    return Fare


def fill_has_cabin(df):
    '''
    Set the number of cabins
    '''
    HasCabin = pd.Series(df['Cabin'], name='cabin')
    # count number of cabins
    return HasCabin.fillna('').map(lambda x: len(str(x).split()))


def fill_cabin_deck(df, train):
    '''
    returns the deck where the cabin is located.
    This is exctracting the first letter of the cabin.
    When no cabin, set emtpy
    '''
    CabinDeck = pd.Series(df['Cabin'], name='deck')
    # missing decks are coded NA
    CabinDeck = CabinDeck.map(lambda x: 'NA' if pd.isnull(x) else str(x)[0])
    if train:
        # we're training the model, so we'll save the decks we found
        # and use it when preprocessing the test data
        train_decks = pd.unique(CabinDeck)
        joblib.dump(train_decks, './vars/train_decks.pkl')
    else:
        train_decks = joblib.load('./vars/train_decks.pkl')
        # we replace the decks we don't know about by "unknown"
        CabinDeck[~CabinDeck.isin(train_decks)] = 'unknown'
        pass
    return CabinDeck

def fill_port(df, train):
    '''
    fill with the most common value, S
    '''
    Port = pd.Series(df['Embarked'])
    if train:
        default_port = 'S'
        joblib.dump(default_port, './vars/default_port.pkl')
    else:
        default_port = joblib.load('./vars/default_port.pkl')
    Port[pd.isnull(Port)] = default_port
    return Port


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
        # create train data folders if required
        if not os.path.exists('./vars'):
            os.mkdir('./vars')
        if not os.path.exists('./models'):
            os.mkdir('./models')
        if not os.path.exists('./encoders'):
            os.mkdir('./encoders')
        if not os.path.exists('./scalers'):
            os.mkdir('./scalers')

    # create output dir
    if not os.path.exists('./output'):
        os.mkdir('./output')

    # load data
    df = pd.read_csv(input_file)

    # feature engineering

    # transform Sex labels to 0/1 and save encoder to disk
    if train:
        lb_sex = preprocessing.LabelEncoder()
        df['Sex'] = lb_sex.fit_transform(df['Sex'])
        joblib.dump(lb_sex, './encoders/lb_sex.pkl')
    else:
        lb_sex = joblib.load('./encoders/lb_sex.pkl')
        df['Sex'] = lb_sex.transform(df['Sex'])
    Sex = pd.Series(df['Sex'], name='sex')

    # Passenger class is ok, just needs flattening
    Pclass = flatten_feature(pd.Series(df['Pclass'], name='class'))

    # Extract titles from passenger names
    Title = names_to_title(df)
    # and flatten
    Title = flatten_feature(Title)

    # Extract brackets from names, returns 0/1 Serie
    Bracket = names_to_bracket(df)

    # get the families from the name
    Family = names_to_family(df, train)
    # and flatten
    Family = flatten_feature(Family)
    if train:
        # save family features
        joblib.dump(Family.keys(), './vars/families_features.pkl')
    else:
        # we must add the families from the train set
        # and set them to 0
        # (that's to have the same number of features)
        families_features = joblib.load('./vars/families_features.pkl')
        missing_indexes = families_features[
            ~np.in1d(families_features, Family.keys())
        ]
        for i in missing_indexes:
            Family[i] = 0
        # also unknown families are not needed, as we don't have
        # any information on them anyway
        Family = Family.drop('family_unknown', 1)
    # fill missing ages using a linear regression model
    Age = fill_missing_ages(df, train)

    # siblings and parents are ok
    Siblings = pd.Series(df['SibSp'], name='siblings')
    Parents = pd.Series(df['Parch'], name='parents')

    # need to fill some missing fares using the median
    Fare = fill_fares(df, train)

    # get feature to know how many cabin are assigned to each passenger
    HasCabin = fill_has_cabin(df)

    # get a feature to get the cabin deck
    CabinDeck = fill_cabin_deck(df, train)
    # and flatten this
    CabinDeck = flatten_feature(CabinDeck)
    if train:
        # save decks that we know about
        joblib.dump(CabinDeck.keys(), './vars/decks_features.pkl')
    else:
        # we must add the descks from the train set
        decks_features = joblib.load('./vars/decks_features.pkl')
        missing_indexes = decks_features[
            ~np.in1d(decks_features, CabinDeck.keys())
        ]
        for i in missing_indexes:
            CabinDeck[i] = 0
            # also remove unknown decks
            if 'deck_unknown' in CabinDeck.keys():
                CabinDeck = CabinDeck.drop('deck_unknown', 1)

    # fill the embarkment port
    Port = fill_port(df, train)
    # and flatten
    Port = flatten_feature(Port)

    # and survived is good as it is, only for training
    if train:
        Survived = pd.Series(df['Survived'], name='survived')

    # feature scaling

    # age needs scaling
    if train:
        scaler_age = preprocessing.MinMaxScaler().fit(Age)
        joblib.dump(scaler_age, './scalers/scaler_age.pkl')
    else:
        scaler_age = joblib.load('./scalers/scaler_age.pkl')
    Age = pd.Series(scaler_age.transform(Age), name='age')

    # so does the fare, we also set a max at 280 to exclude outliers
    Fare[Fare > 280] = 280
    if train:
        scaler_fare = preprocessing.MinMaxScaler().fit(Fare)
        joblib.dump(scaler_fare, './scalers/scaler_fare.pkl')
    else:
        scaler_fare = joblib.load('./scalers/scaler_fare.pkl')
    Fare = pd.Series(scaler_fare.transform(Fare), name='fare')

    # create a new dataframe with the engineered features
    new_df = pd.concat([
        Pclass,
        Title,
        Bracket,
        Family,
        Sex,
        Age,
        Siblings,
        Parents,
        Fare,
        HasCabin,
        CabinDeck,
        Port
    ], axis=1)

    if train:
        # add the survived feature for training
        new_df = pd.concat([
            Survived,
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

