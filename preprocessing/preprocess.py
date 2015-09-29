# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib


def fill_missing_ages(df):
    # for the age, we need to set a value when Nan...
    age_estimator = build_age_regression_model(df)
    # use the estimator to build an array of missing ages
    ages = age_estimator.predict(df[['Sex', 'Fare', 'Pclass']][df['Age'].isnull()])
    indexes = df[['Sex', 'Fare', 'Pclass']][df['Age'].isnull()].index
    # convert ages array to panda serie
    fill_ages = pd.Series(np.array(ages), np.array(indexes))
    # and fill the missing values
    df['Age'].fillna(fill_ages, inplace=True)

    return df['Age']


def build_age_regression_model(df):
    '''
    this builds a simple linear regression
    model to determine the age
    when it's not available
    '''
    df = df[['Sex', 'Fare', 'Pclass', 'Age']][df.Age.notnull()]
    df_age_data = df[['Sex', 'Fare', 'Pclass']]
    df_age_target = df['Age']
    regressor = LinearRegression()
    regressor.fit(df_age_data, df_age_target)
    # save regrseeor model to disk
    joblib.dump(regressor, '../models/age_regressor.pkl')
    return regressor

def names_to_title(df):
    '''
    We use the title here, with a code for Mr, Mrs and Miss
    When not Miss or Mrs, this is assumed to be Mr (like for Don.)
    '''
    # create a new dataframe
    df2 = pd.DataFrame(df['Name'])
    df2['Title'] = df2.apply(lambda row: fill_title(row), axis=1)
    return df2['Title']


def fill_title(row):
    '''
    function to fill in the title from
    the name column
    '''
    if 'Miss' in row['Name']:
        return 'Miss'
    elif 'Mrs' in row['Name']:
        return 'Mrs'
    else:
        return 'Mr'


def names_to_bracket(df):
    '''
    We noticed that having a braket in your name
    has an impact on your survival rate...
    '''
    # create a new dataframe
    df2 = pd.DataFrame(df['Name'])
    df2['Bracket'] = df2.apply(lambda row: fill_bracket(row), axis=1)
    return df2['Bracket']


def fill_bracket(row):
    '''
    returns True if there's a braket in the name,
    False otherwise
    '''
    if '(' in row['Name']:
        return True
    else:
        return False


def fill_fares(df):
    '''
    The fare is an important parameter,
    we fill the missing values with the median
    '''
    Fare = df['Fare']
    Fare.ix[Fare == 0] = np.median(Fare)
    return Fare


def fill_has_cabin(df):
    '''
    returns true or false if the passenger
    has a cabin assigned
    '''
    HasCabin = df['Cabin']
    HasCabin[HasCabin.notnull()] = True
    HasCabin[HasCabin.isnull()] = False
    return HasCabin


def fill_port(df):
    '''
    fill with the most common value, S
    '''
    Port = df['Embarked']
    Port.fillna('S')
    return Port


def main():

    # load data
    df = pd.read_csv('../data/train.csv')

    # feature engineering
    lb = LabelEncoder()
    df['Sex'] = lb.fit_transform(df['Sex'])
    Pclass = df['Pclass']
    Title = names_to_title(df)
    Bracket = names_to_bracket(df)
    Sex = df['Sex']
    Age = fill_missing_ages(df)
    Siblings = df['SibSp']
    Parents = df['Parch']
    Fare = fill_fares(df)
    HasCabin = fill_has_cabin(df)
    Port = fill_port(df)
    Survived = df['Survived']

    new_df = pd.concat([Survived, Pclass, Title, Bracket, Sex, Age, Siblings, Parents, Fare, HasCabin, Port], axis=1)

    # save data
    new_df.to_csv('../preprocessed_data/train.csv', sep=';', encoding='utf-8')


if __name__ == '__main__':
    main()

