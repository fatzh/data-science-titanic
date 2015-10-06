# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import os
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib


class Titanic():
    '''
    That's the main preprocessing class
    able to preprocess all the required fetures
    from the dataset
    '''

    train = True

    # linear model to fill in missing ages
    age_estimator = None

    # age scaler
    age_scaler = None

    # families in the train set
    train_families = None

    # cabin decks in the train set
    train_cabin_decks = None

    # mean fare per passenger class
    mean_fares = None

    def __init__(self, data, train=True, save_root_dir='.'):
        self.df = data
        self.train = train
        self.save_root_dir = save_root_dir

        self.age_estimator_path = os.path.join(save_root_dir, 'models/age_regressor.pkl')
        self.age_scaler_path = os.path.join(save_root_dir, 'scalers/scaler_age.pkl')
        self.train_families_path = os.path.join(save_root_dir, 'vars/train_families.pkl')
        self.mean_fares_path = os.path.join(save_root_dir, 'vars/mean_fares.pkl')
        self.train_cabin_decks_path = os.path.join(save_root_dir, 'vars/train_cabin_decks.pkl')

        # create dirs
        if not os.path.exists(os.path.join(save_root_dir, 'models')):
            os.makedirs(os.path.join(save_root_dir, 'models'))
        if not os.path.exists(os.path.join(save_root_dir, 'vars')):
            os.makedirs(os.path.join(save_root_dir, 'vars'))
        if not os.path.exists(os.path.join(save_root_dir, 'scalers')):
            os.makedirs(os.path.join(save_root_dir, 'scalers'))

        if not train:
            # load estimators and other variables
            try:
                self.age_estimator = joblib.load(self.age_estimator_path)
            except IOError:
                print "Can not find age regression model. Re-run preprocessing on train set."
            try:
                self.scaler_age = joblib.load(self.age_scaler_path)
            except IOError:
                print "Can not find the age scale. Re-run preprocessing on train set."
            try:
                self.train_families = joblib.load(self.train_families_path)
            except IOError:
                print "can not find families. Re-run preprocessing on train set."
            try:
                self.mean_fares = joblib.load(self.mean_fares_path)
            except IOError:
                print 'Can not find mean fare, Re-run preprocessing on train set.'
            try:
                self.train_cabin_decks = joblib.load(self.train_cabin_decks_path)
            except IOError:
                print "Can not find the cabin decks. Re-run preprocessing on train set."

    def preprocess_classes(self):
        '''
        returns a flat dataframe with all classes
        '''
        return pd.get_dummies(pd.Series(self.df['Pclass'], name='class'), prefix='class').applymap(int)

    def preprocess_brackets(self):
        '''
        returns a boolean 1 for bracket, 0 otherwise
        '''
        return pd.Series(self.df['Name'].map(lambda x: '(' in x), name='bracket').map(int)

    def preprocess_title(self):
        '''
        extract title from name
        and flatten it.

        Also fills in the most common values
        '''
        # create a new dataframe
        Title = pd.Series(self.df['Name'].str.extract('.*,\s?\w*?\s?(\w+)\..*'), name='title')
        # group titles as per the observations
        Title[Title.isin(['Lady', 'Countess', 'Dona'])] = 'Lady'
        Title[Title.isin(['Rev', 'Dr', 'Jonkheer', 'Major', 'Master', 'Capt', 'Col', 'Don'])] = 'Sir'
        Title[Title.isin(['Mlle', 'Ms', 'Miss'])] = 'Miss'
        Title[Title.isin(['Mrs', 'Mme'])] = 'Mrs'
        return pd.get_dummies(Title, prefix='title').applymap(int)

    def preprocess_sex(self):
        '''
        returns a Serie with male = 0 and female = 1
        '''
        return pd.Series(self.df['Sex'].map(lambda x: x == 'female'), name='sex').map(int)

    def preprocess_age(self):
        '''
        fills the age using a linear regression model
        '''
        if self.train:
            # build estimators and other variables
            self.age_estimator = self.build_age_regression_model(self.df)
            joblib.dump(self.age_estimator, self.age_estimator_path)
        # for the age, we need to set a value when Nan...
        # use the estimator to build an array of missing ages
        ages = self.age_estimator.predict(self.df[['Sex', 'Fare', 'Pclass']][self.df['Age'].isnull()])
        indexes = self.df[['Sex', 'Fare', 'Pclass']][self.df['Age'].isnull()].index
        # convert ages array to panda serie
        fill_ages = pd.Series(np.array(ages), np.array(indexes))
        # and fill the missing values
        Age = pd.Series(self.df['Age'], name='age')
        Age.fillna(fill_ages, inplace=True)

        # scale age
        if self.train:
            self.scaler_age = preprocessing.MinMaxScaler().fit(Age)
            joblib.dump(self.scaler_age, self.age_scaler_path)

        Age = pd.Series(self.scaler_age.transform(Age), name='age')

        return Age

    def build_age_regression_model(self):
        '''
        this builds a simple linear regression model for the age,
        based on the sex/class and fare paid by the
        passenger
        '''
        df_age = pd.DataFrame(self.df[['Sex', 'Fare', 'Pclass', 'Age']][self.df.Age.notnull()])
        df_age_data = df_age[['Sex', 'Fare', 'Pclass']]
        df_age_target = df_age['Age']
        regressor = LinearRegression()
        regressor.fit(df_age_data, df_age_target)
        # save regression model to disk
        return regressor

    def preprocess_family_size(self):
        '''
        returns a Serie with the family size
        equals to the number of Parch + number of SibSp + 1
        '''
        return pd.Series(self.df['Parch'] + self.df['SibSp'] + 1, name='family_size')


    def preprocess_families(self):
        '''
        creates a new feature for each identified family

        Families found in the train set are saved to adjust the
        test set accordingly
        '''
        df2 = pd.DataFrame(self.df[['Name', 'SibSp', 'Parch']])
        df2['Family'] = df2.apply(lambda row: self.fill_family(row), axis=1)

        families = pd.Series(df2['Family'], name='family')
        # flatten the families
        families = pd.get_dummies(families, prefix='family').applymap(int)
        # for test families, remove unknown and add empty families from train
        # set
        families.drop('family_none', axis=1, inplace=True)
        if self.train:
            # we're training the model, so we'll save the family we found
            # and use it when preprocessing the test data
            self.train_families = families.columns.tolist()
            joblib.dump(self.train_families, self.train_families_path)
        else:
            # remove unknown families
            for family in families.columns:
                if family not in self.train_families:
                    families.drop(family, axis=1, inplace=1)
            # add known families (we need the same features on train and test
            # set)
            for family in self.train_families:
                if family not in families.columns:
                    families[family] = 0

        return families

    def fill_family(self, row):
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

    def preprocess_first_ticket_numbers(self):
        '''
        returns features corresponfing to the first
        digit of the ticket if available.
        '''
        tickets = pd.Series(self.df['Ticket'])
        tickets = tickets.map(lambda x: self.fill_first_digit(x))
        # flatten the first digit
        tickets = pd.get_dummies(tickets, prefix='first_ticket_digit').applymap(int)
        # add all possibilities from 0 til 9
        for i in range(10):
            if 'first_ticket_digit_' + str(i) not in tickets.columns:
                tickets['first_ticket_digit_' + str(i)] = 0
        return tickets

    def fill_first_digit(self, x):
        regex_match = re.compile('[\d]').search(x)
        try:
            return regex_match.group()
        except AttributeError:
            return 0

    def preprocess_fares(self):
        '''
        The fare is an important parameter,
        We fill the value with the mean of the class
        Save the values for future use
        '''
        Fare = pd.Series(self.df['Fare'], name='fare')
        if self.train:
            # build list of mean prices
            t = pd.concat([self.df['Pclass'], Fare], axis=1)
            t = t.loc[(t['fare'] != 0) & (pd.notnull(t['fare']))]
            grouped = t.groupby(['Pclass'])
            self.mean_fares = {}
            for name, group in grouped:
                self.mean_fares[name] = np.mean(group).fare
            joblib.dump(self.mean_fares, self.mean_fares_path)

        t = pd.concat([self.df['Pclass'], Fare], axis=1)
        for index, row in t[(Fare == 0) | (pd.isnull(Fare))].iterrows():
            Fare.loc[index] = self.mean_fares[t.loc[index, 'Pclass']]
        return Fare

    def preprocess_cabin_deck(self):
        '''
        returns boolean values with the deck of the cabin
        if available
        '''
        cabin_deck = pd.Series(self.df['Cabin'], name='cabin_deck')
        # put 0 when no deck information is available
        cabin_deck = cabin_deck.map(lambda x: 0 if pd.isnull(x) else str(x)[0])
        # flattten this
        cabin_deck = pd.get_dummies(cabin_deck, prefix='cabin_deck').applymap(int)
        cabin_deck.drop('cabin_deck_0', axis=1, inplace=True)
        # if training, save known decks
        if self.train:
            self.train_cabin_decks = cabin_deck.columns.tolist()
            joblib.dump(self.train_cabin_decks, self.train_cabin_decks_path)
        else:
            # align cabin decks with train set
            # remove unknown cabin decks
            for deck in cabin_deck.columns:
                if deck not in self.train_cabin_decks:
                    cabin_deck.drop(deck, axis=1, inplace=1)
            # add known decks (we need the same features on train and test
            # set)
            for deck in self.train_cabin_decks:
                if deck not in cabin_deck.columns:
                    cabin_deck[deck] = 0
        return cabin_deck
