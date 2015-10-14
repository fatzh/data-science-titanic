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

    # families in the train set
    train_families = None

    # firstnames in the train set
    train_firstnames = None

    # cabin decks in the train set
    train_cabin_decks = None

    # bins for cabin positions
    train_cabin_positions_bins = None

    # mean fare per passenger class
    mean_fares = None

    def __init__(self, data, train=True, save_root_dir='.'):
        self.df = data
        self.train = train
        self.save_root_dir = save_root_dir

        self.age_estimator_path = os.path.join(save_root_dir, 'models/age_regressor.pkl')
        self.train_families_path = os.path.join(save_root_dir, 'vars/train_families.pkl')
        self.train_firstnames_path = os.path.join(save_root_dir, 'vars/train_firstnames.pkl')
        self.mean_fares_path = os.path.join(save_root_dir, 'vars/mean_fares.pkl')
        self.train_cabin_decks_path = os.path.join(save_root_dir, 'vars/train_cabin_decks.pkl')
        self.train_cabin_positions_bins_path = os.path.join(save_root_dir, 'vars/train_cabin_positions_bins.pkl')

        # create dirs
        if not os.path.exists(os.path.join(save_root_dir, 'models')):
            os.makedirs(os.path.join(save_root_dir, 'models'))
        if not os.path.exists(os.path.join(save_root_dir, 'vars')):
            os.makedirs(os.path.join(save_root_dir, 'vars'))

        if not train:
            # load estimators and other variables
            try:
                self.age_estimator = joblib.load(self.age_estimator_path)
            except IOError:
                print "Can not find age regression model. Re-run preprocessing on train set."
            try:
                self.train_families = joblib.load(self.train_families_path)
            except IOError:
                print "Can not find families. Re-run preprocessing on train set."
            try:
                self.train_firstnames = joblib.load(self.train_firstnames_path)
            except IOError:
                print "Can not find train firstnames. Re-run preprocessing on train set."

            try:
                self.mean_fares = joblib.load(self.mean_fares_path)
            except IOError:
                print 'Can not find mean fare, Re-run preprocessing on train set.'
            try:
                self.train_cabin_decks = joblib.load(self.train_cabin_decks_path)
            except IOError:
                print "Can not find the cabin decks. Re-run preprocessing on train set."
            try:
                self.train_cabin_positions_bins = joblib.load(self.train_cabin_positions_bins_path)
            except IOError:
                print "Can not find the cabin position bins. Re-run preprocessing on train set."

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

    def preprocess_quotes(self):
        '''
        returns a boolean 1 for quotes, 0 otherwise
        '''
        return pd.Series(self.df['Name'].map(lambda x: '"' in x), name='quotes').map(int)

    def preprocess_title(self):
        '''
        extract title from name
        and flatten it.

        also fills in the most common values
        '''
        # create a new dataframe
        title = pd.Series(self.df['Name'].str.extract('.*,\s?\w*?\s?(\w+)\..*'), name='title')
        # group titles as per the observations
        title[title.isin(['Lady', 'Countess', 'Dona'])] = 'Lady'
        title[title.isin(['Rev', 'Dr', 'Jonkheer', 'Major', 'Master', 'Capt', 'Col', 'Don'])] = 'Sir'
        title[title.isin(['Mlle', 'Ms', 'Miss'])] = 'Miss'
        title[title.isin(['Mrs', 'Mme'])] = 'Mrs'
        return pd.get_dummies(title, prefix='title').applymap(int)

    def preprocess_firstname(self, limit=5):
        '''
        extract firstname from name
        and flatten it.

        only consider the firstname that occur more than 'limit' (default 5)
        '''
        # create a new dataframe
        firstnames = pd.Series(self.df['Name'].str.extract('.*\.\so?f?\s?\(?(\w+)'), name='firstname')
        df2 = pd.concat([self.df['Name'], firstnames], axis=1)
        # save the firstnames from training set
        if self.train:
            df2['count'] = df2.groupby(['firstname']).transform('count')
            # ignore firstname that occur less than 'limit' times
            df2.loc[df2['count'] < limit,'firstname'] = 0
            firstnames = pd.get_dummies(df2['firstname'], prefix='firstname').applymap(int)
            firstnames.drop('firstname_0', axis=1, inplace=1)
            self.train_firstnames = firstnames.columns.tolist()
            joblib.dump(self.train_firstnames, self.train_firstnames_path)
        else:
            firstnames = pd.get_dummies(df2['firstname'], prefix='firstname').applymap(int)
            # remove unknown firstnames
            for f in firstnames.columns:
                if f not in self.train_firstnames:
                    firstnames.drop(f, axis=1, inplace=1)
            # add known firstnames (we need the same features on train and test
            # set)
            for f in self.train_firstnames:
                if f not in firstnames.columns:
                    firstnames[f] = 0
        return firstnames

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
            self.age_estimator = self.build_age_regression_model()
            joblib.dump(self.age_estimator, self.age_estimator_path)
        # for the age, we need to set a value when Nan...
        # use the estimator to build an array of missing ages

        # first get the input data for the estimator
        input_df = self.prepare_age_regression_data().drop('Age', axis=1)
        # and we just need the lines where the age is not known
        input_df = input_df[self.df['Age'].isnull()]
        ages = self.age_estimator.predict(input_df)
        # get indexes for missing ages
        indexes = self.df[['Sex', 'Fare', 'Pclass']][self.df['Age'].isnull()].index
        # convert ages array to panda serie, and set the indexes accordingly
        fill_ages = pd.Series(np.array(ages), np.array(indexes))
        # and fill the missing values
        Age = pd.Series(self.df['Age'], name='age')
        Age.fillna(fill_ages, inplace=True)

        return Age

    def build_age_regression_model(self):
        '''
        this builds a simple linear regression model for the age,
        based on the sex/class and fare paid by the
        passenger
        '''
        # first need to convert class/sex to int
        temp_df = self.prepare_age_regression_data()
        # exclude missing ages for training
        df_age = pd.DataFrame(temp_df[temp_df.Age.notnull()])
        df_age_data = df_age.drop('Age', axis=1)
        df_age_target = df_age['Age']
        regressor = LinearRegression()
        regressor.fit(df_age_data, df_age_target)
        # save regression model to disk
        return regressor

    def prepare_age_regression_data(self):
        '''
        we must preprocess the sex and classes in order
        to train the regression model for the missing ages
        '''
        sex = self.preprocess_sex()
        classes = self.preprocess_classes()
        fare = self.preprocess_fares()
        return pd.concat([sex, classes, fare, self.df['Age']], axis=1)

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

    def preprocess_cabin_position(self, bins=10):
        '''
        extracts the number from the cabin and group them into
        10 bins
        '''
        # extract the cabin number and convert to int
        cabins = pd.Series(self.df['Cabin'])
        # we just work on cabins with a number
        cabins[cabins.map(lambda x: str(x).isalpha())] = None
        cabin_positions = cabins[cabins.notnull()].apply(self.fill_cabin_position).apply(int)
        # merge new values
        cabins.update(cabin_positions)
        # cut into bins, labels are indexed from 0 to bins-1
        if self.train:
            # create the bins and save them for the test set
            cabin_positions, self.train_cabin_positions_bins = pd.cut(cabins, bins, labels=range(bins), retbins=True)
            joblib.dump(self.train_cabin_positions_bins, self.train_cabin_positions_bins_path)
        else:
            cabin_positions = pd.cut(cabins, self.train_cabin_positions_bins, labels=range(bins))
        return pd.get_dummies(cabin_positions, prefix='cabin_position').applymap(int)


    def fill_cabin_position(self, x):
        regex_match = re.compile("(\d+).*?").search(x)
        try:
            return regex_match.group()
        except AttributeError:
            return 0


    def preprocess_cabin_count(self):
        '''
        returns a column with the number of cabin per passenger
        '''
        cabin_count = pd.Series(self.df['Cabin'], name='cabin_count')
        return cabin_count.map(lambda x: 0 if pd.isnull(x) else len(str(x).split(' ')))

    def preprocess_port(self):
        '''
        return boolean values for the port of embarkment.

        Empty/Null values are ignored
        '''
        port = pd.Series(self.df['Embarked'], name='port')
        return pd.get_dummies(port, prefix='port').applymap(int)

    def compute_numeric_features(self, numerics):
        '''
        compute numeric features to add/substract/multiply/divide

        '''
        X = pd.DataFrame()
        # for each pair of variables, determine which mathmatical operators to use based on redundancy
        for i in range(0, numerics.columns.size-1):
            for j in range(0, numerics.columns.size-1):
                col1 = str(numerics.columns.values[i])
                col2 = str(numerics.columns.values[j])
                # multiply fields together (we allow values to be squared)
                if i <= j:
                    name = col1 + "*" + col2
                    X = pd.concat([X, pd.Series(numerics.iloc[:,i] * numerics.iloc[:,j], name=name)], axis=1)
                # add fields together
                if i < j:
                    name = col1 + "+" + col2
                    X = pd.concat([X, pd.Series(numerics.iloc[:,i] + numerics.iloc[:,j], name=name)], axis=1)
                # divide and subtract fields from each other
                if not i == j:
                    name = col1 + "/" + col2
                    X = pd.concat([X, pd.Series(numerics.iloc[:,i] / numerics.iloc[:,j], name=name)], axis=1)
                    name = col1 + "-" + col2
                    X = pd.concat([X, pd.Series(numerics.iloc[:,i] - numerics.iloc[:,j], name=name)], axis=1)
        return X
