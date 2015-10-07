import unittest
import pandas as pd
import numpy as np


from .titanic import Titanic


class TitanicPreprocessingTests(unittest.TestCase):

    def test_preprocess_passenger_class(self):
        '''
        Test the preprocessing of the passenger class.
        Should return flatten features of the class
        '''
        # create dummy Pclass serie
        data = np.array([1,3,2,1,3,2])
        df = pd.DataFrame(pd.Series(data, name='Pclass'))
        tp = Titanic(df)
        Pclass = tp.preprocess_classes()
        self.assertEqual(len(Pclass.columns), 3)
        self.assertSequenceEqual(['class_1', 'class_2', 'class_3'], Pclass.columns.tolist())
        expected = {
            'class_1': {0: 1,
                        1: 0,
                        2: 0,
                        3: 1,
                        4: 0,
                        5: 0},
            'class_2': {0: 0,
                        1: 0,
                        2: 1,
                        3: 0,
                        4: 0,
                        5: 1},
            'class_3': {0: 0,
                        1: 1,
                        2: 0,
                        3: 0,
                        4: 1,
                        5: 0}
        }
        self.assertEqual(expected, Pclass.to_dict())


    def test_preprocess_passenger_name_brackets(self):
        '''
        should return a boolean column with 1 for brackets in name
        and 0 otherwise
        '''
        data = np.array([
            'jean bono (zerost)',
            'jean pierre',
            'jean paul (first)',
            'jean baptiste (second)',
            'jean george',
        ])
        df = pd.DataFrame(pd.Series(data, name='Name'))
        tp = Titanic(df)
        brackets = tp.preprocess_brackets()
        expected = {0: 1,
                    1: 0,
                    2: 1,
                    3: 1,
                    4: 0,
                    }
        self.assertEqual(expected, brackets.to_dict())

    def test_preprocess_passenger_name_quotes(self):
        '''
        should return a boolean column with 1 for quotes in name
        and 0 otherwise
        '''
        data = np.array([
            'jean bono "zerost"',
            'jean pierre',
            'jean paul "first"',
            'jean baptiste "second"',
            'jean george',
        ])
        df = pd.DataFrame(pd.Series(data, name='Name'))
        tp = Titanic(df)
        quotes = tp.preprocess_quotes()
        expected = {0: 1,
                    1: 0,
                    2: 1,
                    3: 1,
                    4: 0,
                    }
        self.assertEqual(expected, quotes.to_dict())

    def test_preprocess_passenger_name_title(self):
        data = np.array([
            'Bono, Mr. Jean (zerost)',
            'Pierre, Mrs. Blabla',
            'Paul, Dona. jean paul (first)',
            'Baptiste, Dr. jean baptiste (second)',
            'Pierre, Sir. toto',
            'Pierrette, Lady. toto',
            'PIerette, Mme. toto',
            'Jeanette, Countess. toto. (test)'
        ])
        df = pd.DataFrame(pd.Series(data, name='Name'))
        tp = Titanic(df)
        titles = tp.preprocess_title()
        expected = {
            'title_Lady': {0: 0,
                           1: 0,
                           2: 1,
                           3: 0,
                           4: 0,
                           5: 1,
                           6: 0,
                           7: 1},
            'title_Mr': {0: 1,
                         1: 0,
                         2: 0,
                         3: 0,
                         4: 0,
                         5: 0,
                         6: 0,
                         7: 0},
            'title_Mrs': {0: 0,
                          1: 1,
                          2: 0,
                          3: 0,
                          4: 0,
                          5: 0,
                          6: 1,
                          7: 0},
            'title_Sir': {0: 0,
                          1: 0,
                          2: 0,
                          3: 1,
                          4: 1,
                          5: 0,
                          6: 0,
                          7: 0},
        }
        self.assertEqual(expected, titles.to_dict())

    def test_preprocess_firstnames(self):
        data = np.array([
            'Bono, Mr. Jean (zerost)',
            'Pierre, Mrs. Blabla',
            'Paul, Dona. Jean (first)',
            'Baptiste, Dr. Jean baptiste (second)',
            'Pierre, Sir. Toto',
            'Pierrette, Lady. Toto',
            'PIerette, Mme. Toto',
            'Jeanette, Countess. Toto (test)'
        ])
        df = pd.DataFrame(pd.Series(data, name='Name'))
        tp = Titanic(df, save_root_dir='./test_train_info')
        titles = tp.preprocess_firstname(limit=2)
        expected = {
            'firstname_Jean': {0: 1, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0},
            'firstname_Toto': {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
        }
        self.assertEqual(expected, titles.to_dict())

    def test_preprocess_firstnames_test_set(self):
        data = np.array([
            'Bono, Mr. Jose (zerost)',
            'Pierre, Mrs. Jose',
            'Paul, Dona. Jose (first)',
            'Baptiste, Dr. Jean baptiste (second)',
            'Pierre, Sir. Toto',
            'Pierrette, Lady. Toto',
            'PIerette, Mme. Toto',
            'Jeanette, Countess. Toto (test)'
        ])
        df = pd.DataFrame(pd.Series(data, name='Name'))
        tp = Titanic(df, train=False, save_root_dir='./test_train_info')
        titles = tp.preprocess_firstname(limit=2)
        expected = {
            'firstname_Jean': {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0},
            'firstname_Toto': {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
        }
        self.assertEqual(expected, titles.to_dict())


    def test_preprocess_passenger_sex(self):
        data = np.array([
            'female',
            'male',
            'male',
            'female',
            'female',
        ])
        df = pd.DataFrame(pd.Series(data, name='Sex'))
        tp = Titanic(df)
        sex = tp.preprocess_sex()
        expected = {0: 1,
                    1: 0,
                    2: 0,
                    3: 1,
                    4: 1,
                    }
        self.assertEqual(expected, sex.to_dict())

    def test_preprocess_family_size(self):
        data = {'Parch': {0: 2,
                          1: 1,
                          2: 0,
                          3: 2},
                'SibSp': {0: 3,
                          1: 0,
                          2: 3,
                          3: 0}
                }
        df = pd.DataFrame(data)
        tp = Titanic(df)
        family_size = tp.preprocess_family_size()
        expected = {0: 6,
                    1: 2,
                    2: 4,
                    3: 3}
        self.assertEqual(expected, family_size.to_dict())

    def test_preprocess_families(self):
        data = {'Name': {0: 'Goodwin, Mr. toto',
                         1: 'Goodwin, Mrs. tata',
                         2: 'Pierre, Mr. tete',
                         3: 'Pierre, Mrs. tata',
                         4: 'Pierre, Sir. toto'},
                'Parch': {0: 2,
                          1: 2,
                          2: 1,
                          3: 1,
                          4: 0},
                'SibSp': {0: 1,
                          1: 1,
                          2: 3,
                          3: 3,
                          4: 0},
                }
        df = pd.DataFrame(data)
        tp = Titanic(df, save_root_dir='./test_train_info')
        families = tp.preprocess_families()
        expected = {'family_Goodwin_4': {0: 1,
                                         1: 1,
                                         2: 0,
                                         3: 0,
                                         4: 0},
                    'family_Pierre_5': {0: 0,
                                        1: 0,
                                        2: 1,
                                        3: 1,
                                        4: 0},
                    }
        self.assertEqual(expected, families.to_dict())

    def test_preprocess_families_test_set(self):
        # test set based on trained data (known families)
        data = {'Name': {0: 'Goodwin, Mr. toto',
                         1: 'Goodwin, Mrs. tata',
                         2: 'TPierre, Mr. tete',
                         3: 'TPierre, Mrs. tata',
                         4: 'TPierre, Sir. toto'},
                'Parch': {0: 2,
                          1: 2,
                          2: 1,
                          3: 1,
                          4: 0},
                'SibSp': {0: 1,
                          1: 1,
                          2: 3,
                          3: 3,
                          4: 0},
                }
        df = pd.DataFrame(data)
        tp = Titanic(df, train=False, save_root_dir='./test_train_info')
        families = tp.preprocess_families()
        expected = {'family_Goodwin_4': {0: 1,
                                         1: 1,
                                         2: 0,
                                         3: 0,
                                         4: 0},
                    'family_Pierre_5': {0: 0,
                                        1: 0,
                                        2: 0,
                                        3: 0,
                                        4: 0},
                    }
        self.assertEqual(expected, families.to_dict())

    def test_preprocess_ticket_number(self):
        data = np.array(['ASD/D 12345',
                         'JKJ 5692',
                         '34567',
                         '',
                         '123',
                         'QW/DS.DS 24354'])
        df = pd.DataFrame(data, columns=['Ticket'])
        tp = Titanic(df, save_root_dir='./test_train_info')
        first_ticket_numbers = tp.preprocess_first_ticket_numbers()
        expected = {
            'first_ticket_digit_0': {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0},
            'first_ticket_digit_1': {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0},
            'first_ticket_digit_2': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
            'first_ticket_digit_3': {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 0},
            'first_ticket_digit_4': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            'first_ticket_digit_5': {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0},
            'first_ticket_digit_6': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            'first_ticket_digit_7': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            'first_ticket_digit_8': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            'first_ticket_digit_9': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        }
        self.assertEqual(expected, first_ticket_numbers.to_dict())

    def test_preprocess_cabin_deck(self):
        data = np.array(['B45', None, 'C90', 'B99', 'D00', 'C00'])
        df = pd.DataFrame(data, columns=['Cabin'])
        tp = Titanic(df, save_root_dir='./test_train_info')
        cabin_deck = tp.preprocess_cabin_deck()
        expected = {
            'cabin_deck_B': {0: 1, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0},
            'cabin_deck_C': {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 1},
            'cabin_deck_D': {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0},
        }
        self.assertEqual(expected, cabin_deck.to_dict())

    def test_preprocess_cabin_deck_test_set(self):
        data = np.array(['B45', None, 'B90', 'E99', 'D00', 'D00'])
        df = pd.DataFrame(data, columns=['Cabin'])
        tp = Titanic(df, train=False, save_root_dir='./test_train_info')
        cabin_deck = tp.preprocess_cabin_deck()
        expected = {
            'cabin_deck_B': {0: 1, 1: 0, 2: 1, 3: 0, 4: 0, 5: 0},
            'cabin_deck_C': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            'cabin_deck_D': {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1},
        }
        self.assertEqual(expected, cabin_deck.to_dict())

    def test_preprocess_cabin_position(self):
        data = np.array(['C45', 'B34', 'F2', 'R90', 'R', 'T100 D89', None, 'F T45'])
        df = pd.DataFrame(data, columns=['Cabin'])
        tp = Titanic(df, save_root_dir='./test_train_info')
        cabin_positions = tp.preprocess_cabin_position(bins=2)
        # for this test, we have a range of 100 and 2 bins, so all cabins below
        # 50 go into the bin 0, the others in bin 1
        expected = {
            'cabin_position_0': {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1},
            'cabin_position_1': {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 0}
        }
        self.assertEqual(expected, cabin_positions.to_dict())

    def test_preprocess_cabin_position_test_set(self):
        data = np.array(['C45', 'B34', 'F2', 'R90', 'R', 'T100 D89', None, 'F T45'])
        df = pd.DataFrame(data, columns=['Cabin'])
        tp = Titanic(df, train=False, save_root_dir='./test_train_info')
        cabin_positions = tp.preprocess_cabin_position(bins=2)
        # for this test, we have a range of 100 and 2 bins, so all cabins below
        # 50 go into the bin 0, the others in bin 1
        expected = {
            'cabin_position_0': {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1},
            'cabin_position_1': {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 0}
        }
        self.assertEqual(expected, cabin_positions.to_dict())


    def test_preprocess_cabin_count(self):
        data = np.array(['B45 B34', None, 'C89', 'D23 D90 D74'])
        df = pd.DataFrame(data, columns=['Cabin'])
        tp = Titanic(df)
        cabin_count = tp.preprocess_cabin_count()
        expected = {0: 2,
                    1: 0,
                    2: 1,
                    3: 3}
        self.assertEqual(expected, cabin_count.to_dict())

    def test_preprocess_port(self):
        data = np.array(['C', 'S', 'Q', 'C', 'C', None, 'S', 'Q', 'C'])
        df = pd.DataFrame(data, columns=['Embarked'])
        tp = Titanic(df)
        port = tp.preprocess_port()
        expected = {'port_C': {0: 1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 0, 8: 1},
                    'port_Q': {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 0},
                    'port_S': {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 0, 8: 0},
                    }
        self.assertEqual(expected, port.to_dict())







