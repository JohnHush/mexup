#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import matplotlib.pyplot as plt
import os
from quantization.soccer.soccer_inversion import *
from quantization.soccer.soccer_dynamic_odds_cal import DynamicOddsCal

pd.set_option('display.max_columns', None)


class DataE0(object):
    """
    class handling England Premier League Data

    features including: Team Name,
                        Shots,
                        Shots on Target,
                        Fouls Committed,
                        Corners,
                        Yellow cards,
                        Red cards,

                        HAD odds of different Odds Providers
                        Bb1X2
                        BbOU
                        Bb>2.5
    targets including: Home Goal, Away Goal ( FullTime / HalfTime)
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._train_dict = None
        self._test_dict = None

        self._preprocess()

    def _preprocess(self):
        if 'source' not in self._kwargs.keys():
            raise ValueError('must have source in keywords')

        cache_file_path = os.path.join(
            os.path.dirname(self._kwargs['source'][0]), 'data_e0_cache.csv')

        if os.path.exists(cache_file_path):
            self._df = pd.read_csv(cache_file_path,
                                   date_parser=lambda s: pd.datetime.strptime(s, '%Y-%m-%d'),
                                   infer_datetime_format=True,
                                   parse_dates=['Date', ])
            return

        def date_parser(s):
            try:
                r = pd.datetime.strptime(s, '%d/%m/%y')
            except:
                r = pd.datetime.strptime(s, "%d/%m/%Y")
            else:
                pass
            return r

        df_list = [pd.read_csv(c,
                               date_parser=date_parser,
                               infer_datetime_format=True,
                               parse_dates=['Date', ]
                               ) for c in self._kwargs['source']]

        df = pd.concat(df_list, sort=True)
        df = df[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG',
                 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS',
                 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY',
                 'HR', 'AR', 'B365H', 'B365D', 'B365A', 'BbAv>2.5',
                 'BbAv<2.5', 'BbAHh', 'BbAvAHH', 'BbAvAHA', 'Avg>2.5',
                 'Avg<2.5', 'AHh', 'AvgAHH', 'AvgAHA']]

        df.loc[np.isnan(df['BbAv>2.5']), 'BbAv>2.5'] = df.loc[np.isnan(df['BbAv>2.5']), 'Avg>2.5']
        df.loc[np.isnan(df['BbAv<2.5']), 'BbAv<2.5'] = df.loc[np.isnan(df['BbAv<2.5']), 'Avg<2.5']
        df.loc[np.isnan(df['BbAHh']), 'BbAHh'] = df.loc[np.isnan(df['BbAHh']), 'AHh']
        df.loc[np.isnan(df['BbAvAHH']), 'BbAvAHH'] = df.loc[np.isnan(df['BbAvAHH']), 'AvgAHH']
        df.loc[np.isnan(df['BbAvAHA']), 'BbAvAHA'] = df.loc[np.isnan(df['BbAvAHA']), 'AvgAHA']

        df.drop(['Avg>2.5', 'Avg<2.5', 'AHh', 'AvgAHH', 'AvgAHA'], axis=1, inplace=True)
        df.dropna(axis=0, inplace=True)

        df = df.sort_values(by='Date')

        def _infer_market_sup_ttg_fn(series):
            config = InferSoccerConfig()
            config.ou_line = 2.5
            config.ahc_line = series['BbAHh']
            config.scores = [0, 0]
            config.over_odds = series['BbAv>2.5']
            config.under_odds = series['BbAv<2.5']
            config.home_odds = series['BbAvAHH']
            config.away_odds = series['BbAvAHA']

            return infer_ttg_sup(config)

        df['tmp'] = df.apply(_infer_market_sup_ttg_fn, axis=1)
        df['sup_market'] = df['tmp'].map(lambda s: round(s[0], 5))
        df['ttg_market'] = df['tmp'].map(lambda s: round(s[1], 5))

        df.drop(['tmp'], axis=1, inplace=True)
        df.to_csv(cache_file_path, index=False)

        self._df = df

    def split_ds_by_time(self, y=2018, m=1, d=1, epsilon=0., train_range=200):
        """
        split the dataframe into trainset and testset
        in a dict form
        { 'SOT': 1, 'HomeName': 'Chelsea', 'Goals': 3 }
        Args:
            y:  year
            m: month
            d: day
            epsilon: the time-fading parameter in DC model

        Returns:

        """
        train_start_date = pd.Timestamp(y, m, d) - train_range * pd.Timedelta(days=3.5)

        df_train = self._df[(self._df.Date < pd.Timestamp(y, m, d)) & \
                            (self._df.Date > train_start_date)].copy()
        df_test = self._df[self._df.Date >= pd.Timestamp(y, m, d)].copy()

        df_test['fading'] = 1.
        df_train['fading'] = df_train['Date'].map(
            lambda s: np.exp(-epsilon * (pd.Timestamp(y, m, d) - s) / pd.Timedelta(days=3.5)))

        self._train_dict = df_train.to_dict(orient='records')
        self._test_dict = df_test.to_dict(orient='records')

    def fetch_decay_train_dict(self,
                               y=2018,
                               m=1,
                               d=1,
                               epsilon=0.,
                               range=200,
                               unit=3.5,
                               drop_zero_goal_team=True ):
        """
        fetch train data from the original dataframe in a dict form

        { 'SOT': 1, 'HomeName': 'Chelsea', 'Goals': 3 }, e.g.
        Args:
            y:  year
            m: month
            d: day
            epsilon: the time-fading parameter in DC model
            range: the time range in unit
            unit: days/one unit

        Returns: a dict contain training data

        """
        start_date = pd.Timestamp(y, m, d) - range * pd.Timedelta(days=unit)
        df_train = self._df[(self._df.Date < pd.Timestamp(y, m, d)) & \
                            (self._df.Date > start_date)].copy()

        if drop_zero_goal_team:
            # get rid of the team dataset with 0 home goals or away goals in total
            team_set = set( df_train['HomeTeam'] ) | set( df_train['AwayTeam'] )
            for team in team_set:
                goal_in_toal = df_train.loc[ df_train.HomeTeam==team, 'FTHG'].sum() +\
                               df_train.loc[ df_train.AwayTeam==team, 'FTAG'].sum()

                goal_conceded= df_train.loc[ df_train.HomeTeam==team, 'FTAG'].sum() +\
                               df_train.loc[ df_train.AwayTeam==team, 'FTHG'].sum()

                if goal_in_toal==0 or goal_conceded==0:
                    df_train = df_train.drop( df_train[(df_train.HomeTeam==team)|\
                                                       (df_train.AwayTeam==team)].index )

        df_train['fading'] = df_train['Date'].map(
            lambda s: np.exp(-epsilon * (pd.Timestamp(y, m, d) - s) / pd.Timedelta(days=unit)))

        train_dict = df_train.to_dict(orient='records')

        return train_dict

    def team_index_dict(self):
        # distribute team index
        team2index = {}
        team_set = set( self._df['HomeTeam'] ) | set( self._df['AwayTeam'] )

        team_list = sorted(list(team_set))
        for index, value in enumerate(team_list):
            team2index[value] = index

        return team2index

    def train_dict(self):
        return self._train_dict

    def test_dict(self):
        return self._test_dict


class DixonColesModel_v3(object):
    '''
    ref: Modelling Association Football Scores and Inefficiencies in the Football
    betting Market, 1997, Mark Dixon and Stuart G. Coles

    static model without weighting on match by time


    '''

    def __init__(self, ds):
        self._ds = ds
        self._preprocess()

    def _preprocess(self):
        '''
        make sure which features to be used
        and handle them properly,
        Returns:

        '''

        self._team_index = self._ds.team_index_dict()
        self._team_num = len( self._team_index.keys() )

        for k,v in self._team_index.items():
            print( k, v)

        # distribute team index
        # self._team_index = {}
        # team_set = set()
        #
        # for d in self._ds.train_dict():
        #     team_set.add(d['HomeTeam'])
        #     team_set.add(d['AwayTeam'])
        #
        # for d in self._ds.test_dict():
        #     team_set.add(d['HomeTeam'])
        #     team_set.add(d['AwayTeam'])
        #
        # self._team_num = len(team_set)
        #
        # team_list = sorted(list(team_set))
        # for index, value in enumerate(team_list):
        #     self._team_index[value] = index

    @staticmethod
    def _calibration_matrix(home_goal,
                            away_goal,
                            home_exp,
                            away_exp,
                            rho):
        """

        Args:
            home_goal:
            away_goal:
            home_exp:
            away_exp:
            rho:

        Returns:

        """
        if home_goal == 0 and away_goal == 0:
            return 1. - home_exp * away_exp * rho
        elif home_goal == 0 and away_goal == 1:
            return 1. + home_exp * rho
        elif home_goal == 1 and away_goal == 0:
            return 1. + away_exp * rho
        elif home_goal == 1 and away_goal == 1:
            return 1. - rho
        else:
            return 1.

    # def _grad(self, params):
    #     train_dict_list = self._ds.train_dict()
    #     g = np.zeros(len(params))
    #
    #     for d in train_dict_list:
    #         home_goal = d['FTHG']
    #         away_goal = d['FTAG']
    #
    #         home_ind = self._team_index[d['HomeTeam']]
    #         away_ind = self._team_index[d['AwayTeam']]
    #
    #         ff = d['fading']
    #
    #         home_exp = np.exp(params[home_ind] + params[away_ind + self._team_num] + params[-1])
    #         away_exp = np.exp(params[away_ind] + params[home_ind + self._team_num])
    #
    #         home_indices = [home_ind, away_ind + self._team_num]
    #         away_indices = [away_ind, home_ind + self._team_num]
    #
    #         accumulate derivative of L/alpha and L/beta of Home
            # if home_goal == 0 and away_goal == 0:
            #     g[home_indices[0]] += ff * (home_goal - home_exp + (-home_exp * away_exp * params[-2]) / \
            #                                 (1 - home_exp * away_exp * params[-2]))
            #     g[home_indices[1]] += ff * (home_goal - home_exp + (-home_exp * away_exp * params[-2]) / \
            #                                 (1 - home_exp * away_exp * params[-2]))
            #
            #     g[-1] += ff * (home_goal - home_exp + (-home_exp * away_exp * params[-2]) / \
            #                    (1 - home_exp * away_exp * params[-2]))
            #
            # elif home_goal == 0 and away_goal == 1:
            #     g[home_indices[0]] += ff * (home_goal - home_exp + (home_exp * params[-2]) / \
            #                                 (1 + home_exp * params[-2]))
            #     g[home_indices[1]] += ff * (home_goal - home_exp + (home_exp * params[-2]) / \
            #                                 (1 + home_exp * params[-2]))
            #
            #     g[-1] += ff * (home_goal - home_exp + (home_exp * params[-2]) / \
            #                    (1 + home_exp * params[-2]))
            # else:
            #     g[home_indices[0]] += (home_goal - home_exp) * ff
            #     g[home_indices[1]] += (home_goal - home_exp) * ff
            #     g[-1] += (home_goal - home_exp) * ff
            #
            # accumulate another part
            # if home_goal == 0 and away_goal == 0:
            #     g[away_indices[0]] += ff * (away_goal - away_exp + (-home_exp * away_exp * params[-2]) / \
            #                                 (1 - home_exp * away_exp * params[-2]))
            #     g[away_indices[1]] += ff * (away_goal - away_exp + (-home_exp * away_exp * params[-2]) / \
            #                                 (1 - home_exp * away_exp * params[-2]))
            # elif home_goal == 1 and away_goal == 0:
            #     g[away_indices[0]] += ff * (away_goal - away_exp + (away_exp * params[-2]) / \
            #                                 (1 + away_exp * params[-2]))
            #     g[away_indices[1]] += ff * (away_goal - away_exp + (away_exp * params[-2]) / \
            #                                 (1 + away_exp * params[-2]))
            # else:
            #     g[away_indices[0]] += ff * (away_goal - away_exp)
            #     g[away_indices[1]] += ff * (away_goal - away_exp)
            #
            # if home_goal == 0 and away_goal == 0:
            #     g[-2] += ((-home_exp * away_exp) / (1 - home_exp * away_exp * params[-2])) * ff
            # elif home_goal == 0 and away_goal == 1:
            #     g[-2] += (home_exp / (1 + home_exp * params[-2])) * ff
            # elif home_goal == 1 and away_goal == 0:
            #     g[-2] += (away_exp / (1 + away_exp * params[-2])) * ff
            # elif home_goal == 1 and away_goal == 1:
            #     g[-2] += (-1 / (1 - params[-2])) * ff
            # else:
            #     pass
        #
        # return -1. * g

    def _grad2(self, params, *args ):
        # train_dict_list = self._ds.train_dict()
        train_dict_list = args[0]
        g = np.zeros(len(params))

        for d in train_dict_list:
            home_goal = d['FTHG']
            away_goal = d['FTAG']

            home_ind = self._team_index[d['HomeTeam']]
            away_ind = self._team_index[d['AwayTeam']]

            ff = d['fading']

            home_exp = np.exp(params[home_ind] + params[away_ind + self._team_num] + params[-1])
            away_exp = np.exp(params[away_ind] + params[home_ind + self._team_num])

            home_indices = [home_ind, away_ind + self._team_num]
            away_indices = [away_ind, home_ind + self._team_num]

            # accumulate derivative of L/alpha and L/beta of Home
            if home_goal == 0 and away_goal == 0:
                g[home_indices[0]] += ff * (home_goal - home_exp + (-home_exp * away_exp * params[-2]) / \
                                            (1 - home_exp * away_exp * params[-2]))
                g[home_indices[1]] += ff * (home_goal - home_exp + (-home_exp * away_exp * params[-2]) / \
                                            (1 - home_exp * away_exp * params[-2]))

                g[-1] += ff * (home_goal - home_exp + (-home_exp * away_exp * params[-2]) / \
                               (1 - home_exp * away_exp * params[-2]))

            elif home_goal == 0 and away_goal == 1:
                g[home_indices[0]] += ff * (home_goal - home_exp + (home_exp * params[-2]) / \
                                            (1 + home_exp * params[-2]))
                g[home_indices[1]] += ff * (home_goal - home_exp + (home_exp * params[-2]) / \
                                            (1 + home_exp * params[-2]))

                g[-1] += ff * (home_goal - home_exp + (home_exp * params[-2]) / \
                               (1 + home_exp * params[-2]))
            else:
                g[home_indices[0]] += (home_goal - home_exp) * ff
                g[home_indices[1]] += (home_goal - home_exp) * ff
                g[-1] += (home_goal - home_exp) * ff

            # accumulate another part
            if home_goal == 0 and away_goal == 0:
                g[away_indices[0]] += ff * (away_goal - away_exp + (-home_exp * away_exp * params[-2]) / \
                                            (1 - home_exp * away_exp * params[-2]))
                g[away_indices[1]] += ff * (away_goal - away_exp + (-home_exp * away_exp * params[-2]) / \
                                            (1 - home_exp * away_exp * params[-2]))
            elif home_goal == 1 and away_goal == 0:
                g[away_indices[0]] += ff * (away_goal - away_exp + (away_exp * params[-2]) / \
                                            (1 + away_exp * params[-2]))
                g[away_indices[1]] += ff * (away_goal - away_exp + (away_exp * params[-2]) / \
                                            (1 + away_exp * params[-2]))
            else:
                g[away_indices[0]] += ff * (away_goal - away_exp)
                g[away_indices[1]] += ff * (away_goal - away_exp)

            if home_goal == 0 and away_goal == 0:
                g[-2] += ((-home_exp * away_exp) / (1 - home_exp * away_exp * params[-2])) * ff
            elif home_goal == 0 and away_goal == 1:
                g[-2] += (home_exp / (1 + home_exp * params[-2])) * ff
            elif home_goal == 1 and away_goal == 0:
                g[-2] += (away_exp / (1 + away_exp * params[-2])) * ff
            elif home_goal == 1 and away_goal == 1:
                g[-2] += (-1 / (1 - params[-2])) * ff
            else:
                pass

        return -1. * g

    @staticmethod
    def _likelihood(home_goal,  # home goal in the match
                    away_goal,  # away goal in the match
                    home_exp,  # MLE param.
                    away_exp,  # MLE param.
                    rho):  # MLE param. calibration coefficient, degrades to Poisson Model when rho = 0

        return np.log(DixonColesModel_v3._calibration_matrix(home_goal, away_goal, home_exp, away_exp, rho) + \
                      np.finfo(float).eps) + \
               np.log(poisson.pmf(home_goal, home_exp) + np.finfo(float).eps) + \
               np.log(poisson.pmf(away_goal, away_exp) + np.finfo(float).eps)

    # def _objective_values_sum(self, params):
    #     train_dict_list = self._ds.train_dict()
    #
    #     obj = 0.
    #
    #     for d in train_dict_list:
    #         home_goal = d['FTHG']
    #         away_goal = d['FTAG']
    #
    #         home_ind = self._team_index[d['HomeTeam']]
    #         away_ind = self._team_index[d['AwayTeam']]
    #
    #         fading_factor = d['fading']
    #
    #         home_exp = np.exp(params[home_ind] + params[away_ind + self._team_num] + params[-1])
    #         away_exp = np.exp(params[away_ind] + params[home_ind + self._team_num])
    #
    #         obj = obj - DixonColesModel_v3._likelihood(home_goal,
    #                                                    away_goal,
    #                                                    home_exp,
    #                                                    away_exp,
    #                                                    params[-2]) * fading_factor
    #
    #     return obj

    def _objective_values_sum2(self, params, *args ):
        # train_dict_list = self._ds.train_dict()
        train_dict_list = args[0]

        obj = 0.

        for d in train_dict_list:
            home_goal = d['FTHG']
            away_goal = d['FTAG']

            home_ind = self._team_index[d['HomeTeam']]
            away_ind = self._team_index[d['AwayTeam']]

            fading_factor = d['fading']

            home_exp = np.exp(params[home_ind] + params[away_ind + self._team_num] + params[-1])
            away_exp = np.exp(params[away_ind] + params[home_ind + self._team_num])

            obj = obj - DixonColesModel_v3._likelihood(home_goal,
                                                       away_goal,
                                                       home_exp,
                                                       away_exp,
                                                       params[-2]) * fading_factor

        return obj

    # def solve(self,
    #           **kwargs):
    #
    #     options = kwargs.pop('options', {'disp': True, 'maxiter': 100})
    #     constraints = kwargs.pop('constraints', [{'type': 'eq',
    #                                               'fun': lambda x: sum(x[:self._team_num]) - self._team_num}])
    #
    #     np.random.seed( 0xCAFFE )
    #     init_vals = np.concatenate((np.random.uniform(0, 1, self._team_num),
    #                                 np.random.uniform(0, -1, self._team_num),
    #                                 [0.],
    #                                 [1.]))
    #
    #     self.model = minimize(self._objective_values_sum,
    #                           init_vals,
    #                           options=options,
    #                           constraints=constraints,
    #                           jac=self._grad,
    #                           **kwargs)
    #
    #     return self.model

    def _solve(self,
               train_dict,
               init_vals,
               **kwargs ):
        options = kwargs.pop('options', {'disp': True, 'maxiter': 100} )
        constraints = kwargs.pop('constraints',
                                 [{'type': 'eq',
                                   'fun': lambda x: sum(x[:self._team_num]) - self._team_num}])

        model = minimize( self._objective_values_sum2,
                          init_vals,
                          options=options,
                          constraints=constraints,
                          jac=self._grad2,
                          args=(train_dict,),
                          **kwargs)
        return model

    def inverse_decayed_model(self,
                              y=1989,
                              m=6,
                              d=4,
                              epsilon=0.0065,
                              range=200,
                              unit=3.5,
                              init_vals=None,
                              **kwargs ):

        train_dict = self._ds.fetch_decay_train_dict( y=y,
                                                      m=m,
                                                      d=d,
                                                      epsilon=epsilon,
                                                      range=range,
                                                      unit=unit )
        np.random.seed( 0xCAFFE )

        if init_vals is None:
            init_vals = np.concatenate((np.random.uniform(0, 1, self._team_num),
                                        np.random.uniform(0, -1, self._team_num),
                                        [0.],
                                        [1.]))

        model = self._solve( train_dict, init_vals, **kwargs )

        return model

    def inverse_time_varying_model(self,
                                   start_time=None,
                                   end_time=None,
                                   at_least_time_range=200,
                                   inverse_time_step=3.5,
                                   **kwargs):

        df_time_start = self._ds._df.loc[0, 'Date']
        df_time_end = self._ds._df.loc[len(self._ds._df) - 1, 'Date']

        if start_time is None:
            start_time = df_time_start + pd.Timedelta( days=at_least_time_range )
            if start_time > df_time_end:
                raise Exception( "Not enough data to do the training process!" )

        if end_time is None:
            end_time = df_time_end

        if end_time < start_time:
            raise Exception( "end time less than start time!")

        time_step = pd.Timedelta(days=inverse_time_step)

        dir_name = kwargs.pop('dir_name', 'model_dir')
        dir_path = os.path.join(os.path.abspath('./'), dir_name)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        split_time = start_time

        init_vals = None
        inverse_parameters_list = []

        while split_time < end_time:
            parameters_dict = {}
            split_time += time_step
            split_time_str = split_time.strftime('%Y-%m-%d')
            print(split_time_str)

            model_name = os.path.join(dir_path, split_time_str + '.model')

            model = self.inverse_decayed_model( y=split_time.year,
                                                m=split_time.month,
                                                d=split_time.day,
                                                epsilon=0.0065,
                                                range=200,
                                                unit=3.5,
                                                init_vals=init_vals,
                                                **kwargs
                                                )

            parameters_dict['Date'] = split_time
            for index, value in enumerate( model.x ):
                parameters_dict[index] = round( value, 5 )

            inverse_parameters_list.append( parameters_dict )
            print( model.x )

            with open(model_name, 'wb') as ooo:
                pickle.dump( model, ooo )

            init_vals = model.x

        output_df = pd.DataFrame( inverse_parameters_list )
        output_df.to_csv( 'test.csv', index=False )


    def fetch_model_into_dataframe(self, dir_path ):
        file_list = os.listdir( dir_path )
        for f in file_list:
            if f[-5:] != 'model':
                file_list.remove( f )

        parameters_list = []

        file_list.sort()
        for f in file_list:
            parameters_dict = {}
            p = os.path.join( os.path.abspath(dir_path), f )
            date_time = pd.to_datetime( f.split('.')[0], format='%Y-%m-%d' )
            parameters_dict['Date'] = date_time

            with open( p, 'rb' ) as iii:
                model = pickle.load( iii )
                for index, value in enumerate( model.x ):
                    parameters_dict[index] = round( value, 5 )

            parameters_list.append( parameters_dict )

        output_df = pd.DataFrame( parameters_list )
        output_df.to_csv( 'test.csv', index=False )


    def infer_expectation_rho(self, df, home, away , date_time ):
        # df['Date'] = df['Date'].map( lambda s: pd.to_datetime(s, format='%Y-%m-%d') )
        df_filtered = df.loc[ df['Date']<date_time, : ]

        if len(df_filtered) == 0:
            return None, None, None

        home_index = self._team_index[home]
        away_index = self._team_index[away]

        model_line = df_filtered.loc[ len(df_filtered)-1, : ]

        home_exp = np.exp( model_line[str(home_index)] + \
                           model_line[str(away_index+self._team_num)] + \
                           model_line[str( 2*self._team_num-1 + 2 )] )

        away_exp = np.exp( model_line[str(away_index)] + \
                           model_line[str(home_index+self._team_num)] )

        return home_exp, away_exp, model_line[str( 2*self._team_num -1 + 1 )]


    def over_under_bet_value_single_match(self,
                                          home_exp,
                                          away_exp,
                                          rho,
                                          over_odds,
                                          under_odds ):
        """
        betValue = probability * odds,
        if betValue > 1:
            technically, it has value
        else:
            no value
        Args:
            home_exp:
            away_exp:
            rho:
            over_odds: market over odds
            under_odds: market under odds
        """

        if home_exp is None or away_exp is None or rho is None:
            return 0,0

        sup = home_exp - away_exp
        ttg = home_exp + away_exp

        doc = DynamicOddsCal( sup_ttg=[sup, ttg],
                              present_socre=[0,0],
                              adj_params=[1,rho] )

        margin_dict = doc.over_under( line=2.5 )
        over_margin, under_margin = margin_dict[selection_type.OVER],\
                                    margin_dict[selection_type.UNDER]

        return over_margin*over_odds, under_margin*under_odds


    def backtesting_random_bet(self):
        test_data = self._ds._df

        def apply_fn( se ):
            over_odds = se['BbAv>2.5']
            under_odds = se['BbAv<2.5']
            home_outcome = se['FTHG']
            away_outcome = se['FTAG']

            bet_dir = None
            if np.random.rand() > 0.5:
                # o = over
                bet_dir = 0
            else:
                # 1 = under
                bet_dir = 1

            if home_outcome + away_outcome > 2.5:
                if bet_dir==0:
                    return over_odds -1
                return -1

            if bet_dir==0:
                return -1

            return under_odds -1

        test_data['pnl'] = test_data.apply( apply_fn, axis=1 )
        print( 'sum of pnl: ', test_data['pnl'].sum() )


    def backtesting_bet_value(self, model_df ):
        test_data = self._ds._df

        def apply_fn( series ):
            match_date_time = series['Date']
            over_odds = series['BbAv>2.5']
            under_odds = series['BbAv<2.5']
            home_team = series['HomeTeam']
            away_team = series['AwayTeam']
            home_outcome = series['FTHG']
            away_outcome = series['FTAG']

            home_exp, away_exp, rho = self.infer_expectation_rho( model_df,
                                                                  home_team,
                                                                  away_team,
                                                                  match_date_time )

            over_betvalue, under_betvalue = self.over_under_bet_value_single_match( home_exp,
                                                                                    away_exp,
                                                                                    rho,
                                                                                    over_odds,
                                                                                    under_odds )

            return over_betvalue, under_betvalue

        x = test_data.apply( apply_fn, axis=1 )

        test_data['over_bv'] = x.map( lambda s: s[0] )
        test_data['under_bv'] = x.map( lambda s: s[1] )

        def apply_fn2( se ):
            # evenly bet on every match
            over_odds = se['BbAv>2.5']
            under_odds = se['BbAv<2.5']
            home_outcome = se['FTHG']
            away_outcome = se['FTAG']
            over_bv = se['over_bv']
            under_bv = se['under_bv']

            if over_bv <=1.1 and under_bv <=1.1:
                return 0
            if over_bv > under_bv:
                # bet on over 2.5
                if home_outcome + away_outcome > 2.5:
                    return over_odds -1

                return -1

            if home_outcome + away_outcome < 2.5:
                return under_odds -1

            return -1

        test_data['pnl'] = test_data.apply( apply_fn2, axis=1 )
        print( test_data['pnl'])
        print( 'sum of pnl: ', test_data['pnl'].sum() )
        a = test_data.loc[ test_data.pnl!= 0 , 'pnl' ]
        print( a )



    def view_time_varying_model(self, df, view_list=[] ):
        fig, ax = plt.subplots()

        # ax.plot( df['Date'], np.exp( df['19']+df['59'] ) )
        # ax.plot( df['Date'], np.exp( df['23']+df['55'] ) )
        # ax.plot( df['Date'], df['0'] )
        # ax.plot( df['Date'], df['36'] )
        # ax.plot( df['Date'], df['72'] )

        for i in range(36):
            ax.plot( df['Date'], np.exp( df[str(i)] + df[str(i+36)]) )
        plt.show()
    '''
    def save_model(self, fn):
        with open(fn, 'wb') as ooo:
            pickle.dump(self.model, ooo)

    def load_model(self, fn):
        with open(fn, 'rb') as iii:
            self.model = pickle.load(iii)

    def infer_prob_matrix(self, home, away, num=10):
        # maximum goal of each side = 10
        home_exp, away_exp, rho = self.infer_exp_rho(home, away)
        home_goal_prob = np.array([poisson.pmf(g, home_exp) for g in range(num + 1)])
        away_goal_prob = np.array([poisson.pmf(g, away_exp) for g in range(num + 1)])

        calibration_matrix = np.array([[self._calibration_matrix(hg, ag, home_exp, away_exp, rho) \
                                        for ag in range(2)] for hg in range(2)])

        united_prob_matrix = np.outer(home_goal_prob, away_goal_prob)
        united_prob_matrix[:2, :2] = united_prob_matrix[:2, :2] * calibration_matrix

        return united_prob_matrix

    def infer_team_strength(self, team):
        # team stength is defined as TEAM_ATTACK - TEAM_DEFENCE
        team_ind = self._team_index[team]

        return self.model.x[team_ind] - self.model.x[team_ind + self._team_num]

    def infer_exp_rho(self, home, away):
        home_ind = self._team_index[home]
        away_ind = self._team_index[away]
        home_exp = np.exp(self.model.x[home_ind] +
                          self.model.x[away_ind + self._team_num] +
                          self.model.x[-1])
        away_exp = np.exp(self.model.x[away_ind] +
                          self.model.x[home_ind + self._team_num])

        return home_exp, away_exp, self.model.x[-2]

    def rmse(self):
        def _infer_fn(series):
            hometeam = series['HomeTeam']
            awayteam = series['AwayTeam']

            try:
                home_exp, away_exp, rho = self.infer_exp_rho(hometeam, awayteam)
            except:
                home_exp, away_exp, rho = 0, 0, 0

            return (home_exp, away_exp)

        df_train = pd.DataFrame(self._ds.train_dict())
        df_test = pd.DataFrame(self._ds.test_dict())

        df_test['tmp'] = df_test.apply(_infer_fn, axis=1)
        df_test['lam1_model1'] = df_test['tmp'].map(lambda s: round(s[0], 5))
        df_test['lam2_model1'] = df_test['tmp'].map(lambda s: round(s[1], 5))
        df_test['sup_model1'] = df_test['tmp'].map(lambda s: round(s[0] - s[1], 5))
        df_test['ttg_model1'] = df_test['tmp'].map(lambda s: round(s[0] + s[1], 5))
        df_test.drop(['tmp'], axis=1, inplace=True)

        df_test['true_sup'] = df_test.FTHG - df_test.FTAG
        df_test['true_ttg'] = df_test.FTHG + df_test.FTAG

        df_test = df_test[(df_test.sup_model1 != 0) | (df_test.ttg_model1 != 0)]

        N = len(df_test)

        market_sup_error = ((df_test.true_sup - df_test.sup_market) * \
                            (df_test.true_sup - df_test.sup_market)).sum() / N
        market_ttg_error = ((df_test.true_ttg - df_test.ttg_market) * \
                            (df_test.true_ttg - df_test.ttg_market)).sum() / N

        model1_sup_error = ((df_test.true_sup - df_test.sup_model1) * \
                            (df_test.true_sup - df_test.sup_model1)).sum() / N
        model1_ttg_error = ((df_test.true_ttg - df_test.ttg_model1) * \
                            (df_test.true_ttg - df_test.ttg_model1)).sum() / N

        print('market error of sup and ttg: ', np.sqrt(market_sup_error), np.sqrt(market_ttg_error))
        print('model1 error of sup and ttg: ', np.sqrt(model1_sup_error), np.sqrt(model1_ttg_error))
        '''


class Dataset(object):
    def __init__(self, *args, **kwargs):
        self.df = None
        self.args = args
        self.kwargs = kwargs

        self._fetch()
        self._preprocess()

    def _fetch(self):
        pass

    def _preprocess(self):
        pass

    def data(self):
        return self.df


class XlsData(Dataset):
    def _fetch(self):
        xls_name = self.kwargs.pop('xls', None)
        if not xls_name:
            raise ValueError('no keyword xls in **kwargs: %r' % self.kwargs)

        self.df = pd.read_excel(xls_name)

    def _preprocess(self):
        # transform the column Score into 2 cols
        # first: 4 * 60 outcomes
        # second: the last outcomes, no draw
        def map_with_draw(s):
            if not 'OT' in s:
                return s
            return s.split('(')[0]

        def map_no_draw(s):
            if not 'OT' in s:
                return s
            return s.split(' ')[-1][:-1]

        self.df['Score_with_draw'] = self.df.Score.map(map_with_draw)
        self.df['Score_no_draw'] = self.df.Score.map(map_no_draw)
        self.df = self.df.drop(['Score'], axis=1)

    def view(self):
        # print( self.df.head() )

        dff = self.df[self.df['Home Team'] == 'MINNESOTA TIMBERWOLVES']
        dff2 = self.df[self.df['Away Team'] == 'MINNESOTA TIMBERWOLVES']

        # score = self.df['Score'].map( lambda s: 'xxx' if 'OT' in s else s )

        # home_score = self.df['Score_no_draw'].map( lambda s: int(s.split(':')[0]) )
        # home_score = dff['Score_no_draw'].map( lambda s: int(s.split(':')[0]) ) + \
        #              dff2['Score_no_draw'].map(lambda s: int(s.split(':')[1]))

        home_score = dff['Score_no_draw'].map(lambda s: int(s.split(':')[0]))
        # home_score = dff2['Score_no_draw'].map(lambda s: int(s.split(':')[1]))

        fig, ax = plt.subplots()

        # nn , edge, _ = plt.hist( home_score , bins = 100 )

        # for i in range( len(nn) ):
        #     print( '[ %f, %f ]   %f '% (edge[i], edge[i+1], nn[i]) )
        # for i, n in enumerate(nn):
        #     print( i, n )
        ax.hist(home_score, bins=np.arange(80, 140, 1))
        # ax.plot( home_score.index, home_score )
        plt.show()


class UrlData(Dataset):
    def _fetch(self):
        if 'url' not in self.kwargs.keys():
            raise ValueError('must have url in keywords')

        df_list = [pd.read_csv(u) for u in self.kwargs['url']]
        self.df = pd.concat(df_list, sort=False)
        # self.df = pd.read_csv( self.kwargs['url'] )

    def _encode_feature(self, home, away):
        part1_feature = self.encoder.transform([[home, away]]).tolist()
        part2_feature = self.encoder.transform([[away, home]]).tolist()

        return [part1_feature] + [part2_feature] + [1] + [1]

    def _preprocess(self):
        self.encoder = OneHotEncoder(sparse=False)
        self.encoder.fit(self.df[['HomeTeam', 'AwayTeam']].to_numpy())

        part1_feature = self.encoder.transform(self.df.loc[:, ['HomeTeam', 'AwayTeam']].to_numpy()).tolist()
        part2_feature = self.encoder.transform(self.df.loc[:, ['AwayTeam', 'HomeTeam']].to_numpy()).tolist()

        self.feature = [[p1] + [p2] + [1] + [1] for p1, p2 in zip(part1_feature, part2_feature)]

        self.y = self.df.loc[:, ['FTHG', 'FTAG']].to_numpy()

    def get_train_data(self):
        return self.feature, self.y

    def view(self):
        print(self.df.head())


class DixonColesModel(object):
    '''
    ref: Modelling Association Football Scores and Inefficiencies in the Football
    betting Market, 1997, Mark Dixon and Stuart G. Coles

    static model without weighting on match by time
    '''

    def __init__(self, ds):
        self.ds = ds
        self.X, self.y = self.ds.get_train_data()
        self.dim = len(self.X[0][0]) // 2

    def _preprocessing(self):
        pass

    @staticmethod
    def _calibration_matrix(home_goal,
                            away_goal,
                            home_exp,
                            away_exp,
                            rho):
        if home_goal == 0 and away_goal == 0:
            return 1. - home_exp * away_exp * rho
        elif home_goal == 0 and away_goal == 1:
            return 1. + home_exp * rho
        elif home_goal == 1 and away_goal == 0:
            return 1. + away_exp * rho
        elif home_goal == 1 and away_goal == 1:
            return 1. - rho
        else:
            return 1.

    @staticmethod
    def _likelihood(home_goal,  # home goal in the match
                    away_goal,  # away goal in the match
                    home_exp,  # MLE param.
                    away_exp,  # MLE param.
                    rho):  # MLE param. calibration coefficient, degrades to Poisson Model when rho = 0

        return np.log(DixonColesModel._calibration_matrix(home_goal, away_goal, home_exp, away_exp, rho) + \
                      np.finfo(float).eps) + \
               np.log(poisson.pmf(home_goal, home_exp) + np.finfo(float).eps) + \
               np.log(poisson.pmf(away_goal, away_exp) + np.finfo(float).eps)

    def _objective_values_sum(self, params):
        train_features, train_y = self.ds.get_train_data()
        obj = 0.

        for isample in range(len(train_features)):
            X = train_features[isample]
            y = train_y[isample]

            home_indices = np.where(np.array(X[0]) != 0)[0]
            away_indices = np.where(np.array(X[1]) != 0)[0]

            # home_exp = params[home_indices[0]] * params[home_indices[1]] * params[-1]
            # away_exp = params[away_indices[0]] * params[away_indices[1]]

            home_exp = np.exp((np.array(params[: 2 * self.dim]) * np.array(X[0])).sum() + params[-1])
            away_exp = np.exp((np.array(params[: 2 * self.dim]) * np.array(X[1])).sum())

            obj = obj - DixonColesModel._likelihood(y[0], y[1], home_exp, away_exp, params[-2])

        return obj

    def _grad(self, params):
        train_features, train_y = self.ds.get_train_data()

        g = np.zeros(len(params))

        for isample in range(len(train_features)):
            X = train_features[isample]
            y = train_y[isample]

            home_exp = np.exp((np.array(params[: 2 * self.dim]) * np.array(X[0])).sum() + params[-1])
            away_exp = np.exp((np.array(params[: 2 * self.dim]) * np.array(X[1])).sum())

            # home_exp = min( home_exp, 5 )
            # away_exp = min( away_exp, 5 )

            home_indices = np.where(np.array(X[0]) != 0)[0]
            away_indices = np.where(np.array(X[1]) != 0)[0]

            home_goal = int(y[0])
            away_goal = int(y[1])

            # accumulate derivative of L/alpha and L/beta of Home
            if home_goal == 0 and away_goal == 0:
                g[home_indices[0]] += home_goal - home_exp + (-home_exp * away_exp * params[-2]) / \
                                      (1 - home_exp * away_exp * params[-2])
                g[home_indices[1]] += home_goal - home_exp + (-home_exp * away_exp * params[-2]) / \
                                      (1 - home_exp * away_exp * params[-2])

                g[-1] += home_goal - home_exp + (-home_exp * away_exp * params[-2]) / \
                         (1 - home_exp * away_exp * params[-2])

            elif home_goal == 0 and away_goal == 1:
                g[home_indices[0]] += home_goal - home_exp + (home_exp * params[-2]) / \
                                      (1 + home_exp * params[-2])
                g[home_indices[1]] += home_goal - home_exp + (home_exp * params[-2]) / \
                                      (1 + home_exp * params[-2])

                g[-1] += home_goal - home_exp + (home_exp * params[-2]) / \
                         (1 + home_exp * params[-2])
            else:
                g[home_indices[0]] += home_goal - home_exp
                g[home_indices[1]] += home_goal - home_exp
                g[-1] += home_goal - home_exp

            # accumulate another part
            if home_goal == 0 and away_goal == 0:
                g[away_indices[0]] += away_goal - away_exp + (-home_exp * away_exp * params[-2]) / \
                                      (1 - home_exp * away_exp * params[-2])
                g[away_indices[1]] += away_goal - away_exp + (-home_exp * away_exp * params[-2]) / \
                                      (1 - home_exp * away_exp * params[-2])
            elif home_goal == 1 and away_goal == 0:
                g[away_indices[0]] += away_goal - away_exp + (away_exp * params[-2]) / \
                                      (1 + away_exp * params[-2])
                g[away_indices[1]] += away_goal - away_exp + (away_exp * params[-2]) / \
                                      (1 + away_exp * params[-2])
            else:
                g[away_indices[0]] += away_goal - away_exp
                g[away_indices[1]] += away_goal - away_exp

            if home_goal == 0 and away_goal == 0:
                g[-2] += (-home_exp * away_exp) / (1 - home_exp * away_exp * params[-2])
            elif home_goal == 0 and away_goal == 1:
                g[-2] += home_exp / (1 + home_exp * params[-2])
            elif home_goal == 1 and away_goal == 0:
                g[-2] += away_exp / (1 + away_exp * params[-2])
            elif home_goal == 1 and away_goal == 1:
                g[-2] += -1 / (1 - params[-2])
            else:
                pass

        return -1. * g

    def solve(self,
              **kwargs):

        options = kwargs.get('options', {'disp': True, 'maxiter': 100})
        constraints = kwargs.get('constraints', [{'type': 'eq', 'fun': lambda x: sum(x[:self.dim]) - self.dim}])

        init_vals = np.concatenate((np.random.uniform(0, 1, self.dim),
                                    np.random.uniform(0, -1, self.dim),
                                    [0.],
                                    [1.]))

        self.model = minimize(self._objective_values_sum,
                              init_vals,
                              options=options,
                              constraints=constraints,
                              jac=self._grad,
                              **kwargs)

        return self.model

    def save_model(self, fn):
        with open(fn, 'wb') as ooo:
            pickle.dump(self.model, ooo)

    def load_model(self, fn):
        with open(fn, 'rb') as iii:
            self.model = pickle.load(iii)

    def infer_prob_matrix(self, home, away, num=10):
        # maximum goal of each side = 10
        home_exp, away_exp, rho = self.infer_exp_rho(home, away)
        home_goal_prob = np.array([poisson.pmf(g, home_exp) for g in range(num + 1)])
        away_goal_prob = np.array([poisson.pmf(g, away_exp) for g in range(num + 1)])

        calibration_matrix = np.array([[self._calibration_matrix(hg, ag, home_exp, away_exp, rho) \
                                        for ag in range(2)] for hg in range(2)])

        united_prob_matrix = np.outer(home_goal_prob, away_goal_prob)
        united_prob_matrix[:2, :2] = united_prob_matrix[:2, :2] * calibration_matrix

        return united_prob_matrix

    def infer_team_strength(self, team):
        # team stength is defined as TEAM_ATTACK - TEAM_DEFENCE
        team_one_hot = self.ds.encoder.transform([[team, team]])[0]
        team_one_hot[self.dim:] *= -1

        return (team_one_hot * self.model.x[:2 * self.dim]).sum()

    def infer_exp_rho(self, home, away):
        home_exp = np.exp((self.ds.encoder.transform([[home, away]]) * \
                           np.array(self.model.x[:2 * self.dim])).sum() + self.model.x[-1])
        away_exp = np.exp((self.ds.encoder.transform([[away, home]]) * \
                           np.array(self.model.x[:2 * self.dim])).sum())

        return home_exp, away_exp, self.model.x[-2]


class DixonColesModel_v1(object):
    '''
    ref: Modelling Association Football Scores and Inefficiencies in the Football
    betting Market, 1997, Mark Dixon and Stuart G. Coles

    static model without weighting on match by time
    '''

    def __init__(self, ds):
        self.ds = ds
        self.X, self.y = self.ds.get_train_data()
        self.dim = len(self.X[0][0]) // 2

    def _preprocessing(self):
        pass

    @staticmethod
    def _calibration_matrix(home_goal,
                            away_goal,
                            home_exp,
                            away_exp,
                            rho):
        if home_goal == 0 and away_goal == 0:
            return 1. - home_exp * away_exp * rho
        elif home_goal == 0 and away_goal == 1:
            return 1. + home_exp * rho
        elif home_goal == 1 and away_goal == 0:
            return 1. + away_exp * rho
        elif home_goal == 1 and away_goal == 1:
            return 1. - rho
        else:
            return 1.

    @staticmethod
    def _likelihood(home_goal,  # home goal in the match
                    away_goal,  # away goal in the match
                    home_exp,  # MLE param.
                    away_exp,  # MLE param.
                    rho):  # MLE param. calibration coefficient, degrades to Poisson Model when rho = 0

        return np.log(DixonColesModel._calibration_matrix(home_goal, away_goal, home_exp, away_exp, rho) + \
                      np.finfo(float).eps) + \
               np.log(poisson.pmf(home_goal, home_exp) + np.finfo(float).eps) + \
               np.log(poisson.pmf(away_goal, away_exp) + np.finfo(float).eps)

    def _objective_values_sum(self, params):
        train_features, train_y = self.ds.get_train_data()
        obj = 0.

        for isample in range(len(train_features)):
            X = train_features[isample]
            y = train_y[isample]

            home_exp = np.exp((np.array(params[: 2 * self.dim]) * np.array(X[0])).sum() + params[-1])
            away_exp = np.exp((np.array(params[: 2 * self.dim]) * np.array(X[1])).sum())

            obj = obj - DixonColesModel._likelihood(y[0], y[1], home_exp, away_exp, params[-2])

        return obj

    def solve(self,
              **kwargs):

        options = kwargs.get('options', {'disp': True, 'maxiter': 1000})
        constraints = kwargs.get('constraints', [{'type': 'eq', 'fun': lambda x: sum(x[:self.dim]) - self.dim}])

        init_vals = np.concatenate((np.random.uniform(0, 1, self.dim),
                                    np.random.uniform(0, -1, self.dim),
                                    [0.],
                                    [1.]))

        self.model = minimize(self._objective_values_sum,
                              init_vals,
                              options=options,
                              constraints=constraints,
                              **kwargs)

        return self.model

    def save_model(self, fn):
        with open(fn, 'wb') as ooo:
            pickle.dump(self.model, ooo)

    def load_model(self, fn):
        with open(fn, 'rb') as iii:
            self.model = pickle.load(iii)

    def infer_prob_matrix(self, home, away, num=10):
        # maximum goal of each side = 10
        home_exp, away_exp, rho = self.infer_exp_rho(home, away)
        home_goal_prob = np.array([poisson.pmf(g, home_exp) for g in range(num + 1)])
        away_goal_prob = np.array([poisson.pmf(g, away_exp) for g in range(num + 1)])

        calibration_matrix = np.array([[self._calibration_matrix(hg, ag, home_exp, away_exp, rho) \
                                        for ag in range(2)] for hg in range(2)])

        united_prob_matrix = np.outer(home_goal_prob, away_goal_prob)
        united_prob_matrix[:2, :2] = united_prob_matrix[:2, :2] * calibration_matrix

        return united_prob_matrix

    def infer_team_strength(self, team):
        # team stength is defined as TEAM_ATTACK - TEAM_DEFENCE
        team_one_hot = self.ds.encoder.transform([[team, team]])[0]
        team_one_hot[self.dim:] *= -1

        return (team_one_hot * self.model.x[:2 * self.dim]).sum()

    def infer_exp_rho(self, home, away):
        home_exp = np.exp((self.ds.encoder.transform([[home, away]]) * \
                           np.array(self.model.x[:2 * self.dim])).sum() + self.model.x[-1])
        away_exp = np.exp((self.ds.encoder.transform([[away, home]]) * \
                           np.array(self.model.x[:2 * self.dim])).sum())

        return home_exp, away_exp, self.model.x[-2]


if __name__ == "__main__":
    # ds = XlsData( xls='./StartClosePrices-5.xls' )
    # ds.view()

    # import time
    #
    # t1 = time.time()
    # ds = UrlData( url = [ '../../1920_E0.csv'  ])
    # dcm = DixonColesModel_v1( ds )
    #
    # dcm.solve()
    # print( dcm.model.x )
    # print( 'time: ', time.time() - t1 )
    # dcm.save_model( './EnglandPremierLeague_17181920_dcm.model')
    # dcm.load_model( './EnglandPremierLeague_1718_dcm.model' )
    # print( dcm.model.x )

    # home_exp, away_exp, rho = dcm.infer_exp_rho( 'Arsenal', 'Southampton' )

    # unite_matrix = dcm.infer_prob_matrix( 'Man City', 'Huddersfield' , 4 )

    csv_list = [
        '1112_E0.csv',
        '1011_E0.csv',
        '1213_E0.csv',
        '1314_E0.csv',
        '1415_E0.csv',
        '1516_E0.csv',
        '1617_E0.csv',
        '1718_E0.csv',
        '1819_E0.csv',
        '1920_E0.csv'
    ]
    import time

    csv_list = [os.path.join(os.path.abspath('../../'), c) for c in csv_list]

    ds = DataE0(source=csv_list)
    # ds.split_ds_by_time( y=2013, m=3, d=1, epsilon=0.0065, train_range=400 )

    # print(" ")
    # train_dict = ds.fetch_decay_train_dict( 2011, 8, 17 )
    # print(" ")

    dcm = DixonColesModel_v3(ds)
    t1 = time.time()
    # dcm.inverse_decayed_model( y=2013, m=3, d=1, epsilon=0.0065, range=400 )

    # dcm.inverse_time_varying_model( dir_name='model_dir' ,
    #                                 inverse_time_step=7 )
    # dcm.fetch_model_into_dataframe( 'model_dir' )
    # dcm.solve( options={'disp': True, 'maxiter': 100} )
    # dcm.solve()


    df = pd.read_csv( 'test.csv' )
    df['Date'] = df['Date'].map( lambda s: pd.to_datetime(s, format='%Y-%m-%d') )
    dcm.backtesting_bet_value( df )
    # dcm.backtesting_random_bet()
    # home_exp, away_exp, rho = dcm.infer_expectation_rho( df,
    #                                                      "Arsenal",
    #                                                      "Chelsea",
    #                                                      pd.Timestamp(2011,8,20) )
    #
    # print( home_exp )
    # print( away_exp )
    # print( rho )
    # dcm.view_time_varying_model( df )

    # dcm.load_model( 'ttt.model' )

    # print( dcm.model.x )
    print('Time : ', time.time() - t1)

    # dcm.rmse()
