import pickle
from scipy.optimize import minimize
from scipy.stats import poisson
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from quantization.model.soccer.data import E0Data
from quantization.soccer.soccer_dynamic_odds_cal import DynamicOddsCal
from quantization.constants import *
import matplotlib.pyplot as plt
import time
import pymongo
import json

alpha___ = 1.5
beta___ = 0.5

class Model(metaclass=ABCMeta):
    '''
    model class
    '''

    @abstractmethod
    def data_preprocessing(self, **kwargs ):
        pass

    @abstractmethod
    def inverse(self, **kwargs):
        pass

    @abstractmethod
    def forward(self, **kwargs):
        pass

    @abstractmethod
    def back_testing(self, **kwargs):
        pass

    @abstractmethod
    def save_model(self, **kwargs):
        pass

    @abstractmethod
    def load_model(self, **kwargs):
        pass


class StaticDixonModel(Model):

    def __init__(self):
        self._train_dict = None
        self._test_dict = None
        self._team_num = None
        self._team_index = None
        self._model = None

    def forward(self, home=None, away=None):
        if home is None or away is None:
            raise Exception('must set home name and away name')

        if home not in self._team_index.keys() or \
            away not in self._team_index.keys():
            raise Exception('model doesnot contain the team')

        home_ind = self._team_index[home]
        away_ind = self._team_index[away]

        home_exp = np.exp(self._model.x[home_ind] +
                          self._model.x[away_ind + self._team_num] +
                          self._model.x[-1])
        away_exp = np.exp(self._model.x[away_ind] +
                          self._model.x[home_ind + self._team_num])

        return home_exp, away_exp, self._model.x[-2]

    def forward_team_strength(self, team=None):
        # team stength is defined as TEAM_ATTACK - TEAM_DEFENCE
        if team is None or team not in self._team_index.keys():
            raise Exception('check team name')

        team_ind = self._team_index[team]
        return self._model.x[team_ind] - self._model.x[team_ind + self._team_num]

    def forward_prob_matrix(self, home=None, away=None, num=10):
        # maximum goal of each side = 10
        if home is None or away is None or home not in self._team_index.keys() or \
            away not in self._team_index.keys():
            raise Exception('check team name, home: %s , away: %s' %(home, away) )

        if num <1 or num > 30:
            raise Exception('inappropriate num = %d ' % num )

        home_exp, away_exp, rho = self.forward(home=home, away=away)

        home_goal_prob = np.array([poisson.pmf(g, home_exp) for g in range(num + 1)])
        away_goal_prob = np.array([poisson.pmf(g, away_exp) for g in range(num + 1)])

        calibration_matrix = np.array([[self._calibration_matrix(hg, ag, home_exp, away_exp, rho) \
                                        for ag in range(2)] for hg in range(2)])

        united_prob_matrix = np.outer(home_goal_prob, away_goal_prob)
        united_prob_matrix[:2, :2] = united_prob_matrix[:2, :2] * calibration_matrix

        return united_prob_matrix

    def _objective_values_sum(self, params):
        obj = 0.

        for d in self._train_dict:
            home_goal = d['FTHG']
            away_goal = d['FTAG']

            home_ind = self._team_index[d['HomeTeam']]
            away_ind = self._team_index[d['AwayTeam']]

            home_exp = np.exp(params[home_ind] + params[away_ind + self._team_num] + params[-1])
            away_exp = np.exp(params[away_ind] + params[home_ind + self._team_num])

            obj = obj - StaticDixonModel._likelihood(home_goal,
                                                     away_goal,
                                                     home_exp,
                                                     away_exp,
                                                     params[-2] )

        return obj

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

    def _grad(self, params):
        g = np.zeros(len(params))

        for d in self._train_dict:
            home_goal = d['FTHG']
            away_goal = d['FTAG']

            home_ind = self._team_index[d['HomeTeam']]
            away_ind = self._team_index[d['AwayTeam']]

            home_exp = np.exp(params[home_ind] + params[away_ind + self._team_num] + params[-1])
            away_exp = np.exp(params[away_ind] + params[home_ind + self._team_num])

            home_indices = [home_ind, away_ind + self._team_num]
            away_indices = [away_ind, home_ind + self._team_num]

            # accumulate derivative of L/alpha and L/beta of Home
            if home_goal == 0 and away_goal == 0:
                g[home_indices[0]] +=  (home_goal - home_exp + (-home_exp * away_exp * params[-2]) / \
                                            (1 - home_exp * away_exp * params[-2]))
                g[home_indices[1]] += (home_goal - home_exp + (-home_exp * away_exp * params[-2]) / \
                                            (1 - home_exp * away_exp * params[-2]))

                g[-1] += (home_goal - home_exp + (-home_exp * away_exp * params[-2]) / \
                               (1 - home_exp * away_exp * params[-2]))

            elif home_goal == 0 and away_goal == 1:
                g[home_indices[0]] += (home_goal - home_exp + (home_exp * params[-2]) / \
                                            (1 + home_exp * params[-2]))
                g[home_indices[1]] += (home_goal - home_exp + (home_exp * params[-2]) / \
                                            (1 + home_exp * params[-2]))

                g[-1] += (home_goal - home_exp + (home_exp * params[-2]) / \
                               (1 + home_exp * params[-2]))
            else:
                g[home_indices[0]] += (home_goal - home_exp)
                g[home_indices[1]] += (home_goal - home_exp)
                g[-1] += (home_goal - home_exp)

            # accumulate another part
            if home_goal == 0 and away_goal == 0:
                g[away_indices[0]] += (away_goal - away_exp + (-home_exp * away_exp * params[-2]) / \
                                            (1 - home_exp * away_exp * params[-2]))
                g[away_indices[1]] += (away_goal - away_exp + (-home_exp * away_exp * params[-2]) / \
                                            (1 - home_exp * away_exp * params[-2]))
            elif home_goal == 1 and away_goal == 0:
                g[away_indices[0]] += (away_goal - away_exp + (away_exp * params[-2]) / \
                                            (1 + away_exp * params[-2]))
                g[away_indices[1]] += (away_goal - away_exp + (away_exp * params[-2]) / \
                                            (1 + away_exp * params[-2]))
            else:
                g[away_indices[0]] += (away_goal - away_exp)
                g[away_indices[1]] += (away_goal - away_exp)

            if home_goal == 0 and away_goal == 0:
                g[-2] += ((-home_exp * away_exp) / (1 - home_exp * away_exp * params[-2]))
            elif home_goal == 0 and away_goal == 1:
                g[-2] += (home_exp / (1 + home_exp * params[-2]))
            elif home_goal == 1 and away_goal == 0:
                g[-2] += (away_exp / (1 + away_exp * params[-2]))
            elif home_goal == 1 and away_goal == 1:
                g[-2] += (-1 / (1 - params[-2]))
            else:
                pass

        return -1. * g

    @staticmethod
    def _likelihood(home_goal,  # home goal in the match
                    away_goal,  # away goal in the match
                    home_exp,  # MLE param.
                    away_exp,  # MLE param.
                    rho):  # MLE param. calibration coefficient, degrades to Poisson Model when rho = 0

        return np.log(StaticDixonModel._calibration_matrix(\
            home_goal, away_goal, home_exp, away_exp, rho) + np.finfo(float).eps) + \
               home_goal * np.log( home_exp + np.finfo(float).eps) - home_exp - \
               away_exp + away_goal * np.log( away_exp + np.finfo(float).eps )

    def inverse(self, **kwargs):
        options = kwargs.pop('options', {'disp': True, 'maxiter': 100})
        constraints = kwargs.pop('constraints', \
                                 [{'type': 'eq',
                                   'fun': lambda x: sum(x[:self._team_num]) - self._team_num}])

        np.random.seed( 0xCAFFE )
        init_vals = np.concatenate((np.random.uniform(0, 1, self._team_num),
                                    np.random.uniform(0, -1, self._team_num),
                                    [0.],
                                    [1.]))

        self._model = minimize( self._objective_values_sum,
                               init_vals,
                               options=options,
                               constraints=constraints,
                               jac=self._grad,
                               **kwargs )

        return self._model

    def save_model(self, fn=None ):
        # save _model, _team_index, _team_num simultaneously
        if fn is None:
            return

        with open(fn, 'wb') as ooo:
            pickle.dump( [self._model, self._team_index, self._team_num], ooo)

    def load_model(self, fn=None ):
        if fn is None:
            return

        with open(fn, 'rb') as iii:
            self._model, self._team_index, self._team_num = pickle.load(iii)

    def _rmse_model(self, train_or_test='train'):
        '''
        compute the RMSE of the model
        '''
        def apply_fn(s):
            try:
                return self.forward(home=s['HomeTeam'], away=s['AwayTeam'] )
            except:
                return 0, 0, 0

        if train_or_test != 'train' and train_or_test != 'test':
            raise Exception('specify train or test')

        if train_or_test=='train':
            df = pd.DataFrame( self._train_dict )
        else:
            df = pd.DataFrame( self._test_dict )

        tmp = df.apply( apply_fn, axis=1 )
        df['lambda1_model'] = tmp.map(lambda s: round(s[0], 5))
        df['lambda2_model'] = tmp.map(lambda s: round(s[1], 5))

        df = df[ (df.lambda1_model != 0 ) & (df.lambda2_model != 0 ) ]

        lambda1_model_error = np.sqrt((df.FTHG - df.lambda1_model) *\
                                      (df.FTHG - df.lambda1_model)).mean()
        lambda2_model_error = np.sqrt((df.FTAG - df.lambda2_model) *\
                                      (df.FTAG - df.lambda2_model)).mean()

        return lambda1_model_error, lambda2_model_error

    def _rmse_naive_mean_model(self, train_or_test='train'):
        '''
        use the mean of the FTHG and FTAG as the predicted goal of each match
        '''
        if train_or_test != 'train' and train_or_test != 'test':
            raise Exception('specify train or test')

        if train_or_test=='train':
            df = pd.DataFrame( self._train_dict )
        else:
            df = pd.DataFrame( self._test_dict )

        df['lambda1_nmm'] = df.FTHG.mean()
        df['lambda2_nmm'] = df.FTAG.mean()

        lambda1_nmm_error = np.sqrt((df.FTHG - df.lambda1_nmm) * \
                                    (df.FTHG - df.lambda1_nmm)).mean()
        lambda2_nmm_error = np.sqrt((df.FTAG - df.lambda2_nmm) * \
                                    (df.FTAG - df.lambda2_nmm)).mean()

        return lambda1_nmm_error, lambda2_nmm_error

    def _rmse_market(self, train_or_test='train'):
        '''
        compute the RMSE of the market
        '''
        if train_or_test != 'train' and train_or_test != 'test':
            raise Exception('specify train or test')

        if train_or_test=='train':
            df = pd.DataFrame( self._train_dict )
        else:
            df = pd.DataFrame( self._test_dict )

        df['lambda1_market'] = (df.sup_market + df.ttg_market) / 2
        df['lambda2_market'] = (-df.sup_market + df.ttg_market) / 2

        df = df[ (df.lambda1_market != 0 ) & (df.lambda2_market != 0 ) ]

        lambda1_market_error = np.sqrt((df.FTHG - df.lambda1_market) * \
                                      (df.FTHG - df.lambda1_market)).mean()
        lambda2_market_error = np.sqrt((df.FTAG - df.lambda2_market) * \
                                      (df.FTAG - df.lambda2_market)).mean()
        return lambda1_market_error, lambda2_market_error


    def back_testing_random_model(self,
                                  train_or_test='train',
                                  ratio = 0.2 ):

        if train_or_test != 'train' and train_or_test != 'test':
            raise Exception('specify train or test')

        if train_or_test=='train':
            df = pd.DataFrame( self._train_dict )
        else:
            df = pd.DataFrame( self._test_dict )

        def apply_betValue_random(s, ratio=ratio):
            over_odds = s['BbAv>2.5']
            under_odds = s['BbAv<2.5']
            home_outcome = s['FTHG']
            away_outcome = s['FTAG']

            if s['HomeTeam'] not in self._team_index.keys() or \
                    s['AwayTeam'] not in self._team_index.keys():
                return 0

            is_bet  = np.random.rand() <= ratio
            is_over = np.random.rand() >= 0.5

            if not is_bet:
                return 0

            if is_over:
                # bet on over 2.5
                if home_outcome + away_outcome > 2.5:
                    return over_odds-1
                return -1

            if home_outcome + away_outcome < 2.5:
                return under_odds -1
            return -1

        df['pnl_random'] = df.apply( apply_betValue_random, axis=1 )
        return df.pnl_random.sum() / len( df[df.pnl_random != 0 ])


    def back_testing(self,
                     train_or_test='train',
                     lo=1.0,
                     hi=100,
                     output_df=None ):
        """

        Args:
            train_or_test: backtesting Trainset or Testset
            lo: the low limit when dismissing the bet
            hi: the high limit
            output_df: output filename

        Returns:

        """
        if train_or_test != 'train' and train_or_test != 'test':
            raise Exception('specify train or test')

        if train_or_test=='train':
            df = pd.DataFrame( self._train_dict )
        else:
            df = pd.DataFrame( self._test_dict )

        def apply_betValue_model(s, lo, hi ):
            over_odds = s['BbAv>2.5']
            under_odds = s['BbAv<2.5']
            home_outcome = s['FTHG']
            away_outcome = s['FTAG']

            try:
                home_exp, away_exp, rho = self.forward( home=s['HomeTeam'], away=s['AwayTeam'] )
            except:
                return 0,0,0

            doc = DynamicOddsCal(sup_ttg=[home_exp-away_exp, home_exp+away_exp],
                                 present_socre=[0, 0],
                                 adj_params=[1, rho])

            d = doc.over_under(line=2.5)
            over_bv  = d[selection_type.OVER] * over_odds
            under_bv = d[selection_type.UNDER] * under_odds

            if (over_bv <=lo or over_bv >= hi) and (under_bv <=lo or under_bv>= hi):
                return over_bv, under_bv, 0

            if over_bv > under_bv:
                # bet on over 2.5
                if home_outcome + away_outcome > 2.5:
                    return over_bv, under_bv, over_odds -1
                return over_bv, under_bv, -1

            if home_outcome + away_outcome < 2.5:
                return over_bv, under_bv, under_odds -1
            return over_bv, under_bv, -1

        tmp = df.apply( lambda x: apply_betValue_model(x, lo=lo, hi=hi), axis=1 )

        df['over_bv_model'] = tmp.map( lambda s: s[0] )
        df['under_bv_model'] = tmp.map( lambda s: s[1] )
        df['pnl_model'] = tmp.map( lambda s: s[2] )

        if output_df is not None:
            df.to_csv( output_df, index=False)

        return df


    def data_preprocessing(self, **kwargs ):
        '''
        split the dataset into trainset and testset
        Args:
            **kwargs:

        Returns:

        '''
        data = kwargs.pop('data', None)
        split_time = kwargs.pop( 'split_time', None )
        if data is None:
            raise Exception('must specify data')
        if split_time is None:
            raise Exception('must have split time')

        if kwargs:
            raise TypeError('unexpected kwargs')

        df = data.get_df()

        # split_time checking
        df_time_start = df.loc[0, 'Date']
        df_time_end = df.loc[len(df)-1, 'Date']

        if split_time>=df_time_end or split_time<=df_time_start:
            raise Exception('split time out of range')

        df_train = df[df.Date < split_time].copy()
        df_test = df[df.Date >= split_time].copy()

        # build team to index mapping
        # only map the team in the match and satisfy some conditions
        team2index = {}
        team_set = set( df_train['HomeTeam'] ) | set( df_train['AwayTeam'] )

        for team in team_set:
            goal_in_toal = df_train.loc[ df_train.HomeTeam==team, 'FTHG'].sum() +\
                           df_train.loc[ df_train.AwayTeam==team, 'FTAG'].sum()

            goal_conceded= df_train.loc[ df_train.HomeTeam==team, 'FTAG'].sum() +\
                           df_train.loc[ df_train.AwayTeam==team, 'FTHG'].sum()

            team_match_times = len( df_train[df_train.HomeTeam==team] ) + \
                               len( df_train[df_train.AwayTeam==team] )

            if goal_in_toal==0 or goal_conceded==0 or team_match_times <= 5:
                df_train = df_train.drop( df_train[(df_train.HomeTeam==team)| \
                                                   (df_train.AwayTeam==team)].index )

        team_set = set( df_train['HomeTeam'] ) | set( df_train['AwayTeam'] )
        team_list = sorted(list(team_set))
        for index, value in enumerate(team_list):
            team2index[value] = index

        self._team_num = len(team2index.keys())
        self._team_index = team2index

        # for k,v in team2index.items():
        #     print( k, v )

        self._train_dict = df_train.to_dict(orient='records')
        self._test_dict = df_test.to_dict(orient='records')


class DynamicDixonModel(Model):
    def __init__(self):
        self._team_index = None
        self._team_num = None
        self._df = None
        self._df_dict = None
        self._team_showname_list = None
        self._model = None

    def back_testing(self, **kwargs):
        pass

    @staticmethod
    def _grad(params, *args):
        team_index = args[1]
        team_num = args[2]

        g = np.zeros(len(params))

        for d in args[0]:
            home_goal = d['FTHG']
            away_goal = d['FTAG']

            home_ind = team_index[d['HomeTeam']]
            away_ind = team_index[d['AwayTeam']]

            home_exp = np.exp(params[home_ind] + params[away_ind + team_num] + params[-1])
            away_exp = np.exp(params[away_ind] + params[home_ind + team_num])

            sample_grad1 = home_goal - home_exp
            sample_grad2 = away_goal - away_exp

            rho = params[-2]

            if home_goal == 0 and away_goal == 0:
                keyC = (-home_exp* away_exp) / (1- home_exp* away_exp* rho)
                sample_grad1 += keyC* rho
                sample_grad2 += keyC* rho
            elif home_goal == 0 and away_goal == 1:
                keyC = home_exp / (1 + home_exp * rho)
                sample_grad1 += keyC* rho
            elif home_goal == 1 and away_goal == 0:
                keyC = away_exp / (1 + away_exp * rho)
                sample_grad2 += keyC* rho
            elif home_goal == 1 and away_goal == 1:
                keyC = -1 / (1 - rho)
            else:
                keyC = 0

            g[-2] += keyC * d['fading']
            g[-1] += sample_grad1 * d['fading']
            g[home_ind] += sample_grad1 * d['fading']
            g[away_ind + team_num] += sample_grad1 * d['fading']
            g[away_ind] += sample_grad2 * d['fading']
            g[home_ind + team_num] += sample_grad2 * d['fading']

        return -g

    @staticmethod
    def _objective_values_sum(params, *args):
        team_index = args[1]
        team_num = args[2]

        obj = 0.
        for d in args[0]:
            home_goal = d['FTHG']
            away_goal = d['FTAG']

            home_ind = team_index[d['HomeTeam']]
            away_ind = team_index[d['AwayTeam']]

            fading_factor = d['fading']

            home_exp = np.exp(params[home_ind] + params[away_ind + team_num] + params[-1])
            away_exp = np.exp(params[away_ind] + params[home_ind + team_num])

            obj = obj - DynamicDixonModel._likelihood(home_goal,
                                                      away_goal,
                                                      home_exp,
                                                      away_exp,
                                                      params[-2]) * fading_factor

        return obj

    @staticmethod
    def _likelihood(home_goal,  # home goal in the match
                    away_goal,  # away goal in the match
                    home_exp,  # MLE param.
                    away_exp,  # MLE param.
                    rho):  # MLE param. calibration coefficient, degrades to Poisson Model when rho = 0

        return np.log(DynamicDixonModel._calibration_matrix(home_goal, away_goal, home_exp, away_exp, rho) + \
                      np.finfo(float).eps) + \
               home_goal * np.log( home_exp + np.finfo(float).eps) - home_exp - \
               away_exp + away_goal * np.log( away_exp + np.finfo(float).eps )

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

    def _fetch_data(self,
                    y=1989,
                    m=6,
                    d=4,
                    epsilon=0.0018571,
                    duration=730 ):
        start_date = pd.Timestamp(y, m, d) - pd.Timedelta(days=duration)

        df_train = self._df[(self._df.Date < pd.Timestamp(y, m, d)) & \
                            (self._df.Date > start_date)].copy()

        if len(df_train) == 0:
            return None

        team_set = set( df_train['HomeTeam'] ) | set( df_train['AwayTeam'] )
        for team in team_set:
            goal_in_toal = df_train.loc[ df_train.HomeTeam==team, 'FTHG'].sum() +\
                           df_train.loc[ df_train.AwayTeam==team, 'FTAG'].sum()

            goal_conceded= df_train.loc[ df_train.HomeTeam==team, 'FTAG'].sum() +\
                           df_train.loc[ df_train.AwayTeam==team, 'FTHG'].sum()

            team_match_times = len(df_train[df_train.HomeTeam == team]) + \
                               len(df_train[df_train.AwayTeam == team])

            if goal_in_toal==0 or goal_conceded==0 or team_match_times <= 5:
                df_train = df_train.drop( df_train[(df_train.HomeTeam==team)|\
                                                   (df_train.AwayTeam==team)].index )

        df_train['fading'] = df_train['Date'].map(
            lambda s: np.exp(-epsilon * (pd.Timestamp(y, m, d) - s) / pd.Timedelta(days=1)))

        return df_train

    def _solve(self,
               y=1989,
               m=6,
               d=4,
               epsilon=0.0018571,
               duration=730,
               init_vals=None,
               **kwargs ):

        t1 = time.time()
        train_data = self._fetch_data( y=y,
                                       m=m,
                                       d=d,
                                       epsilon=epsilon,
                                       duration=duration )

        train_dict = train_data.to_dict(orient='records')
        team2index = {}
        index2team = {}
        team_set = set(train_data['HomeTeam']) | set(train_data['AwayTeam'])
        team_list = sorted(list(team_set))
        for index, value in enumerate(team_list):
            team2index[value] = index
            index2team[index] = value

        team_num = len(team2index.keys())

        np.random.seed( 0xCAFFE )

        if init_vals is None:
            init_vals = np.concatenate((np.random.uniform(0, 1 , team_num),
                                        np.random.uniform(0, -1, team_num),
                                        [0.],
                                        [1.]))

        options = kwargs.pop('options', {'disp': True, 'maxiter': 100} )
        constraints = kwargs.pop('constraints',
                                 [{'type': 'eq',
                                   'fun': lambda x: sum(x[:team_num]) - team_num}])

        model = minimize( DynamicDixonModel._objective_values_sum,
                          init_vals,
                          options=options,
                          constraints=constraints,
                          jac=DynamicDixonModel._grad,
                          args=(train_dict, team2index, team_num),
                          **kwargs)

        print( 'time : ', time.time() - t1 )
        return model, team2index, index2team, team_num

    def inverse(self,
                start_time=None,
                end_time=None,
                epsilon=0.0018571,
                duration=730,
                window_size=30,
                display=True,
                **kwargs ):
        df_time_start = self._df_dict[0]['Date']
        df_time_end = self._df_dict[-1]['Date']

        if start_time is None or start_time < df_time_start + pd.Timedelta(days=duration):
            start_time = df_time_start + pd.Timedelta(days=duration)
        if end_time is None or end_time > df_time_end:
            end_time = df_time_end

        if end_time < start_time:
            raise Exception("end time less than start time!")

        time_step = pd.Timedelta(days=window_size)
        split_time = start_time

        parameter_list = []

        while split_time < end_time:
            local_model, local_team2index, local_index2team, local_team_num = \
                self._solve( y=split_time.year,
                             m=split_time.month,
                             d=split_time.day,
                             epsilon=epsilon,
                             duration=duration,
                             init_vals=None,
                             **kwargs
                             )

            # map the result to model in total
            global_model_x = [np.nan] * ( 2 * self._team_num + 2 )
            global_model_x[-1] = local_model.x[-1]
            global_model_x[-2] = local_model.x[-2]

            for index in range(local_team_num):
                team = local_index2team[index]
                global_index = self._team_index[team]
                global_model_x[global_index] = local_model.x[index]
                global_model_x[global_index + self._team_num] = local_model.x[index + local_team_num]

            global_model_x.insert( 0, split_time )
            parameter_list.append( global_model_x )

            if display:
                print( local_model.x )
                print( split_time )

            split_time += time_step

        self._model = pd.DataFrame( data=parameter_list, columns=self._team_showname_list )

    def save_model(self, fn=None , df_name=None ):
        # save _model, _team_index, _team_num simultaneously
        if fn is None:
            return

        with open(fn, 'wb') as ooo:
            pickle.dump( [self._model, self._team_index, self._team_num], ooo)

        if df_name is not None:
            self._model.to_csv( df_name, index=False )

    def load_model(self, fn=None ):
        if fn is None:
            return

        with open(fn, 'rb') as iii:
            self._model, self._team_index, self._team_num = pickle.load(iii)

    def forward(self,
                home=None,
                away=None,
                match_time=None ):
        if home is None or away is None:
            raise Exception('must set home name and away name')

        if home not in self._team_index.keys() or \
            away not in self._team_index.keys():
            raise Exception('model doesnot contain the team')

        if match_time is None:
            match_time = self._model.iloc[-1, 0 ]

        # filter the model df first
        home_index = self._team_index[home] + 1
        away_index = self._team_index[away] + 1

        filtered_df = self._model[ ~np.isnan(self._model.iloc[:,home_index]) & \
            ~np.isnan(self._model.iloc[:,away_index] ) ]

        filtered_df = filtered_df[ filtered_df.Date < match_time ]

        if len(filtered_df)==0:
            raise Exception('cannot predict this match')

        home_attack = filtered_df.iloc[-1, home_index]
        away_attack = filtered_df.iloc[-1, away_index]
        home_defend = filtered_df.iloc[-1, home_index + self._team_num]
        away_defend = filtered_df.iloc[-1, away_index + self._team_num]
        rho = filtered_df.iloc[-1, -2]
        home_adv = filtered_df.iloc[-1, -1]

        home_exp = np.exp( home_attack + away_defend + home_adv )
        away_exp = np.exp( away_attack + home_defend )

        return home_exp, away_exp, rho

    def _rmse_model(self):
        '''
        compute the RMSE of the model
        '''
        def apply_fn(s):
            try:
                return self.forward(home=s['HomeTeam'], away=s['AwayTeam'], match_time=s['Date'])
            except:
                return 0, 0, 0

        df = self._df

        tmp = df.apply( apply_fn, axis=1 )
        df['lambda1_model'] = tmp.map(lambda s: round(s[0], 5))
        df['lambda2_model'] = tmp.map(lambda s: round(s[1], 5))

        df = df[ (df.lambda1_model != 0 ) & (df.lambda2_model != 0 ) ]

        lambda1_model_error = np.sqrt((df.FTHG - df.lambda1_model) *\
                                      (df.FTHG - df.lambda1_model)).mean()
        lambda2_model_error = np.sqrt((df.FTAG - df.lambda2_model) *\
                                      (df.FTAG - df.lambda2_model)).mean()

        return lambda1_model_error, lambda2_model_error

    def _rmse_naive_mean_model(self):
        '''
        use the mean of the FTHG and FTAG as the predicted goal of each match
        '''

        df = self._df

        df['lambda1_nmm'] = df.FTHG.mean()
        df['lambda2_nmm'] = df.FTAG.mean()

        lambda1_nmm_error = np.sqrt((df.FTHG - df.lambda1_nmm) * \
                                    (df.FTHG - df.lambda1_nmm)).mean()
        lambda2_nmm_error = np.sqrt((df.FTAG - df.lambda2_nmm) * \
                                    (df.FTAG - df.lambda2_nmm)).mean()

        return lambda1_nmm_error, lambda2_nmm_error

    def _rmse_market(self):
        '''
        compute the RMSE of the market
        '''

        df = self._df

        df['lambda1_market'] = (df.sup_market + df.ttg_market) / 2
        df['lambda2_market'] = (-df.sup_market + df.ttg_market) / 2

        df = df[ (df.lambda1_market != 0 ) & (df.lambda2_market != 0 ) ]

        lambda1_market_error = np.sqrt((df.FTHG - df.lambda1_market) * \
                                      (df.FTHG - df.lambda1_market)).mean()
        lambda2_market_error = np.sqrt((df.FTAG - df.lambda2_market) * \
                                      (df.FTAG - df.lambda2_market)).mean()
        return lambda1_market_error, lambda2_market_error

    def data_preprocessing(self, **kwargs ):
        '''
        Args:
            **kwargs:

        Returns:
        '''
        data = kwargs.pop('data', None)
        if data is None:
            raise Exception('must specify data')
        if kwargs:
            raise TypeError('unexpected kwargs')

        df = data.get_df()

        team2index = {}
        team_set = set( df['HomeTeam'] ) | set( df['AwayTeam'] )
        team_list = sorted(list(team_set))

        for index, value in enumerate(team_list):
            team2index[value] = index

        team_showname_list = []
        team_showname_list.append( 'Date')
        for index, value in enumerate(team_list):
            team_showname_list.append( value + '_attack')

        for index, value in enumerate(team_list):
            team_showname_list.append( value + '_defend')

        team_showname_list.append( 'rho' )
        team_showname_list.append( 'home_adv' )

        self._team_showname_list = team_showname_list
        self._team_num = len(team2index.keys())
        self._team_index = team2index
        self._df_dict = df.to_dict( orient='records' )
        self._df = df

        # for k,v in team2index.items():
        #     print( k, v )


class DynamicDixonModelV2(Model):
    '''
    with other statistical infomation added
    '''
    def __init__(self):
        pass

    def back_testing(self, **kwargs):
        pass

    # @staticmethod
    def _grad( self, params, *args ):




        data = self.npd

        data_num = data.shape[0]
        FTHG = data[:,0].astype(np.int16)
        FTAG = data[:,1].astype(np.int16)
        fading = data[:,2]
        home_ind = data[:,3].astype(np.int16)
        away_ind = data[:,4].astype(np.int16)

        cmp_ind = data[:, 5 + 2 * self._aux_num].astype(np.int16)

        league_weight = data[:, 5 + 2 * self._aux_num + 1]
        fading = fading * league_weight

        cmp_home_mean = params[self.local_cmp_sid + cmp_ind]
        cmp_away_mean = params[self.local_cmp_sid + self._league_num + cmp_ind]

        home_attack = params[home_ind]
        away_attack = params[away_ind]
        home_defend = params[home_ind + self.local_team_num]
        away_defend = params[away_ind + self.local_team_num]
        rho = params[2 * self.local_team_num]
        # home_adv = params[2*self.local_team_num + 1]

        home_exp = np.exp(cmp_home_mean + home_attack + away_defend)
        away_exp = np.exp(cmp_away_mean + away_attack + home_defend)


        # home_attack = params[home_ind]
        # away_attack = params[away_ind]
        # home_defend = params[home_ind + team_num]
        # away_defend = params[away_ind + team_num]
        # rho = params[2*team_num]
        # home_adv = params[2*team_num + 1]

        # home_exp = np.exp( cmp_home_mean + home_attack + away_defend + home_adv )
        # away_exp = np.exp( cmp_away_mean + away_attack + home_defend )

        # home_exp_global = np.exp( cmp_home_mean + home_adv )
        # away_exp_global = np.exp( cmp_away_mean )

        g_matrix = np.zeros( (data_num, len(params)) )

        home_base_grad = (FTHG - home_exp) *fading
        away_base_grad = (FTAG - away_exp) *fading
        g_matrix[ np.arange(data_num), home_ind ] += home_base_grad
        g_matrix[ np.arange(data_num), self.local_team_num + away_ind] += home_base_grad
        g_matrix[ np.arange(data_num), self.local_cmp_sid + cmp_ind ] += home_base_grad
        g_matrix[ np.arange(data_num), away_ind ] += away_base_grad
        g_matrix[ np.arange(data_num), self.local_team_num + home_ind] += away_base_grad
        g_matrix[ np.arange(data_num), self.local_cmp_sid + self._league_num + cmp_ind] += away_base_grad
        # g_matrix[ np.arange(data_num), 2*team_num+1 ] += ( FTHG - home_exp ) * fading

        # if self._use_cmp:
        #     g_matrix[ np.arange(data_num), 2*team_num + 2 + 4*self._aux_num + cmp_ind ] += (FTHG - home_exp) * fading
        #     g_matrix[ np.arange(data_num), 2*team_num + 2 + 4*self._aux_num + self._league_num + cmp_ind ] += (FTAG - away_exp) * fading

        # 0:0
        ind = np.where((FTHG==0) & (FTAG==0))
        extra_grad = fading[ind] * (-home_exp[ind] * away_exp[ind] * rho) / (1-home_exp[ind] * away_exp[ind] * rho)
        extra_grad_rho = fading[ind] * (-home_exp[ind] * away_exp[ind]) / (1-home_exp[ind] * away_exp[ind] * rho)

        g_matrix[ ind, home_ind[ind] ] += extra_grad
        g_matrix[ ind, away_ind[ind] + self.local_team_num] += extra_grad
        g_matrix[ ind, self.local_cmp_sid + cmp_ind[ind] ] += extra_grad
        g_matrix[ ind, away_ind[ind] ] += extra_grad
        g_matrix[ ind, home_ind[ind] + self.local_team_num] += extra_grad
        g_matrix[ ind, self.local_cmp_sid + self._league_num + cmp_ind[ind] ] += extra_grad
        g_matrix[ ind, 2*self.local_team_num ] += extra_grad_rho


        # g_matrix[ ind, 2*team_num+1 ] += fading[ind] * (-home_exp[ind] * away_exp[ind] * rho ) / \
        #                                  (1 - home_exp[ind] * away_exp[ind] * rho)
        # g_matrix[ ind, 2*team_num ] += fading[ind] * (-home_exp[ind] *away_exp[ind])/ \
        #                                (1-home_exp[ind] *away_exp[ind] * rho)

        # 0:1
        ind = np.where((FTHG==0) & (FTAG==1))
        extra_grad = fading[ind] * (home_exp[ind] * rho) / (1+ home_exp[ind] * rho)
        extra_grad_rho = fading[ind] * home_exp[ind] / (1+ home_exp[ind] * rho)

        g_matrix[ ind, home_ind[ind] ] += extra_grad
        g_matrix[ ind, away_ind[ind] + self.local_team_num] += extra_grad
        g_matrix[ ind, self.local_cmp_sid + cmp_ind[ind] ] += extra_grad
        g_matrix[ ind, 2*self.local_team_num ] += extra_grad_rho


        # g_matrix[ ind, 2*team_num+1 ] += fading[ind] * ( home_exp[ind] * rho )/\
        #                                  (1+ home_exp[ind] * rho)

        # 1:0
        ind = np.where((FTHG==1) & (FTAG==0))
        extra_grad = fading[ind] * (away_exp[ind] * rho) / (1+ away_exp[ind] * rho)
        extra_grad_rho = fading[ind] * away_exp[ind] / (1+ away_exp[ind] * rho)

        g_matrix[ ind, away_ind[ind] ] += extra_grad
        g_matrix[ ind, home_ind[ind] + self.local_team_num ] += extra_grad
        g_matrix[ ind, self.local_cmp_sid + self._league_num + cmp_ind[ind] ] += extra_grad
        g_matrix[ ind, 2*self.local_team_num ] += extra_grad_rho

        # 1:1
        ind = np.where((FTHG == 1) & (FTAG == 1))
        g_matrix[ ind, 2*self.local_team_num ] += fading[ind] / ( rho -1 )


        # add global
        # if self._use_cmp:
        #     g_matrix[ np.arange(data_num), 2*team_num + 2 + 4*self._aux_num + cmp_ind ] += (FTHG - home_exp_global) * fading * self.glb_weight
        #     g_matrix[ np.arange(data_num), 2*team_num + 2 + 4*self._aux_num + self._league_num + cmp_ind ] += (FTAG - away_exp_global) * fading * self.glb_weight
        #     g_matrix[ np.arange(data_num), 2*team_num+1 ] += (FTHG - home_exp_global) * fading * self.glb_weight

        for index in range( self._aux_num ):
            home_aux = data[:, 5 + index*2 ]
            away_aux = data[:, 6 + index*2 ]
            hsm = params[ self.local_aux_sid + 4*index ]
            asm = params[ self.local_aux_sid + 4*index + 1 ]
            hss = params[ self.local_aux_sid + 4*index + 2 ]
            ass = params[ self.local_aux_sid + 4*index + 3 ]
            home_aux_exp = np.exp( hsm + hss * (home_attack + away_defend) )
            away_aux_exp = np.exp( asm + ass * (away_attack + home_defend) )

            extra_grad_home = (home_aux - home_aux_exp) * fading * self.aux_weight[index] * hss
            extra_grad_away = (away_aux - away_aux_exp) * fading * self.aux_weight[index] * ass

            g_matrix[ np.arange(data_num), home_ind ] += extra_grad_home
            g_matrix[ np.arange(data_num), away_ind + self.local_team_num ] += extra_grad_home
            g_matrix[ np.arange(data_num), away_ind ] += extra_grad_away
            g_matrix[ np.arange(data_num), home_ind + self.local_team_num ] += extra_grad_away

            g_matrix[ np.arange(data_num), self.local_aux_sid + 4*index] += (home_aux - home_aux_exp) * \
                                                                            fading * self.aux_weight[index]
            g_matrix[ np.arange(data_num), self.local_aux_sid + 4*index+1] += ( away_aux - away_aux_exp) * \
                                                                              fading * self.aux_weight[index]
            g_matrix[ np.arange(data_num), self.local_aux_sid + 4*index+2] += (home_aux - home_aux_exp) * \
                                                                              fading * self.aux_weight[index] * \
                                                                              (home_attack + away_defend)
            g_matrix[ np.arange(data_num), self.local_aux_sid + 4*index+3] += (away_aux - away_aux_exp) * \
                                                                              fading * self.aux_weight[index] * \
                                                                              (away_attack + home_defend)

        grad = -g_matrix.sum( axis=0 )
        grad[ : self.local_team_num ] += 2 *self.l2_attack_weight * params[ : self.local_team_num ]
        grad[ self.local_team_num: 2*self.local_team_num ] += 2 * self.l2_defend_weight *\
                                                              params[ self.local_team_num: 2*self.local_team_num ]

        return grad

    def _objective_values_sum(self, params):
        '''
        the data will be stored in Numpy array in a form:
            FTHG, FTAG, fading, home_ind, away_ind, HTHG, HTAG, HST, AST...
        Args:
            params:

        Returns:

        '''
        data = self.npd

        obj = 0.
        data_num = data.shape[0]

        FTHG = data[:,0].astype(np.int16)
        FTAG = data[:,1].astype(np.int16)
        fading = data[:,2]
        home_ind = data[:,3].astype(np.int16)
        away_ind = data[:,4].astype(np.int16)
        cmp_ind = data[:, 5 + 2*self._aux_num ].astype(np.int16)
        league_weight = data[:, 5 + 2*self._aux_num + 1]

        fading = fading * league_weight

        cmp_home_mean = params[ self.local_cmp_sid + cmp_ind ]
        cmp_away_mean = params[ self.local_cmp_sid + self._league_num + cmp_ind ]


        home_attack = params[home_ind]
        away_attack = params[away_ind]
        home_defend = params[home_ind + self.local_team_num]
        away_defend = params[away_ind + self.local_team_num]
        rho = params[ 2*self.local_team_num ]
        # home_adv = params[2*self.local_team_num + 1]

        home_exp = np.exp( cmp_home_mean + home_attack + away_defend )
        away_exp = np.exp( cmp_away_mean + away_attack + home_defend )
        # home_exp = np.exp(  home_attack + away_defend + home_adv)
        # away_exp = np.exp(  away_attack + home_defend )

        # home_exp_global = np.exp( cmp_home_mean + home_adv )
        # away_exp_global = np.exp( cmp_away_mean )

        dixon_coef = np.ones(data_num)
        index1 = np.where( (FTHG==0) & (FTAG==0) )
        index2 = np.where( (FTHG==0) & (FTAG==1) )
        index3 = np.where( (FTHG==1) & (FTAG==0) )
        index4 = np.where( (FTHG==1) & (FTAG==1) )

        dixon_coef[index1] = 1. - home_exp[index1] * away_exp[index1] * rho
        dixon_coef[index2] = 1. + home_exp[index2] * rho
        dixon_coef[index3] = 1. + away_exp[index3] * rho
        dixon_coef[index4] = 1. - rho

        # llh_main = np.log( dixon_coef + np.finfo(float).eps ) + \
        #            FTHG * np.log( home_exp + np.finfo(float).eps) - home_exp + \
        #            FTAG * np.log( away_exp + np.finfo(float).eps) - away_exp
        llh_main = np.log(dixon_coef) + FTHG * np.log(home_exp) - home_exp + FTAG * np.log(away_exp) - away_exp
        llh_main = -llh_main * fading

        obj = obj + llh_main.sum()

        # if self._use_cmp:
            # llh_global = FTHG * np.log( home_exp_global + np.finfo(float).eps) - home_exp_global + \
            #              FTAG * np.log( away_exp_global + np.finfo(float).eps) - away_exp_global
            # llh_global = FTHG * np.log( home_exp_global ) - home_exp_global + \
            #              FTAG * np.log( away_exp_global ) - away_exp_global

            # llh_global = -llh_global * fading * self.glb_weight
            # obj = obj + llh_global.sum()

        for index in range( self._aux_num ):
            home_aux = data[:, 5 + index*2 ]
            away_aux = data[:, 6 + index*2 ]
            hsm = params[ self.local_aux_sid + 4*index ]
            asm = params[ self.local_aux_sid + 4*index + 1 ]
            hss = params[ self.local_aux_sid + 4*index + 2 ]
            ass = params[ self.local_aux_sid + 4*index + 3 ]
            home_aux_exp = np.exp( hsm + hss * (home_attack + away_defend) )
            away_aux_exp = np.exp( asm + ass * (away_attack + home_defend) )

            # llh_aux = home_aux * np.log(home_aux_exp + np.finfo(float).eps) - home_aux_exp + \
            #           away_aux * np.log(away_aux_exp + np.finfo(float).eps) - away_aux_exp
            llh_aux = home_aux * np.log(home_aux_exp) - home_aux_exp + \
                      away_aux * np.log(away_aux_exp) - away_aux_exp

            llh_aux = -llh_aux * fading * self.aux_weight[index]
            obj = obj + llh_aux.sum()

        for index in range( self.local_team_num ):
            obj += params[index] * params[index] * self.l2_attack_weight
            obj += params[index+self.local_team_num] * params[index+self.local_team_num] * self.l2_defend_weight

        return obj

    def _fetch_data(self,
                    y=1989,
                    m=6,
                    d=4,
                    epsilon=0.0018571,
                    duration=730 ):
        start_date = pd.Timestamp(y, m, d) - pd.Timedelta(days=duration)

        df_train = self._df[(self._df.Date < pd.Timestamp(y, m, d)) & \
                            (self._df.Date > start_date)].copy()

        if len(df_train) == 0:
            return None

        team_set = set( df_train['HomeTeam'] ) | set( df_train['AwayTeam'] )
        for team in team_set:
            goal_in_toal = df_train.loc[ df_train.HomeTeam==team, 'FTHG'].sum() + \
                           df_train.loc[ df_train.AwayTeam==team, 'FTAG'].sum()

            goal_conceded= df_train.loc[ df_train.HomeTeam==team, 'FTAG'].sum() + \
                           df_train.loc[ df_train.AwayTeam==team, 'FTHG'].sum()

            team_match_times = len(df_train[df_train.HomeTeam == team]) + \
                               len(df_train[df_train.AwayTeam == team])

            if goal_in_toal==0 or goal_conceded==0 or team_match_times <= 5:
                df_train = df_train.drop( df_train[(df_train.HomeTeam==team)| \
                                                   (df_train.AwayTeam==team)].index )

        df_train['fading'] = df_train['Date'].map(
            lambda s: np.exp(-epsilon * (pd.Timestamp(y, m, d) - s) / pd.Timedelta(days=1)))

        return df_train

    def _solve(self,
               y=1989,
               m=6,
               d=4,
               init_vals=None,
               pre_teams=set(),
               **kwargs ):

        train_data = self._fetch_data( y=y,
                                       m=m,
                                       d=d,
                                       epsilon=self.epsilon,
                                       duration=self.duration )

        team_set = set(train_data['HomeTeam']) | set(train_data['AwayTeam'])
        team_list = sorted(list(team_set))
        self.local_team_num = len(team_list)
        self.local_aux_sid = 2*self.local_team_num + self._dep_num
        self.local_cmp_sid = self.local_aux_sid + 4*self._aux_num

        self.local_team2index = {}
        self.local_index2team = {}
        for index, value in enumerate(team_list):
            self.local_team2index[value] = index
            self.local_index2team[index] = value

        # transfer train_dict into numpy array
        train_data['home_ind'] = train_data['HomeTeam'].map( lambda s: self.local_team2index[s] )
        train_data['away_ind'] = train_data['AwayTeam'].map( lambda s: self.local_team2index[s] )
        train_data['league_ind'] = train_data['Div'].map(lambda s: self._league_index[s])

        col_list = [ 'FTHG', 'FTAG', 'fading', 'home_ind', 'away_ind' ]
        for aux in self._aux_tar:
            col_list.append( aux[0] )
            col_list.append( aux[1] )

        col_list.append( 'league_ind' )
        col_list.append( 'league_weight' )
        self.npd = train_data.loc[:, col_list].values

        c1 = {'type': 'eq', 'fun': lambda x: sum(x[:self.local_team_num]) - self.local_team_num}
        c2 = {'type': 'eq', 'fun': lambda x: sum(x[self.local_team_num : 2*self.local_team_num]) + self.local_team_num}
        options = kwargs.pop('options', {'disp': False, 'maxiter': 200} )
        constraints = kwargs.pop('constraints', [c1,c2] )

        if team_set == pre_teams:
            init_vals = init_vals
        else:
            np.random.seed(0xCAFFE)
            init_vals = np.concatenate((np.random.uniform(0, 1 , self.local_team_num),
                                        np.random.uniform(0, -1, self.local_team_num),
                                        [0.],
                                        [1.5, 1.5, 0.5, 0.5] * self._aux_num,
                                        [0., 0.] * self._league_num
                                        ))

        self.local_model = minimize( self._objective_values_sum,
                                     init_vals,
                                     options=options,
                                     constraints=constraints,
                                     jac=self._grad,
                                     **kwargs)

    def inverse(self,
                start_time=None,
                end_time=None,
                epsilon=0.0018571,
                duration=730,
                window_size=30,
                aux_weight=1.,
                league_weight=1.,
                l2_attack_weight=0.,
                l2_defend_weight=0.,
                display=True,
                **kwargs ):
        """
        the main inverse function, save the final model into a DataFrame
        Args:
            start_time: inverse from
            end_time: inverse to
            epsilon: time decay of data
            duration: time duration of data
            window_size: time gap of data
            aux_weight: the auxiliary targets' weights
            league_weight: the weights of different leagues
            l2_attack_weight: l2 normalization weights of attack params
            l2_defend_weight: l2 normalization weights of defend params
            display: if display switch
            **kwargs:
        """
        df_time_start = self._df['Date'].iloc[0]
        df_time_end = self._df['Date'].iloc[-1]

        self.duration = duration
        self.window_size = window_size
        self.epsilon = epsilon
        self.l2_attack_weight = l2_attack_weight
        self.l2_defend_weight = l2_defend_weight

        self.aux_weight = aux_weight
        if not isinstance( self.aux_weight, list ):
            self.aux_weight = [self.aux_weight] * self._aux_num

        self.league_weight = league_weight
        if isinstance( self.league_weight, list ):
            if len(self.league_weight) != self._league_num:
                raise Exception('league weight wrong')
        else:
            self.league_weight = [self.league_weight] * self._league_num

        self._df['league_weight'] = self._df['Div'].map( lambda s: self.league_weight[self._league_index[s]] )

        if start_time is None or start_time < df_time_start + pd.Timedelta(days=duration):
            start_time = df_time_start + pd.Timedelta(days=duration)
        if end_time is None or end_time > df_time_end:
            end_time = df_time_end

        if end_time < start_time:
            raise Exception("end time less than start time!")

        time_step = pd.Timedelta(days=window_size)
        split_time = start_time

        parameter_list = []
        init_vals = None
        pre_teams = set()

        while split_time < end_time:
            self._solve( y=split_time.year,
                         m=split_time.month,
                         d=split_time.day,
                         init_vals=init_vals,
                         pre_teams=pre_teams,
                         **kwargs
                         )

            init_vals = self.local_model.x
            pre_teams = set(self.local_team2index.keys())

            # map the result to model in total
            global_model_x = [np.nan] * ( self._cmp_sid + 2*self._league_num )
            global_model_x[ 2*self._team_num: ] = self.local_model.x[ 2* self.local_team_num : ]

            for index in range(self.local_team_num):
                team = self.local_index2team[index]
                global_index = self._team_index[team]
                global_model_x[global_index] = self.local_model.x[index]
                global_model_x[global_index + self._team_num] = self.local_model.x[index + self.local_team_num]

            global_model_x.insert( 0, split_time )
            parameter_list.append( global_model_x )

            if display:
                # print( self.local_model.x )
                print( split_time )

            split_time += time_step

        show_name_list = list(map( lambda s: self._params_name[s], range(self._cmp_sid + 2*self._league_num) ))
        show_name_list.insert( 0, 'Date' )
        self._model = pd.DataFrame( data=parameter_list, columns=show_name_list )

    def save_to_mongodb(self,
                        id,
                        db_name,
                        collection_name,
                        host='localhost',
                        port=27017 ):
        client = pymongo.MongoClient(host=host, port=port)
        dbs = client[db_name]
        ddm_collection = dbs[collection_name]

        d = {}
        d['model_id'] = id
        d['model'] = self._model.to_dict('list')

        params_dict = {}
        params_dict['team2index'] = self._team_index
        params_dict['league2index'] = self._league_index
        params_dict['team_num'] = self._team_num
        params_dict['league_num'] = self._league_num
        params_dict['cmp_sid'] = self._cmp_sid

        d['model_aux'] = params_dict

        hyper_params_dict = {}
        hyper_params_dict['epsilon'] = self.epsilon
        hyper_params_dict['duration'] = self.duration
        hyper_params_dict['window_size'] = self.window_size
        hyper_params_dict['aux_weight'] = self.aux_weight
        hyper_params_dict['league_weight'] = self.league_weight
        hyper_params_dict['l2_attack_weight'] = self.l2_attack_weight
        hyper_params_dict['l2_defend_weight'] = self.l2_defend_weight

        d['model_hyper_params'] = hyper_params_dict

        ddm_collection.insert_one(d)

    def load_from_mongodb(self,
                          id,
                          db_name,
                          collection_name,
                          host = 'localhost',
                          port = 27017 ):
        client = pymongo.MongoClient(host=host, port=port)
        dbs = client[db_name]
        ddm_collection = dbs[collection_name]

        d = ddm_collection.find_one( {'model_id': id} )

        self._model = pd.DataFrame.from_dict( d['model'] )
        self._team_index = d['model_aux']['team2index']
        self._league_index = d['model_aux']['league2index']
        self._team_num = d['model_aux']['team_num']
        self._league_num = d['model_aux']['league_num']
        self._cmp_sid = d['model_aux']['cmp_sid']

    def load_from_dict(self, d):
        self._model = pd.DataFrame.from_dict( d['model'] )
        self._team_index = d['model_aux']['team2index']
        self._league_index = d['model_aux']['league2index']
        self._team_num = d['model_aux']['team_num']
        self._league_num = d['model_aux']['league_num']
        self._cmp_sid = d['model_aux']['cmp_sid']

    def save_model(self, fn=None , df_name=None ):
        # save _model, _team_index, _team_num simultaneously
        if fn is None:
            return

        with open(fn, 'wb') as ooo:
            pickle.dump( [self._model, self._team_index, self._team_num], ooo)

        if df_name is not None:
            self._model.to_csv( df_name, index=False )

    def load_model(self, fn=None ):
        if fn is None:
            return

        with open(fn, 'rb') as iii:
            self._model, self._team_index, self._team_num = pickle.load(iii)

    def forward(self,
                home=None,
                away=None,
                div=None,
                match_time=None ):
        if home is None or away is None:
            raise Exception('must set home name and away name')

        if home not in self._team_index.keys() or \
                away not in self._team_index.keys():
            raise Exception('model doesnot contain the team')

        if match_time is None:
            match_time = self._model.iloc[-1, 0 ]

        # filter the model df first
        home_index = self._team_index[home] + 1
        away_index = self._team_index[away] + 1
        div_index = self._league_index[div] + 1

        filtered_df = self._model[ ~np.isnan(self._model.iloc[:,home_index]) & \
                                   ~np.isnan(self._model.iloc[:,away_index] ) ]

        filtered_df = filtered_df[ filtered_df.Date < match_time ]

        if len(filtered_df)==0:
            raise Exception('cannot predict this match')

        home_attack = filtered_df.iloc[-1, home_index]
        away_attack = filtered_df.iloc[-1, away_index]
        home_defend = filtered_df.iloc[-1, home_index + self._team_num]
        away_defend = filtered_df.iloc[-1, away_index + self._team_num]
        rho = filtered_df.iloc[-1, 2* self._team_num +1 ]
        home_cmp = filtered_df.iloc[-1, self._cmp_sid + div_index ]
        away_cmp = filtered_df.iloc[-1, self._cmp_sid + self._league_num + div_index ]

        home_exp = np.exp( home_cmp + home_attack + away_defend )
        away_exp = np.exp( away_cmp + away_attack + home_defend )

        return home_exp, away_exp, rho

    def _rmse_model(self):
        '''
        compute the RMSE of the model
        '''
        def apply_fn(s):
            try:
                return self.forward(home=s['HomeTeam'], away=s['AwayTeam'], div=s['Div'], match_time=s['Date'])
            except:
                return 0, 0, 0

        df = self._df
        df = df[ df.Div=='E0' ]
        df = df.set_index( np.arange(len(df)) )

        tmp = df.apply( apply_fn, axis=1 )
        lambda1_model = tmp.map(lambda s: round(s[0], 5))
        lambda2_model = tmp.map(lambda s: round(s[1], 5))

        df = df[ (lambda1_model != 0 ) & (lambda2_model != 0 ) ]
        print( 'lenght, length,:::: ' , len(df) )

        # print( df.head() )

        # b = self._df.iloc[df.index.tolist(), :]
        # print( b.head() )

        lambda1_model_error = np.sqrt((df.FTHG - lambda1_model) * \
                                      (df.FTHG - lambda1_model)).mean()
        lambda2_model_error = np.sqrt((df.FTAG - lambda2_model) * \
                                      (df.FTAG - lambda2_model)).mean()

        return lambda1_model_error, lambda2_model_error, df.index.tolist()

    def _rmse_naive_mean_model(self):
        '''
        use the mean of the FTHG and FTAG as the predicted goal of each match
        '''

        df = self._df
        df = df[ df.Div=='E0' ]
        df = df.set_index( np.arange(len(df)) )

        lambda1_nmm = df.FTHG.mean()
        lambda2_nmm = df.FTAG.mean()

        lambda1_nmm_error = np.sqrt((df.FTHG - lambda1_nmm) * \
                                    (df.FTHG - lambda1_nmm)).mean()
        lambda2_nmm_error = np.sqrt((df.FTAG - lambda2_nmm) * \
                                    (df.FTAG - lambda2_nmm)).mean()

        return lambda1_nmm_error, lambda2_nmm_error

    def _rmse_market(self, pick_list=None ):
        '''
        compute the RMSE of the market
        '''

        df = self._df
        df = df[ df.Div=='E0' ]
        df = df.set_index( np.arange(len(df)) )

        if pick_list is not None:
            df = df.iloc[pick_list, :]

        lambda1_market = (df.sup_market + df.ttg_market) / 2
        lambda2_market = (-df.sup_market + df.ttg_market) / 2

        # df = df[ (df.lambda1_market != 0 ) & (df.lambda2_market != 0 ) ]

        lambda1_market_error = np.sqrt((df.FTHG - lambda1_market) * \
                                       (df.FTHG - lambda1_market)).mean()
        lambda2_market_error = np.sqrt((df.FTAG - lambda2_market) * \
                                       (df.FTAG - lambda2_market)).mean()
        return lambda1_market_error, lambda2_market_error

    def data_preprocessing(self,
                           data=None,
                           # use_cmp=True,
                           aux_tar=[],
                           **kwargs ):
        '''
        1. decide for the parameter format
        2. prepare proper data for inversion
        3. make crucial variables
        '''
        if data is None:
            raise Exception('must specify data')

        self._df = data.get_df()
        self._params_name = {}

        # analyze team information
        keep_cols = ['Date', 'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        team_list = sorted(list(set(self._df['HomeTeam']) | set(self._df['AwayTeam'])))
        self._team_num = len(team_list)

        self._team_index = {}
        for index, value in enumerate(team_list):
            self._team_index[value] = index
            self._params_name[index] = value + '_attack'
            self._params_name[index + self._team_num] = value + '_defend'

        # rho and home_adv
        self._dep_num = 1
        self._params_name[ 2*self._team_num ] = 'rho'

        # analyze auxiliary targets
        self._aux_tar = aux_tar
        self._aux_num = len(aux_tar)
        self._aux_sid = 2*self._team_num + self._dep_num

        for index in range( self._aux_num ):
            pre_str = 'stat' + str(index) + '_'
            self._params_name[ self._aux_sid + 4*index ] = pre_str + 'home_mean'
            self._params_name[ self._aux_sid + 4*index + 1 ] = pre_str + 'away_mean'
            self._params_name[ self._aux_sid + 4*index + 2 ] = pre_str + 'home_scale'
            self._params_name[ self._aux_sid + 4*index + 3 ] = pre_str + 'away_scale'

            keep_cols.append( self._aux_tar[index][0] )
            keep_cols.append( self._aux_tar[index][1] )

        # analyze league information
        self._cmp_sid = self._aux_sid + 4*self._aux_num

        league_list = sorted(list(set(self._df['Div'])))
        self._league_num = len(league_list)

        self._league_index = {}
        for index, value in enumerate(league_list):
            self._league_index[value] = index
            pre_str = 'league' + str(index) + '_'
            self._params_name[ self._cmp_sid + index ] = pre_str + 'home_mean'
            self._params_name[ self._cmp_sid + self._league_num + index ] = pre_str + 'away_mean'

        keep_cols.append('sup_market')
        keep_cols.append('ttg_market')

        self._df = self._df.loc[:, keep_cols]
        self._model = None


if __name__=='__main__':
    import time
    import sys

    data = E0Data()
    # data.from_csv( caches='../../data_e0_cache.csv' )
    data.from_sql()

    # ddm = DynamicDixonModelV2( aux_tar=[ ['HST', 'AST'], ['HTHG', 'HTAG']])
    ddm = DynamicDixonModelV2()
    ddm.data_preprocessing( data=data,
                            aux_tar=[['HST', 'AST'] ] )


    t1 = time.time()
    # ddm.inverse( window_size=500,
    #              epsilon=0.0018571,
    #              duration=1065,
    #              aux_weight=0.2,
    #              league_weight=[3., 1.],
    #              l2_attack_weight=.03,
    #              l2_defend_weight=.03 )
    #
    # ddm.save_to_mongodb( id=1,
    #                      db_name='all_models_db',
    #                      collection_name='dynamic_dixon_coles_model' )
    #
    ddm.load_from_mongodb( id=1,
                           db_name='all_models_db',
                           collection_name='dynamic_dixon_coles_model' )

    # sys.exit()
    print( 'time : ', time.time() - t1 )
    # ddm.save_model( fn='ddmV2.model', df_name='ddmV2.csv' )

    # ddm.load_model( fn='ddmV2.model' )

    # ddm.forward( home='Chelsea', away='Middlesbrough', match_time=pd.Timestamp( 2019, 3 , 29 ) )

    l1e, l2e, pick_list = ddm._rmse_model()

    # print( ddm._rmse_model() )
    print( l1e, l2e )
    print( ddm._rmse_naive_mean_model())
    print( ddm._rmse_market( pick_list=pick_list) )
    #
    # ddm.forward( home='Sheffield United', away='QPR' )

    # df = pd.read_csv('ddm.csv',
    #                  date_parser=lambda s: pd.datetime.strptime(s, '%Y-%m-%d'),
    #                  infer_datetime_format=True,
    #                  parse_dates=['Date', ])

    # pd.set_option('display.max_columns', 100 )
    # print( df )

    # t1 = time.time()
    #
    # data = E0Data()
    # data.from_csv( caches='../../data_e0_cache.csv')
    # df = data.get_df()
    # print( df.head() )

    # plot_something()
    # plot_back_testing_result()
    # sdm = StaticDixonModel()
    # sdm.data_preprocessing( data=data , split_time=pd.Timestamp(2012, 8, 13 ))
    # sdm.inverse()
    #
    # print( sdm._model.x )
    # sdm.save_model( fn='tt.model' )
    # sdm.load_model( fn='tt.model' )

    # print( sdm._rmse_model( train_or_test='test' ) )
    # print( sdm._rmse_market( train_or_test='test' ) )
    # print( sdm._rmse_naive_mean_model( train_or_test='test' ) )
    # print( sdm.back_testing( train_or_test='train',
    #                   output_df='t.csv') )

    # print( sdm.back_testing( train_or_test='test' ) )

    # sdm.back_testing_random_model( train_or_test='test' , ratio=-1 )

    # print( 'time = %f ' %( time.time() - t1 ) )

    # home_e, away_e, rho = sdm.forward( home='Arsenal', away='Chelsea' )
    # print( home_e )
    # print( away_e )
    # print( rho )
    # print( sdm.forward_team_strength( team='Newcastle') )
    # print( sdm.forward_prob_matrix(home='Arsenal', away='Newcastle', num=1 ) )

    # print( sdm._rmse_model() )
    # print( sdm._rmse_market() )
    # print( sdm._rmse_naive_mean_model() )
