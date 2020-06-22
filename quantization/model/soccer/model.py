import pickle
from scipy.optimize import minimize
from scipy.stats import poisson
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from quantization.model.soccer.data import E0Data
import matplotlib.pyplot as plt

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
               np.log(poisson.pmf(home_goal, home_exp) + np.finfo(float).eps) + \
               np.log(poisson.pmf(away_goal, away_exp) + np.finfo(float).eps)

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
        # save _model, _team_index, _team_num simutaneously
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

    def back_testing(self, **kwargs):
        pass

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


def plot_something():
    data = E0Data()
    data.from_csv( caches='../../data_e0_cache.csv')
    sdm = StaticDixonModel()

    time_start = pd.Timestamp( 2011, 7, 15 )
    time_delta = pd.Timedelta( days=365 )

    t = time_start

    l1_model = []
    l2_model = []
    l1_market = []
    l2_market = []
    l1_nmm = []
    l2_nmm = []

    time_list = []
    time_list.append( t )

    for _ in range( 9 ):
        sdm.data_preprocessing( data=data, split_time=t )
        sdm.inverse()

        l1, l2 = sdm._rmse_naive_mean_model( train_or_test='test' )
        l1_nmm.append( l1 )
        l2_nmm.append( l2 )

        l1, l2 = sdm._rmse_model( train_or_test='test' )
        l1_model.append( l1 )
        l2_model.append( l2 )

        l1, l2 = sdm._rmse_market( train_or_test='test' )
        l1_market.append( l1 )
        l2_market.append( l2 )

        t = t + time_delta
        time_list.append( t )

    time_list.pop()

    fig, ax = plt.subplots()
    ax.plot( time_list, l1_nmm , label='naive mean model', marker='o', linestyle=':')
    ax.plot( time_list, l1_market, label='bookie model' , marker='+', linestyle=':')
    ax.plot( time_list, l1_model , label='Dixon model' , marker='*' , linestyle=':')

    plt.xlabel( 'time' )
    plt.ylabel( 'rmse' )

    plt.legend( loc=2 )
    plt.show()

if __name__=='__main__':
    data = E0Data()
    data.from_csv( caches='../../data_e0_cache.csv')
    # df = data.get_df()
    # print( df.head() )

    plot_something()
    # sdm = StaticDixonModel()
    # sdm.data_preprocessing( data=data , split_time=pd.Timestamp(2012, 2, 15 ))
    # sdm.inverse()
    # sdm.save_model( fn='tt.model' )
    # sdm.load_model( fn='tt.model' )

    # home_e, away_e, rho = sdm.forward( home='Arsenal', away='Chelsea' )
    # print( home_e )
    # print( away_e )
    # print( rho )
    # print( sdm.forward_team_strength( team='Newcastle') )
    # print( sdm.forward_prob_matrix(home='Arsenal', away='Newcastle', num=1 ) )

    # print( sdm._rmse_model() )
    # print( sdm._rmse_model( train_or_test='test' ) )
    # print( sdm._rmse_market() )
    # print( sdm._rmse_market( train_or_test='test' ) )
    # print( sdm._rmse_naive_mean_model() )
    # print( sdm._rmse_naive_mean_model( train_or_test='test' ) )
