#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle

pd.set_option( 'display.max_columns', None )

class Dataset( object ):
    def __init__(self, *args, **kwargs ):
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

class UrlData( Dataset ):
    def _fetch(self):
        if 'url' not in self.kwargs.keys():
            raise ValueError('must have url in keywords')

        self.df = pd.read_csv( self.kwargs['url'] )

    def _encode_feature(self, home, away ):
        part1_feature = self.encoder.transform( [[ home, away ]] ).tolist()
        part2_feature = self.encoder.transform( [[ away, home ]] ).tolist()

        return [part1_feature] + [part2_feature] + [1] + [1]

    def _preprocess(self):

        self.encoder = OneHotEncoder( sparse=False )
        self.encoder.fit( self.df[ ['HomeTeam', 'AwayTeam'] ].to_numpy() )

        part1_feature = self.encoder.transform( self.df.loc[ : , ['HomeTeam', 'AwayTeam'] ].to_numpy() ).tolist()
        part2_feature = self.encoder.transform( self.df.loc[ : , ['AwayTeam', 'HomeTeam'] ].to_numpy() ).tolist()

        self.feature = [ [p1] + [p2] + [1] + [1] for p1, p2 in zip( part1_feature, part2_feature ) ]

        self.y = self.df.loc[ : , ['FTHG', 'FTAG'] ].to_numpy()

    def get_train_data(self):
        return self.feature, self.y

    def view(self):
        print( self.df.head() )

class DixonColesModel( object ):
    '''
    ref: Modelling Association Football Scores and Inefficiencies in the Football
    betting Market, 1997, Mark Dixon and Stuart G. Coles

    static model without weighting on match by time
    '''
    def __init__(self, ds ):
        self.ds = ds
        self.X, self.y = self.ds.get_train_data()
        self.dim = len( self.X[0][0] ) // 2

    def _preprocessing(self):
        pass

    @staticmethod
    def _calibration_matrix( home_goal,
                             away_goal,
                             home_exp,
                             away_exp,
                             rho ):
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
    def _likelihood( home_goal, # home goal in the match
                     away_goal, # away goal in the match
                     home_exp, # MLE param.
                     away_exp, # MLE param.
                     rho): # MLE param. calibration coefficient, degrades to Poisson Model when rho = 0

        return np.log( DixonColesModel._calibration_matrix( home_goal, away_goal, home_exp, away_exp, rho )) +\
            np.log( poisson.pmf( home_goal, home_exp ) ) +\
            np.log( poisson.pmf( away_goal, away_exp ) )

    def _objective_values_sum( self, params ):
        train_features, train_y = self.ds.get_train_data()
        obj = 0.

        for isample in range( len(train_features) ):
            X = train_features[isample]
            y = train_y[isample]

            home_exp = np.exp( ( np.array(params[:40]) * np.array(X[0]) ).sum() + params[-1] )
            away_exp = np.exp( ( np.array(params[:40]) * np.array(X[1]) ).sum() )

            obj = obj - DixonColesModel._likelihood( y[0], y[1] , home_exp, away_exp, params[-2] )

        return obj

    def solve(self,
              **kwargs ):

        options = kwargs.get( 'options', { 'disp': True, 'maxiter': 100} )
        constraints = kwargs.get( 'constraints', [ { 'type': 'eq', 'fun': lambda x: sum(x[:self.dim]) -self.dim } ] )

        init_vals = np.concatenate( ( np.random.uniform(0,1, self.dim ),
                                      np.random.uniform(0,-1, self.dim ),
                                      [0.],
                                      [1.]) )

        self.model = minimize( self._objective_values_sum,
                           init_vals,
                           options = options,
                           constraints = constraints,
                           **kwargs )

        return self.model

    def save_model(self, fn ):
        with open( fn , 'wb' ) as ooo:
            pickle.dump( self.model, ooo )

    def load_model(self, fn ):
        with open( fn, 'rb' ) as iii:
            self.model = pickle.load( iii )

    def infer_prob_matrix(self, home, away , num = 10 ):
        # maximum goal of each side = 10
        home_exp, away_exp, rho = self.infer_exp_rho( home, away )
        home_goal_prob = np.array( [ poisson.pmf( g, home_exp ) for g in range( num + 1 ) ] )
        away_goal_prob = np.array( [ poisson.pmf( g, away_exp ) for g in range( num + 1 ) ] )

        calibration_matrix = np.array( [ [self._calibration_matrix( hg, ag, home_exp, away_exp, rho ) \
                                          for ag in range(2) ] for hg in range(2) ] )

        united_prob_matrix = np.outer( home_goal_prob, away_goal_prob )
        united_prob_matrix[:2, :2] = united_prob_matrix[:2, :2] * calibration_matrix

        return united_prob_matrix


    def infer_exp_rho(self, home, away ):
        home_exp = np.exp( ( self.ds.encoder.transform( [[home,away]] ) * \
                             np.array( self.model.x[:2*self.dim] ) ).sum() + self.model.x[-1] )
        away_exp = np.exp( ( self.ds.encoder.transform( [[away,home]] ) * \
                             np.array( self.model.x[:2*self.dim] ) ).sum() )

        return home_exp, away_exp, self.model.x[-2]

class DixonColesModel_bk( object ):
    '''
    ref: Modelling Association Football Scores and Inefficiencies in the Football
    betting Market, 1997, Mark Dixon and Stuart G. Coles

    static model without weighting on march by time
    '''
    def __init__(self):
        self.dataset = pd.read_csv('https://www.football-data.co.uk/mmz4281/1718/E0.csv')
        self.dataset = self.dataset[['HomeTeam','AwayTeam','FTHG','FTAG']]
        self.dataset = self.dataset.rename( columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'} )

        self.teams = np.sort( self.dataset['HomeTeam'].unique() )
        self.team_number = len( self.teams )
        self.attack_coe_team_index_dict = dict( zip( self.teams, range( self.team_number ) ) )
        self.defend_coe_team_index_dict = dict( zip( self.teams, range( self.team_number , 2* self.team_number) ) )

        print( self.dataset.head() )

    def _preprocessing(self):
        pass

    @staticmethod
    def _calibration_matrix( home_goal,
                             away_goal,
                             home_exp,
                             away_exp,
                             rho ):
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
    def _likelihood( home_goal, # home goal in the match
                     away_goal, # away goal in the match
                     home_attack_coe, # MLE param.
                     home_defend_coe, # MLE param.
                     away_attack_coe, # MLE param.
                     away_defend_coe, # MLE param.
                     rho, # MLE param. calibration coefficient, degrades to Poisson Model when rho = 0
                     gamma): # MLE param. home advantages, > 0

        home_exp = np.exp( home_attack_coe + away_defend_coe + gamma )
        away_exp = np.exp( away_attack_coe + home_defend_coe )

        l = np.log( DixonColesModel._calibration_matrix( home_goal, away_goal, home_exp, away_exp, rho )) + \
            np.log( poisson.pmf( home_goal, home_exp ) ) + \
            np.log( poisson.pmf( away_goal, away_exp ) )
        return l

    def _objective_values_sum( self, params ):

        # likelihood_list = [ DixonColesModel._likelihood( r.HomeGoals,
        #                                                  r.AwayGoals,
        #                                                  params[self.attack_coe_team_index_dict[r.HomeTeam]],
        #                                                  params[self.defend_coe_team_index_dict[r.HomeTeam]],
        #                                                  params[self.attack_coe_team_index_dict[r.AwayTeam]],
        #                                                  params[self.defend_coe_team_index_dict[r.AwayTeam]],
        #                                                  params[-2],
        #                                                  params[-1]
        #                                                  ) for r in self.dataset.itertuples() ]
        # return -sum( likelihood_list )

        obj = 0.
        #
        for r in self.dataset.itertuples():
            obj = obj - DixonColesModel._likelihood( r.HomeGoals,
                                                     r.AwayGoals,
                                                     params[self.attack_coe_team_index_dict[r.HomeTeam]],
                                                     params[self.defend_coe_team_index_dict[r.HomeTeam]],
                                                     params[self.attack_coe_team_index_dict[r.AwayTeam]],
                                                     params[self.defend_coe_team_index_dict[r.AwayTeam]],
                                                     params[-2],
                                                     params[-1])
        return obj

    def solve(self,
              options = { 'disp': True, 'maxiter': 100},
              constraints = [ { 'type': 'eq', 'fun': lambda x: sum(x[:20]) -20 } ],
              **kwargs ):

        init_vals = np.concatenate( ( np.random.uniform(0,1,self.team_number),
                                      np.random.uniform(0,-1,self.team_number),
                                      [0.],
                                      [1.]) )

        result = minimize( self._objective_values_sum,
                           init_vals,
                           options = options,
                           constraints = constraints,
                           **kwargs )

        return result

if __name__ == "__main__":
    # kwargs = { 'constraints': {'type': 'eq', 'fun': lambda x: sum(x[:20]) - 20} }

    ds = UrlData( url = './E0.csv' )
    dcm = DixonColesModel( ds )

    dcm.load_model( './EnglandPremierLeague_1718_dcm.model' )
    print( dcm.model.x )

    # home_exp, away_exp, rho = dcm.infer_exp_rho( 'Arsenal', 'Southampton' )

    unite_matrix = dcm.infer_prob_matrix( 'Man City', 'Huddersfield' , 4 )

    print( type(ds.encoder.categories_[0] ))
    print( (ds.encoder.categories_ ))


    # print( home_exp )
    # print( away_exp )
    # print( rho )

    # r = dcm.solve()
    # print( r.x)

    # dcm.save_model( './EnglandPremierLeague_1718_dcm.model' )
