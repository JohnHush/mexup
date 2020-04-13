#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize

pd.set_option( 'display.max_columns', None )

def fetchData():
    epl_1819 = pd.read_csv( 'https://www.football-data.co.uk/mmz4281/1718/E0.csv' )
    print( epl_1819.head() )


class DixonColesModel( object ):
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

        l = np.log( DixonColesModel._calibration_matrix( home_goal, away_goal, home_exp, away_exp, rho )) +\
            np.log( poisson.pmf( home_goal, home_exp ) ) +\
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
    dcm = DixonColesModel()

    r = dcm.solve()
    print( r.x)
