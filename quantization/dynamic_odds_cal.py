from scipy.stats import poisson
import numpy as np
from quantization.constants import selection_type
import math
from enum import Enum

class DynamicOddsCal( object ):
    """
    calculate odds in different games
    according to the change of present scores
    """
    def __init__(self, sup_ttg , present_score, adj_params ):
        self.sup_tgg = sup_ttg
        self.present_score = present_score
        self.adj_params = adj_params
        self.dim = 16
        self._refresh()
        self._calculate_all()

    def _refresh(self):
        self.home_exp = (self.sup_tgg[1] + self.sup_tgg[0]) / 2.
        self.away_exp = (self.sup_tgg[1] - self.sup_tgg[0]) / 2.
        self.present_diff = self.present_score[0] - self.present_score[1]
        self.present_sum  = self.present_score[0] + self.present_score[1]

        self.home_goal_prob = np.array([ poisson.pmf( i, self.home_exp ) for i in range(self.dim) ])
        self.away_goal_prob = np.array([ poisson.pmf( i, self.away_exp ) for i in range(self.dim) ])
        self.prob_matrix = np.outer( self.home_goal_prob, self.away_goal_prob )

        #TODO only support rho ajustment now
        if self.adj_params[0] == 1:
            rho = self.adj_params[1]
            self.prob_matrix[0, 0] *= 1 - self.home_exp * self.away_exp * rho
            self.prob_matrix[0, 1] *= 1 + self.home_exp * rho
            self.prob_matrix[1, 0] *= 1 + self.away_exp * rho
            self.prob_matrix[1, 1] *= 1 - rho

    def _calculate_all(self):
        self.tril_dict = {}
        self.triu_dict = {}
        self.diag_dict = {}
        for i in range( -self.dim+1, self.dim ):
            self.tril_dict[i] = np.tril( self.prob_matrix, i ).sum()
            self.triu_dict[i] = np.triu( self.prob_matrix, i ).sum()
            self.diag_dict[i] = np.diag( self.prob_matrix, i ).sum()

        # calculate HAD margin

    def had(self):
        return { selection_type.HOME: round(self.tril_dict[-1+self.present_diff], 5 ),
                 selection_type.DRAW: round(self.diag_dict[self.present_diff], 5),
                 selection_type.AWAY: round(self.triu_dict[1+self.present_diff], 5) }

    def double_chance(self):
        return { selection_type.HOME_OR_DRAW:
                     round(self.tril_dict[-1+self.present_diff] + self.diag_dict[self.present_diff], 5 ),
                 selection_type.AWAY_OR_DRAW:
                     round(self.triu_dict[1+self.present_diff] + self.diag_dict[self.present_diff], 5 ),
                 selection_type.HOME_OR_AWAY:
                     round(self.tril_dict[-1+self.present_diff] + self.triu_dict[1+self.present_diff], 5 ) }

    class AsianType( Enum ):
        DIRECT = 1
        NORMALIZE = 2
        HOME_FIRST = 3
        AWAY_FIRST = 4

    @staticmethod
    def _calculate_critical_line( line ):
        mc = math.ceil(line) - line

        if mc < 0.1 and mc > -0.1:
            return line, DynamicOddsCal.AsianType.NORMALIZE
        if mc < 0.6 and mc > 0.4:
            return line, DynamicOddsCal.AsianType.DIRECT
        if mc < 0.3 and mc > 0.2:
            return math.ceil(line), DynamicOddsCal.AsianType.HOME_FIRST
        if mc < 0.8 and mc > 0.7:
            return math.floor(line), DynamicOddsCal.AsianType.AWAY_FIRST

    def asian_handicap(self, line):
        """
        line could be: { 0, N(+-)0.25, N(+-)0.5, N(+-)0.75  | N belongs to 0,1,2,3,...}

        1st get the odds for every bet, its return expectation should be 1 if investing 1
        2nd calculate the margin according to the odds
        """
        l , t = DynamicOddsCal._calculate_critical_line( line )

        if t == DynamicOddsCal.AsianType.NORMALIZE:
            home_margin = self.tril_dict[ -1 + int(line) ]
            away_margin = self.triu_dict[ 1  + int(line) ]

            home_margin, away_margin = home_margin / (home_margin + away_margin ),\
                                       away_margin / ( home_margin + away_margin )
        elif t == DynamicOddsCal.AsianType.HOME_FIRST:
            home_margin = self.tril_dict[l-1] / ( 1. - 0.5 * self.diag_dict[l] )
            away_margin = 1. - home_margin
        elif t == DynamicOddsCal.AsianType.AWAY_FIRST:
            away_margin = self.triu_dict[l+1] / ( 1. - 0.5 * self.diag_dict[l] )
            home_margin = 1. - away_margin
        elif t == DynamicOddsCal.AsianType.DIRECT:
            home_margin = self.tril_dict[ math.floor(line) ]
            away_margin = 1. - home_margin
        else:
            raise Exception

        return { selection_type.HOME: round( home_margin, 5 ),
                 selection_type.AWAY: round( away_margin, 5 ) }

