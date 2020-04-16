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
    def __init__(self, sup_ttg , present_score, adj_params , dim=16 ):
        self.sup_ttg = sup_ttg
        self.present_score = present_score
        self.adj_params = adj_params
        self.dim = dim
        self._calculate_all()

    def refresh(self, **kwargs ):
        self.sup_ttg = kwargs.get( 'sup_ttg', self.sup_ttg )
        self.present_score  = kwargs.get( 'present_socre', self.present_score )
        self.adj_params = kwargs.get( 'adj_params', self.adj_params )
        self.dim = kwargs.get( 'dim', self.dim )

        self._calculate_all()

    def _calculate_all(self):
        self.home_exp = (self.sup_ttg[1] + self.sup_ttg[0]) / 2.
        self.away_exp = (self.sup_ttg[1] - self.sup_ttg[0]) / 2.
        self.present_diff = self.present_score[0] - self.present_score[1]
        self.present_sum  = self.present_score[0] + self.present_score[1]

        self.home_goal_prob = np.array([ poisson.pmf( i, self.home_exp ) for i in range(self.dim) ])
        self.away_goal_prob = np.array([ poisson.pmf( i, self.away_exp ) for i in range(self.dim) ])
        self._M = np.outer( self.home_goal_prob, self.away_goal_prob )

        #TODO only support rho ajustment now
        if self.adj_params[0] == 1:
            rho = self.adj_params[1]
            self._M[0, 0] *= 1 - self.home_exp * self.away_exp * rho
            self._M[0, 1] *= 1 + self.home_exp * rho
            self._M[1, 0] *= 1 + self.away_exp * rho
            self._M[1, 1] *= 1 - rho

        self.tril_dict = {}
        self.triu_dict = {}
        self.diag_dict = {}
        for i in range( -self.dim+1, self.dim ):
            self.tril_dict[i] = np.tril( self._M, i ).sum()
            self.triu_dict[i] = np.triu( self._M, i ).sum()
            self.diag_dict[i] = np.diag( self._M, i ).sum()


    def had(self):
        return { selection_type.HOME: round(np.tril( self._M, -1+self.present_diff ).sum() , 5),
                 selection_type.DRAW: round(np.diag( self._M, self.present_diff ).sum(), 5),
                 selection_type.AWAY: round(np.triu( self._M,  1+self.present_diff).sum(), 5) }

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
        DN_FIRST = 3
        UP_FIRST = 4

    @staticmethod
    def _calculate_critical_line( line ):
        mc = math.ceil(line) - line

        if mc < 0.1 and mc > -0.1:
            return line, DynamicOddsCal.AsianType.NORMALIZE
        if mc < 0.6 and mc > 0.4:
            return line, DynamicOddsCal.AsianType.DIRECT
        if mc < 0.3 and mc > 0.2:
            return math.ceil(line), DynamicOddsCal.AsianType.DN_FIRST
        if mc < 0.8 and mc > 0.7:
            return math.floor(line), DynamicOddsCal.AsianType.UP_FIRST

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
        elif t == DynamicOddsCal.AsianType.DN_FIRST:
            home_margin = self.tril_dict[l-1] / ( 1. - 0.5 * self.diag_dict[l] )
            away_margin = 1. - home_margin
        elif t == DynamicOddsCal.AsianType.UP_FIRST:
            away_margin = self.triu_dict[l+1] / ( 1. - 0.5 * self.diag_dict[l] )
            home_margin = 1. - away_margin
        elif t == DynamicOddsCal.AsianType.DIRECT:
            home_margin = self.tril_dict[ math.floor(line) ]
            away_margin = 1. - home_margin
        else:
            raise Exception

        return { selection_type.HOME: round( home_margin, 5 ),
                 selection_type.AWAY: round( away_margin, 5 ) }

    def over_under(self, line):
        """
        similar to asianhandicap,
        Homewin <----> under
        Awaywin <----> over
        """
        # rotate the probability matrix first
        rotated_matrix = np.transpose( self._M )
        rotated_matrix = rotated_matrix[::-1]

        net_line = line - self.present_sum

        if net_line <= -0.5:
            # definitely happen
            return { selection_type.OVER:  round( 1. , 5 ),
                     selection_type.UNDER: round( 0. , 5 )}

        if net_line > -0.5 and net_line < 0.5 :
            # net_line in [-0.25, 0, 0.25 ]
            # considered error param
            # because goal sum = 0 means cannot betting on this
            # if outcome = 0 , get money back,
            # if outcome > 0 , win
            return { selection_type.OVER:  round( 0. , 5 ),
                     selection_type.UNDER: round( 0. , 5 )}

        l, t = DynamicOddsCal._calculate_critical_line( net_line )

        if t == DynamicOddsCal.AsianType.NORMALIZE:
            under_margin = np.tril( rotated_matrix, -1 + net_line - (self.dim - 1 ) ).sum()
            over_margin  = np.triu( rotated_matrix,  1 + net_line - (self.dim - 1 ) ).sum()

            under_margin, over_margin = under_margin / (under_margin + over_margin), \
                                       over_margin / (under_margin + over_margin)

        elif t == DynamicOddsCal.AsianType.DN_FIRST:
            under_margin = np.tril( rotated_matrix, l - (self.dim - 1 ) - 1 ).sum() / \
                           ( 1. - 0.5 * np.diag( rotated_matrix, l - (self.dim - 1 ) ).sum() )
            over_margin = 1. - under_margin
        elif t == DynamicOddsCal.AsianType.UP_FIRST:
            over_margin = np.triu( rotated_matrix, l - (self.dim - 1 ) + 1 ).sum() / \
                          ( 1. - 0.5 * np.diag( rotated_matrix, l - (self.dim - 1 ) ).sum() )
            under_margin = 1. - over_margin

        elif t == DynamicOddsCal.AsianType.DIRECT:
            under_margin = np.tril( rotated_matrix, math.floor(net_line) - (self.dim -1 ) ).sum()
            over_margin = 1. - under_margin

        else:
            raise Exception

        return { selection_type.OVER:  round( over_margin , 5 ),
                 selection_type.UNDER: round( under_margin , 5 )}

    def over_under_home(self, line ):
        # only consider goal of home
        net_line = line - self.present_score[0]

        if net_line <= -0.5:
            # definitely happen
            return { selection_type.OVER:  round( 1. , 5 ),
                     selection_type.UNDER: round( 0. , 5 )}

        if net_line > -0.5 and net_line < 0.5 :
            # net_line in [-0.25, 0, 0.25 ]
            # considered error param
            # because goal sum = 0 means cannot betting on this
            # if outcome = 0 , get money back,
            # if outcome > 0 , win
            return { selection_type.OVER:  round( 0. , 5 ),
                     selection_type.UNDER: round( 0. , 5 )}

        l, t = DynamicOddsCal._calculate_critical_line( net_line )
        # in this 1D case, the return l is useless
        # and t is just an indicator

        if t == DynamicOddsCal.AsianType.DIRECT:
            home_under_margin = self.home_goal_prob[ : math.ceil(net_line) ].sum()
            home_over_margin = 1. - home_under_margin

        elif t == DynamicOddsCal.AsianType.DN_FIRST:
            home_under_margin = self.home_goal_prob[ : math.ceil(net_line)].sum() / \
                                ( 1. - 0.5 * self.home_goal_prob[math.ceil(net_line)] )
            home_over_margin = 1. - home_under_margin
        elif t == DynamicOddsCal.AsianType.UP_FIRST:
            home_over_margin =  self.home_goal_prob[ math.floor(net_line) +1 : ].sum() / \
                                ( 1. - 0.5 * self.home_goal_prob[math.floor(net_line)] )
            home_under_margin = 1. - home_over_margin
        elif t == DynamicOddsCal.AsianType.NORMALIZE:
            home_under_margin = self.home_goal_prob[ : int(net_line) ].sum() / \
                                ( 1. - self.home_goal_prob[int(net_line)] )
            home_over_margin  = self.home_goal_prob[ int(net_line)+1 : ].sum() / \
                                ( 1. - self.home_goal_prob[int(net_line)] )
        else:
            raise Exception

        return { selection_type.OVER:  round( home_over_margin, 5),
                 selection_type.UNDER: round( home_under_margin, 5)}

    def over_under_away(self, line ):
        # only consider goal of away
        # exactly the same logic with over_under_home
        net_line = line - self.present_score[1]

        if net_line <= -0.5:
            # definitely happen
            return { selection_type.OVER:  round( 1. , 5 ),
                     selection_type.UNDER: round( 0. , 5 )}

        if net_line > -0.5 and net_line < 0.5 :
            # net_line in [-0.25, 0, 0.25 ]
            # considered error param
            # because goal sum = 0 means cannot betting on this
            # if outcome = 0 , get money back,
            # if outcome > 0 , win
            return { selection_type.OVER:  round( 0. , 5 ),
                     selection_type.UNDER: round( 0. , 5 )}

        l, t = DynamicOddsCal._calculate_critical_line( net_line )
        # in this 1D case, the return l is useless
        # and t is just an indicator

        if t == DynamicOddsCal.AsianType.DIRECT:
            away_under_margin = self.away_goal_prob[ : math.ceil(net_line) ].sum()
            away_over_margin = 1. - away_under_margin

        elif t == DynamicOddsCal.AsianType.DN_FIRST:
            away_under_margin = self.away_goal_prob[ : math.ceil(net_line)].sum() / \
                                ( 1. - 0.5 * self.away_goal_prob[math.ceil(net_line)] )
            away_over_margin = 1. - away_under_margin
        elif t == DynamicOddsCal.AsianType.UP_FIRST:
            away_over_margin =  self.away_goal_prob[ math.floor(net_line) +1 : ].sum() / \
                                ( 1. - 0.5 * self.away_goal_prob[math.floor(net_line)] )
            away_under_margin = 1. - away_over_margin
        elif t == DynamicOddsCal.AsianType.NORMALIZE:
            away_under_margin = self.away_goal_prob[ : int(net_line) ].sum() / \
                                ( 1. - self.away_goal_prob[int(net_line)] )
            away_over_margin  = self.away_goal_prob[ int(net_line)+1 : ].sum() / \
                                ( 1. - self.away_goal_prob[int(net_line)] )
        else:
            raise Exception

        return { selection_type.OVER:  round( away_over_margin, 5),
                 selection_type.UNDER: round( away_under_margin, 5)}

    def exact_score(self, home, away ):
        #TODO need to fix the outofrange problem , definitely a bug
        return {selection_type.YES:
                    round(self._M[ home - self.present_score[0], away - self.present_score[1]], 5)}

    def exact_ttg(self, ttg ):
        rotated_matrix = np.transpose( self._M )
        rotated_matrix = rotated_matrix[::-1]

        net_ttg = ttg - self.present_sum
        return {selection_type.YES:
                    round(np.diag( rotated_matrix, net_ttg - (self.dim -1) ).sum(), 5)}

    def exact_ttg_home(self, ttg ):
        #TODO need to fix the outofrange problem , definitely a bug
        net_ttg = ttg - self.present_score[0]
        return {selection_type.YES: round( self.home_goal_prob[net_ttg], 5)}

    def exact_ttg_away(self, ttg ):
        #TODO need to fix the outofrange problem , definitely a bug
        net_ttg = ttg - self.present_score[1]
        return {selection_type.YES: round( self.away_goal_prob[net_ttg], 5)}

    def winning_by_home(self, winning_by ):
        net_wb = winning_by - self.present_diff
        return {selection_type.YES: round( np.diag( self._M, -net_wb ).sum(), 5)}

    def winning_by_away(self, winning_by ):
        net_wb = winning_by + self.present_diff
        return {selection_type.YES: round( np.diag( self._M, net_wb ).sum(), 5)}

    def both_scored(self):
        pos_prob = self.over_under_home(0.5)[ selection_type.OVER ] * \
            self.over_under_away(0.5)[ selection_type.OVER ]
        neg_prob = 1. - pos_prob

        return { selection_type.YES: round(pos_prob, 5),
                 selection_type.NO:  round(neg_prob, 5)}

    def odd_even_ttg(self):
        odd_prob  = 0.
        even_prob = 0.

        flag = ( self.present_sum%2 == 0 )

        for i in range( self.dim ):
            for j in range( self.dim ):
                if flag:
                    if (i+j)%2 == 0:
                        even_prob += self._M[ i,j ]
                    else:
                        odd_prob  += self._M[ i,j ]
                else:
                    if (i+j)%2 == 1:
                        even_prob += self._M[ i,j ]
                    else:
                        odd_prob  += self._M[ i,j ]

        return { selection_type.ODD:  round(odd_prob, 5),
                 selection_type.EVEN: round(even_prob, 5) }

    def double_chance_over_under(self, line):
        raise NotImplementedError

