import pytest
from quantization.util import *
from quantization.match_odds import cal_match_odds
import numpy as np

class TestClass( object ):
    def test_cal_decay_sup_ttg(self):
        # match.set_value([0.5,2.7],[[0,0],[0,0]],[8,45*60,1*60,3*60],0.88,[1,-0.08])
        stage = 6
        ht_add = 2 * 60
        ft_add = 3 * 60
        running_time = 25 * 60
        decay = 0.88
        mu = [ 0.5, 2.7 ]
        rho = -0.08

        x = calculate_decayed_sup_ttg( mu,
                                       stage=stage,
                                       running_time=running_time,
                                       ht_add=ht_add,
                                       ft_add=ft_add,
                                       decay=decay )
        print( x )

        match = cal_match_odds()
        match.set_value( mu,
                         [[0, 0], [0, 0]],
                         [ stage , running_time, ht_add, ft_add],
                         decay,
                         [1, -0.08])


        full_time_odds_config = DocConfig()
        full_time_odds_config.sup = x[0][0]
        full_time_odds_config.ttg = x[0][1]
        full_time_odds_config.rho = rho
        full_time_odds_config.scores = [0,0]
        full_time_odds_config.ahc_line_list = np.arange(-10, 10.25, 0.25)
        full_time_odds_config.hilo_line_list = np.arange(0.5, 15.25, 0.25)
        full_time_odds_config.correct_score_limit = 7
        full_time_odds_config.home_ou_line_list = np.arange(0.5, 10.25, 0.25)
        full_time_odds_config.away_ou_line_list = np.arange(0.5, 10.25, 0.25)

        d1 = collect_games_odds( full_time_odds_config )
        d2 = match.odds_output()[period.SOCCER_FIRST_HALF]

        print( d1[market_type.SOCCER_BOTH_TEAMS_TO_SCORE])
        print( d2[market_type.SOCCER_BOTH_TEAMS_TO_SCORE])

        assert ( d1 == d2 )


