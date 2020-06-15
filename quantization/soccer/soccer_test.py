import pytest
from pandas import np

from quantization.old.soccer.match_odds import cal_soccer_odds, cal_match_odds

from quantization.constants import *
from quantization.soccer.soccer_api import INTERFACE_infer_soccer_sup_ttg, collect_games_odds, \
    calculate_decayed_sup_ttg, INTERFACE_collect_soccer_odds
from quantization.soccer.soccer_dynamic_odds_cal import DynamicOddsCal, DocConfig
from quantization.soccer.soccer_inversion import InferSoccerConfig


class TestClass( object ):
    rho = [ -0.1285, 10. ]
    score = [
        [ 0, 0 ],
        [ 3, 2 ],
        [ 1, 20 ],
    ]

    sup_ttg = [
        [ 0.5, 2.74 ],
        [ 0, 20 ],
    ]

    line = [ -100.5, -8.25, -7.75, -1.25, -2.5, -0.75, 0, 0.25, 0.75, 1.5, 2.5 , 4.5, 11.75, 100 ]

    target_scores = [[3, 2],
                     [0, 0],
                     [10, 0],
                     [20, 10]
                     ]

    @pytest.mark.skip()
    def test_cal_decay_sup_ttg(self):
        # match.set_value([0.5,2.7],[[0,0],[0,0]],[8,45*60,1*60,3*60],0.88,[1,-0.08])
        stage = 6
        ht_add = 2 * 60
        ft_add = 3 * 60
        running_time = 25 * 60
        decay = 0.88
        mu = [0.5, 2.7]
        rho = -0.08

        x = calculate_decayed_sup_ttg(mu,
                                      stage=stage,
                                      running_time=running_time,
                                      ht_add=ht_add,
                                      ft_add=ft_add,
                                      decay=decay)
        print(x)

        match = cal_match_odds()
        match.set_value(mu,
                        [[0, 0], [0, 0]],
                        [stage, running_time, ht_add, ft_add],
                        decay,
                        [1, -0.08])

        full_time_odds_config = DocConfig()
        full_time_odds_config.sup = x[0][0]
        full_time_odds_config.ttg = x[0][1]
        full_time_odds_config.rho = rho
        full_time_odds_config.scores = [0, 0]
        full_time_odds_config.ahc_line_list = np.arange(-10, 10.25, 0.25)
        full_time_odds_config.hilo_line_list = np.arange(0.5, 15.25, 0.25)
        full_time_odds_config.correct_score_limit = 7
        full_time_odds_config.home_ou_line_list = np.arange(0.5, 10.25, 0.25)
        full_time_odds_config.away_ou_line_list = np.arange(0.5, 10.25, 0.25)

        d1 = collect_games_odds(full_time_odds_config)
        d2 = match.odds_output()[period.SOCCER_FIRST_HALF]

        print(d1[market_type.SOCCER_BOTH_TEAMS_TO_SCORE])
        print(d2[market_type.SOCCER_BOTH_TEAMS_TO_SCORE])

        assert (d1 == d2)

    # @pytest.mark.skip()
    def test_INTERFACE_collect_odds(self):
        # match.set_value([0.5,2.7],[[0,0],[0,0]],[8,45*60,1*60,3*60],0.88,[1,-0.08])
        stage = 4
        ht_add = 2 * 60
        ft_add = 3 * 60
        running_time = 0
        decay = 0.88
        mu = [ 0.5, 2.7 ]
        rho = -0.08

        match = cal_match_odds()
        match.set_value( mu,
                         [[0, 0], [0, 0]],
                         [ stage, running_time, ht_add, ft_add],
                         decay,
                         [1, -0.08])

        r = INTERFACE_collect_soccer_odds(mu,
                                           [[0,0], [0,0]],
                                           [ stage , running_time, ht_add, ft_add],
                                           decay,
                                           -0.08 )

        # d1 = r[period.SOCCER_FIRST_HALF]
        # d2 = match.odds_output()[period.SOCCER_FIRST_HALF]

        d11 = r[period.SOCCER_FULL_TIME]
        # d22 = match.odds_output()[period.SOCCER_FULL_TIME]

        # print( d11[market_type.SOCCER_BOTH_TEAMS_TO_SCORE])
        print( d11[market_type.SOCCER_HOME_TO_WIN_EITHER_HALVE] )
        # print( d2[market_type.SOCCER_BOTH_TEAMS_TO_SCORE])

        # assert ( d1 == d2 )
        # assert ( d11 == d22 )


    @pytest.mark.skip()
    def test_had(self):
        doc_v2 = DynamicOddsCal()
        doc_v1 = cal_soccer_odds()
        for r in TestClass.rho:
            for s in TestClass.score:
                for st in TestClass.sup_ttg:
                    print( r, s, st )
                    doc_v1.set_value( st , s , [1,r] )
                    doc_v2.refresh( sup_ttg=st, present_socre=s, adj_params=[1,r] )
                    assert doc_v1.had() == doc_v2.had()
    @pytest.mark.skip()
    def test_double_chance(self):
        for r in TestClass.rho:
            for s in TestClass.score:
                for st in TestClass.sup_ttg:
                    print( r, s, st )
                    doc_v1 = cal_soccer_odds()
                    doc_v1.set_value(st, s, [1, r])
                    doc_v2 = DynamicOddsCal(st, s, [1, r])
                    assert doc_v1.double_chance() == doc_v2.double_chance()

    @pytest.mark.skip()
    def test_asian_handicap(self):
        doc_v1 = cal_soccer_odds()
        doc_v2 = DynamicOddsCal( [0,0], [0,0], [1,0])
        for r in TestClass.rho:
            for s in TestClass.score:
                for st in TestClass.sup_ttg:
                    for l in TestClass.line:
                        print( r, s, st , l)
                        doc_v1.set_value(st, s, [1, r])
                        doc_v2.refresh( sup_ttg=st, present_socre=s, adj_params=[1,r] )
                        assert doc_v1.asian_handicap( l ) == doc_v2.asian_handicap( l )

    @pytest.mark.skip()
    def test_asian_handicap1(self):
        # doc_v1 = cal_soccer_odds()
        # doc_v2 = DynamicOddsCal( [0,0], [0,0], [1,0])
        doc_v2 = DynamicOddsCal()
        # for r in TestClass.rho:
        #     for s in TestClass.score:
        #         for st in TestClass.sup_ttg:
        #             for l in TestClass.line:
        #                 print( r, s, st , l)
        #                 doc_v1.set_value(st, s, [1, r])
        doc_v2.refresh( sup_ttg=[-0.2, 2.3], present_socre=[1, 0], adj_params=[1,-0.08] )
        print( doc_v2.asian_handicap( 0.25 ) )
                        # assert doc_v1.asian_handicap( l ) == doc_v2.asian_handicap( l )
    @pytest.mark.skip()
    def test_over_under(self):
        doc_v1 = cal_soccer_odds()
        doc_v2 = DynamicOddsCal( [0,0], [0,0], [1,0])
        for r in TestClass.rho:
            for s in TestClass.score:
                for st in TestClass.sup_ttg:
                    for l in TestClass.line:
                        print( r, s, st , l)
                        doc_v1.set_value(st, s, [1, r])
                        doc_v2.refresh( sup_ttg=st, present_socre=s, adj_params=[1,r] )
                        assert doc_v1.over_under( l ) == doc_v2.over_under( l )

    @pytest.mark.skip()
    def test_over_under_home(self):
        doc_v1 = cal_soccer_odds()
        doc_v2 = DynamicOddsCal( [0,0], [0,0], [1,0])
        for r in TestClass.rho:
            for s in TestClass.score:
                for st in TestClass.sup_ttg:
                    for l in TestClass.line:
                        print( r, s, st , l)
                        doc_v1.set_value(st, s, [1, r])
                        doc_v2.refresh( sup_ttg=st, present_socre=s, adj_params=[1,r] )
                        assert doc_v1.home_over_under( l ) == doc_v2.over_under_home( l )

    @pytest.mark.skip()
    def test_over_under_away(self):
        doc_v1 = cal_soccer_odds()
        doc_v2 = DynamicOddsCal( [0,0], [0,0], [1,0])
        for r in TestClass.rho:
            for s in TestClass.score:
                for st in TestClass.sup_ttg:
                    for l in TestClass.line:
                        print( r, s, st , l)
                        doc_v1.set_value(st, s, [1, r])
                        doc_v2.refresh( sup_ttg=st, present_socre=s, adj_params=[1,r] )
                        assert doc_v1.away_over_under( l ) == doc_v2.over_under_away( l )

    @pytest.mark.skip()
    def test_exact_score(self):
        doc_v1 = cal_soccer_odds()
        doc_v2 = DynamicOddsCal( [0,0], [0,0], [1,0])
        for r in TestClass.rho:
            for s in TestClass.score:
                for st in TestClass.sup_ttg:
                    for ts in TestClass.target_scores:
                        doc_v2.refresh( sup_ttg=st, present_socre=s, adj_params=[1,r] )
                        print( 'rho = %f, score = %f, %f , targets = %f, %f,  ssssss= %f' %(\
                            r, s[0], s[1], ts[0], ts[1], doc_v2.exact_score(ts[0], ts[1])[selection_type.YES] ) )

    @pytest.mark.skip()
    def test_exact_ttg(self):
        doc_v1 = cal_soccer_odds()
        doc_v2 = DynamicOddsCal( [0,0], [0,0], [1,0])
        for r in TestClass.rho:
            for s in TestClass.score:
                for st in TestClass.sup_ttg:
                    for l in range( -20, 20 ):
                        print( r, s, st , l)
                        doc_v1.set_value(st, s, [1, r])
                        doc_v2.refresh( sup_ttg=st, present_socre=s, adj_params=[1,r] )
                        assert doc_v1.exact_totals( l ) == doc_v2.exact_ttg( l )

    @pytest.mark.skip()
    def test_exact_ttg_home(self):
        doc_v1 = cal_soccer_odds()
        doc_v2 = DynamicOddsCal( [0,0], [0,0], [1,0])
        for r in TestClass.rho:
            for s in TestClass.score:
                for st in TestClass.sup_ttg:
                    for l in range( -20, 20 ):
                        print( r, s, st , l)
                        doc_v1.set_value(st, s, [1, r])
                        doc_v2.refresh( sup_ttg=st, present_socre=s, adj_params=[1,r] )
                        assert doc_v1.home_exact_totals( l ) == doc_v2.exact_ttg_home( l )

    @pytest.mark.skip()
    def test_exact_ttg_away(self):
        doc_v1 = cal_soccer_odds()
        doc_v2 = DynamicOddsCal( [0,0], [0,0], [1,0])
        for r in TestClass.rho:
            for s in TestClass.score:
                for st in TestClass.sup_ttg:
                    for l in range( -20, 20 ):
                        print( r, s, st , l)
                        doc_v1.set_value(st, s, [1, r])
                        doc_v2.refresh( sup_ttg=st, present_socre=s, adj_params=[1,r] )
                        assert doc_v1.away_exact_totals( l ) == doc_v2.exact_ttg_away( l )

    '''
    def test_home_winning_by(self):
        examp=cal_soccer_odds()
        examp.set_value([0.5,3.7],[1,3],[1,-0.08])
        doc = DynamicOddsCal( [0.5,3.7],[1,3],[1,-0.08] )

        l = list( range( -10, 10, 1 ) )
        for line in l:
            # print( doc.winning_by_home( line ))
            # print( examp.home_winning_by(line))
            assert( doc.winning_by_home( line ) == examp.home_winning_by( line ) )

    def test_away_winning_by(self):
        examp=cal_soccer_odds()
        examp.set_value([0.5,3.7],[1,7],[1,-0.08])
        doc = DynamicOddsCal( [0.5,3.7],[1,7],[1,-0.08] )

        l = list( range( -10, 10, 1 ) )
        for line in l:
            # print( doc.winning_by_home( line ))
            # print( examp.home_winning_by(line))
            assert( doc.winning_by_away( line ) == examp.away_winning_by( line ) )

    def test_both_scored(self):
        examp=cal_soccer_odds()
        examp.set_value([0.5,3.7],[1,0],[1,-0.08])
        doc = DynamicOddsCal( [0.5,3.7],[1,0],[1,-0.08] )

        assert( doc.both_scored() == examp.both_scored() )

    def test_odd_even_ttg(self):
        examp=cal_soccer_odds()
        examp.set_value([0.25,3.7],[1,0],[1,-0.08])
        doc = DynamicOddsCal( [0.25,3.7],[1,0],[1,-0.08] )

        assert( doc.odd_even_ttg()  == examp.odd_even() )
    '''


if __name__ == '__main__':
    config = InferSoccerConfig()
    config.ou_line = 3.5
    config.ahc_line = -0.25
    config.scores = [0, 0]
    config.over_odds = 1.9
    config.under_odds = 1.78
    config.home_odds = 2.1
    config.away_odds = 1.8


    from time import time

    t1 = time()

    x = INTERFACE_infer_soccer_sup_ttg(config)
    # x,y  = INTERFACE_infer_basketball_sup_ttg( c, clock, match_format, decay )
    print(x)


    print('time =:', time() - t1)
    # infer_ttg_sup( config )

    # from sympy import *
    # x = symbols( 'x' )
    # mu = symbols( 'mu' )
    # sigma = symbols( 'sigma')
    #
    # f = ( 1. / (sigma * sqrt( 2* pi) ) ) * exp(-(x-mu) * (x-mu)  / ( 2*sigma*sigma))
    # f2 = ( (x-mu) / (sigma*sigma) )
    # print( integrate( f2*f , mu) )
    #
    # import matplotlib.pyplot as plt

    # y = [ i ** 0.5 for i in np.arange( 1, 0 , -0.01) ]
    #
    # fig, ax = plt.subplots()
    # ax.plot( range(0,100), y )
    # plt.show()
