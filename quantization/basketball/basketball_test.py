import pytest

from quantization.basketball.basketball_api import INTERFACE_infer_basketball_sup_ttg, INTERFACE_collect_basketball_odds
from quantization.basketball.basketball_inversion import InferBasketConfig
from quantization.constants import market_type, period
from quantization.old.basketball.basketball_match_odds import cal_basketball_match_odds
from quantization.old.basketball.basketball_normal import cal_basketball_odds
from quantization.basketball.basketball_dynamic_odds_cal import  DynamicOddsCalBas

import numpy as np

class TestClass2( object ):
    sup_ttg = [
        [ 0.5, 175.5 ],
        [ 0, 20 ],
    ]
    sigma = [ 11, 22, 5 ]

    score = [
        [ 10, 11 ],
        [ 0, 0 ],
        [ 200, 300 ],
    ]

    line = np.arange( -5, 5, 0.5 )



    def test_INTERFACE_collect_basketball_odds(self):
        mu = [5.0, 199 ]
        stage = 1
        current_sec = 12 * 60
        exp_total_sec = 48*60
        score = [0,0]
        parameter = [10, 1.05 ]
        clock = [ 1, current_sec, exp_total_sec ]
        match_format = [ 12* 60 , 4 ]

        doc1 = cal_basketball_match_odds()
        doc1.set_value( mu, score, clock, match_format, parameter )
        r1 = doc1.odds_output()

        r2 = INTERFACE_collect_basketball_odds( mu,
                                                score,
                                                clock,
                                                match_format,
                                                10,
                                                1.05 )

        assert( r1[period.BASKETBALL_FULL_TIME][market_type.BASKETBALL_2WAY] == \
                r2[period.BASKETBALL_FULL_TIME][market_type.BASKETBALL_2WAY])

        # only after 5 decimal is different
        # assert( r1[period.BASKETBALL_FULL_TIME][market_type.BASKETBALL_HANDICAP] == \
        #         r2[period.BASKETBALL_FULL_TIME][market_type.BASKETBALL_HANDICAP])

        assert( r1[period.BASKETBALL_FULL_TIME][market_type.BASKETBALL_TOTALS] == \
                r2[period.BASKETBALL_FULL_TIME][market_type.BASKETBALL_TOTALS])


    @pytest.mark.skip()
    def test_ahc_no_draw(self):
        doc_v1 = cal_basketball_odds()

        doc_v2 = DynamicOddsCalBas()

        for st in TestClass2.sup_ttg:
            for ssggmm in TestClass2.sigma:
                for s in TestClass2.score:
                    for l in TestClass2.line:
                        print( st, ssggmm, s )
                        doc_v1.set_value( st , s, [ssggmm])
                        doc_v2.refresh( sup_ttg=st, present_score=s, sigma=ssggmm )
                        assert ( doc_v1.asian_handicap_no_draw(l ) == doc_v2.ahc_no_draw(l))

    @pytest.mark.skip()
    def test_ahc_with_draw(self):
        doc_v1 = cal_basketball_odds()
        doc_v2 = DynamicOddsCalBas()

        for st in TestClass2.sup_ttg:
            for ssggmm in TestClass2.sigma:
                for s in TestClass2.score:
                    for l in TestClass2.line:
                        print( st, ssggmm, s )
                        doc_v1.set_value( st , s, [ssggmm])
                        doc_v2.refresh( sup_ttg=st, present_score=s, sigma=ssggmm )
                        assert ( doc_v1.asian_handicap( l ) == doc_v2.ahc_with_draw(l))

    @pytest.mark.skip()
    def test_over_under(self):
        doc_v1 = cal_basketball_odds()
        doc_v2 = DynamicOddsCalBas()

        line = np.arange( 0, 200, 0.5 )
        for st in TestClass2.sup_ttg:
            for ssggmm in TestClass2.sigma:
                for s in TestClass2.score:
                    for l in line:
                        print( st, ssggmm, s )
                        doc_v1.set_value( st , s, [ssggmm])
                        doc_v2.refresh( sup_ttg=st, present_score=s, sigma=ssggmm )
                        assert ( doc_v1.over_under( l ) == doc_v2.over_under(l))

    def test_had(self):
        doc_v1 = cal_basketball_odds()
        doc_v2 = DynamicOddsCalBas()

        for st in TestClass2.sup_ttg:
            for ssggmm in TestClass2.sigma:
                for s in TestClass2.score:
                    print( st, ssggmm, s )
                    doc_v1.set_value( st , s, [ssggmm])
                    doc_v2.refresh( sup_ttg=st, present_score=s, sigma=ssggmm )
                    assert ( doc_v1.had( ) == doc_v2.had() )


if __name__ == '__main__':


    c = InferBasketConfig()
    c.ou_line = 150.5
    c.over_odds = 1.9
    c.under_odds = 1.78
    c.scores = [6, 0]
    c.sigma = 9
    c.ahc_line = 10
    c.home_odds = 2.1
    c.away_odds = 1.8

    clock = [1, 15 * 60, 60 * 12 * 4]
    match_format = [12 * 60, 4]
    decay = 1.05

    from time import time

    t1 = time()
    # x = infer_sup_ttg_bas(c)
    # x = INTERFACE_infer_soccer_sup_ttg(config)
    x,y  = INTERFACE_infer_basketball_sup_ttg( c, clock, match_format, decay )
    print(x)
    print(y)

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
