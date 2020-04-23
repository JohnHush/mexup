import pytest
from quantization.soccer_poisson import cal_soccer_odds
from quantization.dynamic_odds_cal import DynamicOddsCal, DynamicOddsCalBas
from quantization.basketball_normal import cal_basketball_odds
import numpy as np
from quantization.constants import *

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
