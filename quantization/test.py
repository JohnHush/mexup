import pytest
from quantization.soccer_poisson import cal_soccer_odds
from quantization.dynamic_odds_cal import DynamicOddsCal
import numpy as np

class TestClass( object ):
    def test_had(self):
        examp=cal_soccer_odds()
        examp.set_value([0.5,3.7],[1,0],[1,-0.08])
        doc = DynamicOddsCal( [0.5,3.7],[1,0],[1,-0.08] )
        # print(examp.had())
        # print( doc.had() )
        assert examp.had() == doc.had()

    def test_double_chance(self):
        examp=cal_soccer_odds()
        examp.set_value([0.5,3.7],[1,0],[1,-0.08])
        doc = DynamicOddsCal( [0.5,3.7],[1,0],[1,-0.08] )
        # print(examp.double_chance())
        # print( doc.double_chance() )
        assert examp.double_chance() == doc.double_chance()

    def test_asian_handicap(self):
        examp=cal_soccer_odds()
        examp.set_value([0.5,3.7],[2,2],[1,-0.08])
        doc = DynamicOddsCal( [0.5,3.7],[2,2],[1,-0.08] )

        # l = [ -2.5, -2.25, -2, -1.75, -1.50, -1.25 , \
        #       -1, -0.5, -0.25, 0, 0.25, 0.5, 1., 1.25 ]
        l = [ -1.25 , 0, 1.25 ]

        for line in l:
            print( examp.asian_handicap(line))
            assert( doc.asian_handicap(line) == examp.asian_handicap(line) )

    def test_over_under(self):
        examp=cal_soccer_odds()
        examp.set_value([0.5,1.7],[0,0],[1,-0.08])
        doc = DynamicOddsCal( [0.5,1.7],[0,0],[1,-0.08] )

        l = [ -2.5, -2.25, -2, -1.75, -1.50, -1.25 , \
              -1, -0.5, -0.25, 0, 0.25, 0.5, 1., 1.25 , 1.5, 2.5, 3.5, 4.25, 3.75 ]

        for line in l:
            assert( doc.over_under(line) == examp.over_under(line) )

        # print( doc.over_under( -0.25 ) )
        # print( examp.over_under( -0.25 ))

    def test_over_under_home(self):
        examp=cal_soccer_odds()
        examp.set_value([0.5,1.7],[0,0],[1,-0.08])
        doc = DynamicOddsCal( [0.5,1.7],[0,0],[1,-0.08] )

        l = [ -2.5, -2.25, -2, -1.75, -1.50, -1.25 , \
              -1, -0.5, -0.25, 0, 0.25, 0.5, 1., 1.25 , 1.5, 2.5, 3.5, 4.25, 3.75 ]

        for line in l:
            # print(doc.over_under_home(line))
            # print(examp.home_over_under(line))
            assert( doc.over_under_home(line) == examp.home_over_under(line) )

    def test_over_under_away(self):
        examp=cal_soccer_odds()
        examp.set_value([0.5,1.7],[1,2],[1,-0.08])
        doc = DynamicOddsCal( [0.5,1.7],[1,2],[1,-0.08] )

        l = [ -2.5, -2.25, -2, -1.75, -1.50, -1.25 , \
              -1, -0.5, -0.25, 0, 0.25, 0.5, 1., 1.25 , 1.5, 2.5, 3.5, 4.25, 3.75 ]

        for line in l:
            # print(doc.over_under_away(line))
            # print(examp.away_over_under(line))
            assert( doc.over_under_away(line) == examp.away_over_under(line) )

    def test_exact_score(self):
        examp=cal_soccer_odds()
        examp.set_value([0.5,3.7],[1,0],[1,-0.08])
        doc = DynamicOddsCal( [0.5,3.7],[1,0],[1,-0.08] )

        # l = [ -2.5, -2.25, -2, -1.75, -1.50, -1.25 , \
        #       -1, -0.5, -0.25, 0, 0.25, 0.5, 1., 1.25 ]

        score = [ (1,0),
                  (2,1),
                  (0,1),
                  (0,0),
                  (3,2)]
        for h, a  in score:
            # print( doc.exact_score(h,a ))
            # print( examp.correct_score(h,a))
            assert( doc.exact_score(h, a ) == examp.correct_score(h, a ) )

    def test_exact_ttg(self):
        examp=cal_soccer_odds()
        examp.set_value([0.5,3.7],[1,0],[1,-0.08])
        doc = DynamicOddsCal( [0.5,3.7],[1,0],[1,-0.08] )

        l = list( range( -10, 10, 1 ) )
        for line in l:
            # print( doc.exact_ttg( line ))
            # print( examp.exact_totals(line))
            assert( doc.exact_ttg( line ) == examp.exact_totals( line ) )

    def test_exact_ttg_home(self):
        examp=cal_soccer_odds()
        examp.set_value([0.5,3.7],[1,0],[1,-0.08])
        doc = DynamicOddsCal( [0.5,3.7],[1,0],[1,-0.08] )

        l = list( range( -10, 10, 1 ) )
        for line in l:
            # print( doc.exact_ttg( line ))
            # print( examp.exact_totals(line))
            assert( doc.exact_ttg_home( line ) == examp.home_exact_totals( line ) )

    def test_exact_ttg_away(self):
        examp=cal_soccer_odds()
        examp.set_value([0.5,3.7],[1,0],[1,-0.08])
        doc = DynamicOddsCal( [0.5,3.7],[1,0],[1,-0.08] )

        l = list( range( -10, 10, 1 ) )
        for line in l:
            # print( doc.exact_ttg( line ))
            # print( examp.exact_totals(line))
            assert( doc.exact_ttg_away( line ) == examp.away_exact_totals( line ) )

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
