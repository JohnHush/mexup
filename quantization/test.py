import pytest
from quantization.soccer_poisson import cal_soccer_odds
from quantization.dynamic_odds_cal import DynamicOddsCal

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
        examp.set_value([0.5,3.7],[1,0],[1,-0.08])
        doc = DynamicOddsCal( [0.5,3.7],[1,0],[1,-0.08] )

        l = [ -2.5, -2.25, -2, -1.75, -1.50, -1.25 , \
              -1, -0.5, -0.25, 0, 0.25, 0.5, 1., 1.25 ]

        for line in l:
            assert( doc.asian_handicap(line) == examp.asian_handicap(line) )

