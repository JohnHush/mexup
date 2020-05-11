from scipy.stats import norm
import numpy as np
from quantization.constants import selection_type
import math


class DynamicOddsCalBas( object ):
    """
    calculate odds for basketball
    using Gaussian distribution, much easier to implement
    compared to Poisson model used in Soccer games
    """
    def __init__(self, **kwargs ):
        self.sup_ttg = kwargs.get( 'sup_ttg', [0,0] )
        self.present_score = kwargs.get( 'present_score', [0,0] )
        self.sigma = kwargs.get( 'sigma', 5 ) * np.sqrt(2)

        # self.home_exp = (self.sup_ttg[1] + self.sup_ttg[0]) / 2.
        # self.away_exp = (self.sup_ttg[1] - self.sup_ttg[0]) / 2.
        self._sgap = self.present_score[0] - self.present_score[1]
        self._ssum = self.present_score[0] + self.present_score[1]

    def refresh(self, **kwargs ):
        self.sup_ttg = kwargs.get( 'sup_ttg', self.sup_ttg )
        self.present_score = kwargs.get( 'present_score', self.present_score )
        self.sigma = kwargs.get( 'sigma', self.sigma/np.sqrt(2) ) * np.sqrt(2)

        # self.home_exp = (self.sup_ttg[1] + self.sup_ttg[0]) / 2.
        # self.away_exp = (self.sup_ttg[1] - self.sup_ttg[0]) / 2.
        self._sgap = self.present_score[0] - self.present_score[1]
        self._ssum = self.present_score[0] + self.present_score[1]

    def ahc_no_draw(self, line ):
        draw_prob = norm( loc=self.sup_ttg[0], scale=self.sigma ).cdf( -self._sgap + 0.5 ) - \
                    norm( loc=self.sup_ttg[0], scale=self.sigma ).cdf( -self._sgap - 0.5)

        if math.ceil(line) - line == 0.5:
            if line < 0:
                home_margin = 1. - norm(loc=self.sup_ttg[0], scale=self.sigma).cdf(-line-self._sgap)
                away_margin = 1. - home_margin - draw_prob
            else:
                away_margin = norm(loc=self.sup_ttg[0], scale=self.sigma).cdf(-line-self._sgap)
                home_margin = 1. - away_margin - draw_prob

        elif math.ceil(line) - line == 0:
            righ = norm(loc=self.sup_ttg[0], scale=self.sigma).cdf(-line-self._sgap + 0.5)
            left = norm(loc=self.sup_ttg[0], scale=self.sigma).cdf(-line-self._sgap - 0.5)

            if line < 0:
                home_margin = 1. - righ
                away_margin = left - draw_prob
            elif line > 0:
                home_margin = 1. - righ - draw_prob
                away_margin = left
            else:
                home_margin = 1. - righ
                away_margin = left
        else:
            raise Exception

        home_margin, away_margin = home_margin / ( home_margin + away_margin ),\
                                   away_margin / ( home_margin + away_margin )
        return {selection_type.HOME: round(home_margin, 5),
                selection_type.AWAY: round(away_margin, 5)}

    def ahc_with_draw(self, line ):
        if math.ceil(line) - line == 0.5:
            away_margin = norm( loc=self.sup_ttg[0], scale=self.sigma).cdf(-line-self._sgap)
            home_margin = 1. - away_margin
        elif math.ceil(line) - line == 0:
            home_margin = 1.- norm( loc=self.sup_ttg[0], scale=self.sigma).cdf(-line-self._sgap + 0.5)
            away_margin = norm( loc=self.sup_ttg[0], scale=self.sigma).cdf(-line-self._sgap - 0.5 )
            home_margin, away_margin = home_margin/ ( home_margin+away_margin ), \
                                       away_margin / (home_margin + away_margin)
        else:
            raise Exception

        return {selection_type.HOME: round(home_margin, 5),
                selection_type.AWAY: round(away_margin, 5)}

    def over_under(self, line):
        if math.ceil(line)-line == 0.5:
            under_margin = norm(loc=self.sup_ttg[1], scale=self.sigma).cdf(line-self._ssum)
            over_margin = 1. - under_margin
        elif math.ceil(line)-line == 0:
            under_margin = norm(loc=self.sup_ttg[1], scale=self.sigma).cdf(line-self._ssum-0.5)
            over_margin = 1. - norm(loc=self.sup_ttg[1], scale=self.sigma).cdf(line-self._ssum+0.5)

            over_margin, under_margin = over_margin /( over_margin + under_margin ), \
                                        under_margin / ( over_margin + under_margin )
        else:
            raise Exception

        return {selection_type.OVER: round(over_margin, 5),
                selection_type.UNDER: round(under_margin, 5)}

    def had(self):
        away_margin = norm(loc=self.sup_ttg[0], scale=self.sigma).cdf(-self._sgap - 0.5)
        draw_margin = norm(loc=self.sup_ttg[0], scale=self.sigma).cdf(-self._sgap + 0.5) - \
                      norm(loc=self.sup_ttg[0], scale=self.sigma).cdf(-self._sgap - 0.5)
        home_margin = 1. - away_margin - draw_margin

        return {selection_type.HOME: round(home_margin, 5),
                selection_type.DRAW: round(draw_margin, 5),
                selection_type.AWAY: round(away_margin, 5)}

    def match_winner(self):
        return self.ahc_no_draw(-0.5)

