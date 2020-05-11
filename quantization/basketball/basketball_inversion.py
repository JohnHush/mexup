#basketball 反查模型
from scipy.optimize import minimize
import numpy as np
from scipy.stats import  norm
import math


class InferBasketConfig(object):
    def __init__(self):
        self._scores = None
        self._ou_line = None
        self._ahc_line = None
        self._over_odds = None
        self._under_odds = None
        self._home_odds = None
        self._away_odds = None
        self._sigma = None
        self._x0 = [None, None]

    @property
    def scores(self):
        return self._scores

    @scores.setter
    def scores(self, scores):
        self._scores = scores

    @property
    def ou_line(self):
        return self._ou_line

    @ou_line.setter
    def ou_line(self, ou_line):
        self._ou_line = ou_line
        self._x0[1] = ou_line

    @property
    def ahc_line(self):
        return self._ahc_line

    @ahc_line.setter
    def ahc_line(self, ahc_line):
        self._ahc_line = ahc_line
        self._x0[0] = -ahc_line

    @property
    def over_odds(self):
        return self._over_odds

    @over_odds.setter
    def over_odds(self, over_odds):
        self._over_odds = over_odds

    @property
    def under_odds(self):
        return self._under_odds

    @under_odds.setter
    def under_odds(self, under_odds):
        self._under_odds = under_odds

    @property
    def home_odds(self):
        return self._home_odds

    @home_odds.setter
    def home_odds(self, home_odds):
        self._home_odds = home_odds

    @property
    def away_odds(self):
        return self._away_odds

    @away_odds.setter
    def away_odds(self, away_odds):
        self._away_odds = away_odds

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma

    @property
    def x0(self):
        return self._x0




def infer_sup_ttg_bas(config: InferBasketConfig):
    def _obj(x, *args):

        config = args[0]

        line = config.ou_line
        over_odds = config.over_odds
        under_odds = config.under_odds
        score = config.scores
        sigma = config.sigma * np.sqrt(2)
        home_odds = config.home_odds
        away_odds = config.away_odds
        ahc_line = config.ahc_line

        ssum = score[0] + score[1]
        sgap = score[0] - score[1]

        om = 1. / over_odds
        um = 1. / under_odds
        om = om / (om + um)

        hm = 1. / home_odds
        am = 1. / away_odds
        hm = hm / (hm + am)

        if math.ceil(line) - line == 0.5:
            under_margin = norm(loc=x[1], scale=sigma).cdf(line - ssum)
            over_margin = 1. - under_margin
        elif math.ceil(line) - line == 0:
            under_margin = norm(loc=x[1], scale=sigma).cdf(line - ssum - 0.5)
            over_margin = 1. - norm(loc=x[1], scale=sigma).cdf(line - ssum + 0.5)

            over_margin, under_margin = over_margin / (over_margin + under_margin), \
                                        under_margin / (over_margin + under_margin)
        else:
            raise Exception

        draw_prob = norm(loc=x[0], scale=sigma).cdf(-sgap + 0.5) - \
                    norm(loc=x[0], scale=sigma).cdf(-sgap - 0.5)

        if math.ceil(ahc_line) - ahc_line == 0.5:
            if ahc_line < 0:
                home_margin = 1. - norm(loc=x[0], scale=sigma).cdf(-ahc_line - sgap)
                away_margin = 1. - home_margin - draw_prob
            else:
                away_margin = norm(loc=x[0], scale=sigma).cdf(-ahc_line - sgap)
                home_margin = 1. - away_margin - draw_prob

        elif math.ceil(ahc_line) - ahc_line == 0:
            righ = norm(loc=x[0], scale=sigma).cdf(-ahc_line - sgap + 0.5)
            left = norm(loc=x[0], scale=sigma).cdf(-ahc_line - sgap - 0.5)

            if ahc_line < 0:
                home_margin = 1. - righ
                away_margin = left - draw_prob
            elif ahc_line > 0:
                home_margin = 1. - righ - draw_prob
                away_margin = left
            else:
                home_margin = 1. - righ
                away_margin = left
        else:
            raise Exception

        home_margin = home_margin / (home_margin + away_margin)

        return 0.5 * (om - over_margin) * (om - over_margin) + \
               0.5 * (hm - home_margin) * (hm - home_margin)

    model = minimize(_obj, x0=config.x0,
                     options={'disp': True}, args=(config))

    return model.x
