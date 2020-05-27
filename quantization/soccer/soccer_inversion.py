#soccer 反查模型
import math

from scipy.optimize import minimize
import numpy as np
import math

from scipy.stats import poisson

from quantization.constants import selection_type
from quantization.soccer.soccer_dynamic_odds_cal import DynamicOddsCal


class InferSoccerConfig(object):
    def __init__(self):
        self._rho = -0.08
        self._scores = None
        self._ou_line = None
        self._ahc_line = None
        self._over_odds = None
        self._under_odds = None
        self._home_odds = None
        self._away_odds = None
        self._x0 = [None, None]

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, rho):
        self._rho = rho

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
    def x0(self):
        return self._x0



def infer_ttg_sup(config):
    def _grad(x, *args):
        sup, ttg = x[0], x[1]
        c = args[0]
        doc = args[1]
        om_normalized = args[2]
        ha_normalized = args[3]

        scores = c.scores
        rho = c.rho
        ou_line = c.ou_line
        ahc_line = c.ahc_line

        home_exp = (ttg + sup) / 2
        away_exp = (ttg - sup) / 2

        home_exp = max( home_exp, 0.00001 )
        away_exp = max( away_exp, 0.00001 )

        dim = 18 + int(max(home_exp, away_exp))

        ssum = scores[0] + scores[1]

        # compute gradient matrix of items over HOME
        home_prob_vec = np.array([poisson.pmf(i, home_exp) for i in range(dim)])
        away_prob_vec = np.array([poisson.pmf(i, away_exp) for i in range(dim)])
        home_grad_vec = np.array([i / home_exp - 1. for i in range(dim)]) * home_prob_vec
        away_grad_vec = np.array([i / away_exp - 1. for i in range(dim)]) * away_prob_vec

        P_raw = np.outer(home_prob_vec, away_prob_vec)

        P_raw[0, 0] *= 1 - home_exp * away_exp * rho
        P_raw[0, 1] *= 1 + home_exp * rho
        P_raw[1, 0] *= 1 + away_exp * rho
        P_raw[1, 1] *= 1 - rho

        P = np.transpose(P_raw)
        P = P[::-1]

        home_grad_matrix_raw = np.outer(home_grad_vec, away_prob_vec)
        home_grad_matrix_raw[0, 0] = (1 - rho * home_exp * away_exp) * home_grad_matrix_raw[0, 0] - \
                                     rho * away_exp * home_prob_vec[0] * away_prob_vec[0]

        home_grad_matrix_raw[0, 1] = (1 + rho * home_exp) * home_grad_matrix_raw[0, 1] + \
                                     rho * home_prob_vec[0] * away_prob_vec[1]
        home_grad_matrix_raw[1, 0] = (1 + rho * away_exp) * home_grad_matrix_raw[1, 0]
        home_grad_matrix_raw[1, 1] = (1 - rho) * home_grad_matrix_raw[1, 1]

        # compute gradient matrix of items over AWAY
        away_grad_matrix_raw = np.outer(home_prob_vec, away_grad_vec)
        away_grad_matrix_raw[0, 0] = (1 - rho * home_exp * away_exp) * away_grad_matrix_raw[0, 0] - \
                                     rho * home_exp * home_prob_vec[0] * away_prob_vec[0]

        away_grad_matrix_raw[0, 1] = (1 + rho * home_exp) * away_grad_matrix_raw[0, 1]
        away_grad_matrix_raw[1, 0] = (1 + rho * away_exp) * away_grad_matrix_raw[1, 0] + \
                                     rho * home_prob_vec[1] * away_prob_vec[0]
        away_grad_matrix_raw[1, 1] = (1 - rho) * away_grad_matrix_raw[1, 1]

        home_grad_matrix = np.transpose(home_grad_matrix_raw)
        away_grad_matrix = np.transpose(away_grad_matrix_raw)

        home_grad_matrix = home_grad_matrix[::-1]
        away_grad_matrix = away_grad_matrix[::-1]

        net_line = ou_line - ssum

        l, t = DynamicOddsCal._calculate_critical_line(net_line)
        ahc_l, ahc_t = DynamicOddsCal._calculate_critical_line(ahc_line)

        if ahc_t == DynamicOddsCal.AsianType.DIRECT:
            gggg2 = np.tril(home_grad_matrix_raw, math.floor(ahc_line)).sum() + \
                    np.tril(away_grad_matrix_raw, math.floor(ahc_line)).sum()
            tttt2 = np.tril(home_grad_matrix_raw, math.floor(ahc_line)).sum() - \
                    np.tril(away_grad_matrix_raw, math.floor(ahc_line)).sum()

            y_hat2 = np.tril(P_raw, math.floor(ahc_line)).sum()

        elif ahc_t == DynamicOddsCal.AsianType.NORMALIZE:
            tril_value = np.tril(P_raw, int(ahc_line) - 1).sum()
            triu_value = np.triu(P_raw, int(ahc_line) + 1).sum()

            tril_h = np.tril(home_grad_matrix_raw, int(ahc_line) - 1).sum()
            triu_h = np.triu(home_grad_matrix_raw, int(ahc_line) + 1).sum()
            tril_a = np.tril(away_grad_matrix_raw, int(ahc_line) - 1).sum()
            triu_a = np.triu(away_grad_matrix_raw, int(ahc_line) + 1).sum()

            gggg2 = ((triu_value + triu_value) * (tril_h + tril_a) + \
                     tril_value * (tril_h + tril_a + triu_h + triu_a)) / \
                    ((tril_value + triu_value) * (tril_value + triu_value))
            tttt2 = ((triu_value + triu_value) * (tril_h - tril_a) + \
                     tril_value * (tril_h - tril_a + triu_h - triu_a)) / \
                    ((tril_value + triu_value) * (tril_value + triu_value))

            y_hat2 = tril_value / (tril_value + triu_value)

        elif ahc_t == DynamicOddsCal.AsianType.DN_FIRST:
            diag_value = np.diag(P_raw, ahc_l).sum()
            tril_value = np.tril(P_raw, ahc_l - 1).sum()

            tril_h = np.tril(home_grad_matrix_raw, ahc_l - 1).sum()
            tril_a = np.tril(away_grad_matrix_raw, ahc_l - 1).sum()
            diag_h = np.diag(home_grad_matrix_raw, ahc_l).sum()
            diag_a = np.diag(away_grad_matrix_raw, ahc_l).sum()

            gggg2 = ((1 - 0.5 * diag_value) * (tril_h + tril_a) + \
                     0.5 * tril_value * (diag_h + diag_a)) / \
                    ((1 - 0.5 * diag_value) * (1 - 0.5 * diag_value))
            tttt2 = ((1 - 0.5 * diag_value) * (tril_h - tril_a) + \
                     0.5 * tril_value * (diag_h - diag_a)) / \
                    ((1 - 0.5 * diag_value) * (1 - 0.5 * diag_value))

            y_hat2 = tril_value / (1. - 0.5 * diag_value)

        elif ahc_t == DynamicOddsCal.AsianType.UP_FIRST:
            diag_value = np.diag(P_raw, ahc_l).sum()
            triu_value = np.triu(P_raw, ahc_l + 1).sum()

            triu_h = np.triu(home_grad_matrix_raw, ahc_l + 1).sum()
            triu_a = np.triu(away_grad_matrix_raw, ahc_l + 1).sum()
            diag_h = np.diag(home_grad_matrix_raw, ahc_l).sum()
            diag_a = np.diag(away_grad_matrix_raw, ahc_l).sum()

            gggg2 = -((1 - 0.5 * diag_value) * (triu_h + triu_a) + \
                      0.5 * triu_value * (diag_h + diag_a)) / \
                    ((1 - 0.5 * diag_value) * (1 - 0.5 * diag_value))
            tttt2 = -((1 - 0.5 * diag_value) * (triu_h - triu_a) + \
                      0.5 * triu_value * (diag_h - diag_a)) / \
                    ((1 - 0.5 * diag_value) * (1 - 0.5 * diag_value))

            y_hat2 = 1. - triu_value / (1. - 0.5 * diag_value)
        else:
            pass

        if t == DynamicOddsCal.AsianType.DIRECT:
            gggg = -1 * (np.tril(home_grad_matrix, math.floor(net_line) - dim + 1).sum() + \
                         np.tril(away_grad_matrix, math.floor(net_line) - dim + 1).sum())
            tttt = -1 * (np.tril(home_grad_matrix, math.floor(net_line) - dim + 1).sum() - \
                         np.tril(away_grad_matrix, math.floor(net_line) - dim + 1).sum())

            y_hat = 1. - np.tril(P, math.floor(net_line) - (dim - 1)).sum()

        elif t == DynamicOddsCal.AsianType.NORMALIZE:
            triu_value = np.triu(P, 1 + net_line - dim + 1).sum()
            tril_value = np.tril(P, -1 + net_line - dim + 1).sum()
            triu_h = np.triu(home_grad_matrix, 1 + net_line - dim + 1).sum()
            tril_h = np.tril(home_grad_matrix, -1 + net_line - dim + 1).sum()

            triu_a = np.triu(away_grad_matrix, 1 + net_line - dim + 1).sum()
            tril_a = np.tril(away_grad_matrix, -1 + net_line - dim + 1).sum()

            gggg = ((triu_value + tril_value) * (triu_h + triu_a) - \
                    triu_value * (triu_h + triu_a + tril_h + tril_a)) / \
                   ((triu_value + tril_value) * (triu_value + tril_value))

            tttt = ((triu_value + tril_value) * (triu_h - triu_a) - \
                    triu_value * (triu_h - triu_a + tril_h - tril_a)) / \
                   ((triu_value + tril_value) * (triu_value + tril_value))

            y_hat = triu_value / (triu_value + tril_value)

        elif t == DynamicOddsCal.AsianType.DN_FIRST:
            tril_h = np.tril(home_grad_matrix, l - (dim - 1) - 1).sum()
            tril_a = np.tril(away_grad_matrix, l - (dim - 1) - 1).sum()
            diag_h = np.diag(home_grad_matrix, l - (dim - 1)).sum()
            diag_a = np.diag(away_grad_matrix, l - (dim - 1)).sum()

            diag_value = np.diag(P, l - (dim - 1)).sum()
            tril_value = np.tril(P, l - (dim - 1) - 1).sum()

            gggg = -((1 - 0.5 * diag_value) * (tril_h + tril_a) + (0.5 * tril_value) * (diag_h + diag_a)) / \
                   ((1 - 0.5 * diag_value) * (1 - 0.5 * diag_value))

            tttt = -((1 - 0.5 * diag_value) * (tril_h - tril_a) + (0.5 * tril_value) * (diag_h - diag_a)) / \
                   ((1 - 0.5 * diag_value) * (1 - 0.5 * diag_value))

            y_hat = 1. - tril_value / (1. - 0.5 * diag_value)

        elif t == DynamicOddsCal.AsianType.UP_FIRST:
            triu_h = np.triu(home_grad_matrix, l - (dim - 1) + 1).sum()
            triu_a = np.triu(away_grad_matrix, l - (dim - 1) + 1).sum()
            diag_h = np.diag(home_grad_matrix, l - (dim - 1)).sum()
            diag_a = np.diag(away_grad_matrix, l - (dim - 1)).sum()

            diag_value = np.diag(P, l - (dim - 1)).sum()
            triu_value = np.triu(P, l - (dim - 1) + 1).sum()

            gggg = ((1 - 0.5 * diag_value) * (triu_h + triu_a) + (0.5 * triu_value) * (diag_h + diag_a)) / \
                   ((1 - 0.5 * diag_value) * (1 - 0.5 * diag_value))
            tttt = ((1 - 0.5 * diag_value) * (triu_h - triu_a) + (0.5 * triu_value) * (diag_h - diag_a)) / \
                   ((1 - 0.5 * diag_value) * (1 - 0.5 * diag_value))

            y_hat = triu_value / (1. - 0.5 * diag_value)
        else:
            pass

        # doc.refresh( sup_ttg=x, present_socre=scores, adj_params=[1,rho] )
        # y_hat = doc.over_under( ou_line )[selection_type.OVER]
        # y_hat2 = doc.asian_handicap( ahc_line )[selection_type.HOME]

        return np.array([0.5 * (y_hat - om_normalized) * tttt + 0.5 * (y_hat2 - ha_normalized) * tttt2, \
                         0.5 * (y_hat - om_normalized) * gggg + 0.5 * (y_hat2 - ha_normalized) * gggg2])

    def _obj(x, *args):
        '''
        '''
        c = args[0]
        doc = args[1]
        om_normalized = args[2]
        ha_normalized = args[3]

        doc.refresh(sup_ttg=x, present_socre=c.scores, adj_params=[1, c.rho])
        y_hat = doc.over_under(c.ou_line)[selection_type.OVER]
        y_hat2 = doc.asian_handicap(c.ahc_line)[selection_type.HOME]

        return 0.5 * (y_hat - om_normalized) * (y_hat - om_normalized) + \
               0.5 * (y_hat2 - ha_normalized) * (y_hat2 - ha_normalized)

    try:
        c = config
        ou_line = c.ou_line
        over_odds = c.over_odds
        under_odds = c.under_odds
        home_odds = c.home_odds
        away_odds = c.away_odds
        scores = c.scores
        x0 = c.x0

    except KeyError:
        return

    over_margin = 1. / over_odds
    under_margin = 1. / under_odds

    home_margin = 1. / home_odds
    away_margin = 1. / away_odds
    # target 1 to be matched
    om_normalized = over_margin / (over_margin + under_margin)
    ha_normalized = home_margin / (home_margin + away_margin)

    doc = DynamicOddsCal()

    model = minimize(_obj, x0=x0, \
                     args=(c, doc, om_normalized, ha_normalized), \
                     options={'disp': False, 'maxiter': 100}, jac=_grad)
    # model = minimize( _obj, x0=x0,
    #                   args=( c , doc, om_normalized , ha_normalized ),
    #                   options={ 'disp': True, 'maxiter': 300} )
    #

    return model.x.tolist()
