from quantization.dynamic_odds_cal import DynamicOddsCal, DynamicOddsCalBas
from quantization.constants import *
from scipy.optimize import minimize
import numpy as np
from scipy.stats import poisson, norm
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


class DocConfig(object):
    sup = None
    ttg = None
    rho = None
    scores = None

    ahc_line_list = None
    hilo_line_list = None
    correct_score_limit = None
    home_ou_line_list = None
    away_ou_line_list = None


def INTERFACE_infer_soccer_sup_ttg(c: InferSoccerConfig):
    return infer_ttg_sup(c)


def INTERFACE_infer_basketball_sup_ttg( c: InferBasketConfig,
                                        clock,
                                        match_format,
                                        decay ):
    """
    compute the current sup_ttg and the original sup_ttg
    Args:
        c:  config object to infer static sup and ttg
        clock: [ current_stage, current_seconds, expected_total_seconds ]
                current_stage is not used now,
                current_seconds , e.g. 12*60 stands for 12 min
                expected_total_seconds will be different from 4 * 12* 60 when
                    the game steps into overtime
        match_format: e.g. [ 12* 60 , 4 ]
        decay: decay factor when simulating the SUP_TTG decaying process
    """
    current_stage = clock[0]
    current_seconds = clock[1]
    expected_total_seconds = clock[2]
    expected_left_seconds = expected_total_seconds - current_seconds
    regular_total_seconds = match_format[0] * match_format[1]

    # use current time to calibrate Simga
    c.sigma = c.sigma * np.sqrt( expected_left_seconds / regular_total_seconds )
    multiple_factor = ( expected_left_seconds / expected_total_seconds ) ** decay
    [sup_now, ttg_now ] = infer_sup_ttg_bas( c )

    sup_original = sup_now / multiple_factor
    ttg_original = ttg_now / multiple_factor

    return [sup_now, sup_original], [ttg_now, ttg_original]


def INTERFACE_collect_soccer_odds( sup_ttg,
                                   scores,
                                   clock,
                                   decay,
                                   rho ):
    """

    Args:
        sup_ttg: [ sup, ttg ]
        scores: e.g [ [half_time_home_score, half_time_away_score],
                      [current_time_home_score, current_time_away_score] ]
        clock: [ stage, running_time, half_time_add, full_time_add ]
        decay: factor of expectation decay in game process
        rho: calibration factor for Dixon-COles model
    """
    first_half_score = scores[0]
    recent_score = scores[1]
    second_half_score = [ scores[1][0]-scores[0][0], scores[1][1]-scores[0][1] ]
    stage, running_time, ht_add, ft_add = clock[:4]

    decayed_sup_ttg_list = calculate_decayed_sup_ttg( sup_ttg,
                                                      stage,
                                                      running_time,
                                                      ht_add,
                                                      ft_add,
                                                      decay )

    doc_config1 = DocConfig()
    doc_config2 = DocConfig()

    if stage in [4,6]:
        sup_now, ttg_now = decayed_sup_ttg_list[-1][0], decayed_sup_ttg_list[-1][1]
        sup_1st, ttg_1st = decayed_sup_ttg_list[0][0], decayed_sup_ttg_list[0][1]
    if stage in [7,8]:
        sup_now, ttg_now = decayed_sup_ttg_list[0], decayed_sup_ttg_list[1]
        sup_1st, ttg_1st = None, None

    doc_config1.sup = sup_now
    doc_config1.ttg = ttg_now
    doc_config1.rho = rho
    doc_config1.scores = recent_score
    doc_config1.ahc_line_list = np.arange(-15, 12.25, 0.25)
    doc_config1.hilo_line_list = np.arange(0.5, 20.25, 0.25)
    doc_config1.correct_score_limit = 7
    doc_config1.home_ou_line_list = np.arange(0.5, 15.25, 0.25)
    doc_config1.away_ou_line_list = np.arange(0.5, 15.25, 0.25)

    if sup_1st:
        doc_config2.sup = sup_1st
        doc_config2.ttg = ttg_1st
        doc_config2.rho = rho
        doc_config2.scores = first_half_score
        doc_config2.ahc_line_list = np.arange(-10, 10.25, 0.25)
        doc_config2.hilo_line_list = np.arange(0.5, 15.25, 0.25)
        doc_config2.correct_score_limit = 7
        doc_config2.home_ou_line_list = np.arange(0.5, 10.25, 0.25)
        doc_config2.away_ou_line_list = np.arange(0.5, 10.25, 0.25)

    r = {period.SOCCER_FULL_TIME: collect_games_odds(doc_config1)}
    if sup_1st:
        r[period.SOCCER_FIRST_HALF] = collect_games_odds( doc_config2 )
    return r


def INTERFACE_collect_basketball_odds( sup_ttg,
                                       scores,
                                       clock,
                                       match_format,
                                       sigma,
                                       decay):
    """

    Args:
        sup_ttg: [ sup, ttg ]
        scores: [ home_score, away_score ]
        clock: [ stage, current_seconds, expected_total_seconds ]
                stage = 0: before match
                stage = 1: first quarter
                stage = 2: 2nd quarter
                stage = 3: 3rd quarter
                stage = 4: 4th quarter
                stage = 5: overtime period
                stage = 6: ending
        match_format: e.g. [ 12*60, 4 ]
        sigma: standard deviation of Gaussian model, e.g. 10
        decay: decay factor, like, 1.05
    """
    score_gap = scores[0] - scores[1]
    score_sum = scores[0] + scores[1]

    stage = clock[0]
    current_seconds = clock[1]
    expected_total_seconds = clock[2]
    expected_left_seconds = expected_total_seconds - current_seconds
    regular_total_seconds = match_format[0] * match_format[1]

    current_sigma = sigma * np.sqrt( expected_left_seconds / regular_total_seconds )
    current_sup = sup_ttg[0] * ( (expected_left_seconds / expected_total_seconds ) ** decay )
    current_ttg = sup_ttg[1] * ( (expected_left_seconds / expected_total_seconds ) ** decay )

    doc = DynamicOddsCalBas( sup_ttg=[ current_sup, current_ttg ],
                             present_score=scores,
                             sigma=current_sigma )

    r = { period.BASKETBALL_FULL_TIME : {} }

    r[period.BASKETBALL_FULL_TIME][market_type.BASKETBALL_2WAY] = doc.match_winner()

    ahc = {}
    ahc_line = math.ceil( sup_ttg[0] + score_gap )
    for i in np.arange( -ahc_line-10, -ahc_line+10.5, 0.5 ):
        ahc[str(i)] = doc.ahc_no_draw(i)

    r[period.BASKETBALL_FULL_TIME][market_type.BASKETBALL_HANDICAP] = ahc

    ou = {}
    ou_line = math.ceil( sup_ttg[1] + score_sum )
    for i in np.arange( ou_line-10, ou_line+10.5, 0.5 ):
        ou[str(i)] = doc.over_under(i)
    r[period.BASKETBALL_FULL_TIME][market_type.BASKETBALL_TOTALS] = ou

    return r


def calculate_decayed_sup_ttg(mu, stage, running_time, ht_add, ft_add, decay):
    """
    @param mu : [sup, ttg]
    @param stage: 4- pre match, 6 - first half, 7 - middle interval , 8 - second half
                    13 - end of regular time, 0 - all end
    """
    sup = mu[0]
    ttg = mu[1]
    last_half = (45 * 60 + ft_add) / (90 * 60 + ft_add)

    if stage in [4, 6]:
        t0 = (90 * 60 + ft_add + ht_add - running_time) / (90 * 60 + ft_add + ht_add)

        sup_now = sup * (t0 ** decay)
        ttg_now = ttg * (t0 ** decay)

        sup_2nd_half = sup * (last_half ** decay)
        ttg_2nd_half = ttg * (last_half ** decay)

        sup_1st_half = sup_now - sup_2nd_half
        ttg_1st_half = ttg_now - ttg_2nd_half

        return ([sup_1st_half, ttg_1st_half],
                [sup_2nd_half, ttg_2nd_half],
                [sup_now, ttg_now])

    if stage in [7, 8]:
        t0 = (90 * 60 + ft_add - running_time) / (90 * 60 + ft_add)

        sup_now = sup * (t0 ** decay)
        ttg_now = ttg * (t0 ** decay)

        return [sup_now, ttg_now]


def collect_games_odds(config: DocConfig):
    doc = DynamicOddsCal(sup_ttg=[config.sup, config.ttg],
                         present_socre=config.scores,
                         adj_params=[1, config.rho])

    odds_dict = {}

    odds_dict[market_type.SOCCER_3WAY] = doc.had()

    ahc_dict = {}
    for i in config.ahc_line_list:
        ahc_dict[str(i)] = doc.asian_handicap(i)
    odds_dict[market_type.SOCCER_ASIAN_HANDICAP] = ahc_dict

    ou_dict = {}
    for i in config.hilo_line_list:
        ou_dict[str(i)] = doc.over_under(i)
    odds_dict[market_type.SOCCER_ASIAN_TOTALS] = ou_dict

    cs_dict = {}
    for i in range(config.correct_score_limit):
        for j in range(config.correct_score_limit):
            cs_dict[str(i) + '_' + str(j)] = doc.exact_score(i, j)
    odds_dict[market_type.SOCCER_CORRECT_SCORE] = cs_dict

    hou_dict = {}
    for i in config.home_ou_line_list:
        hou_dict[str(i)] = doc.over_under_home(i)
    odds_dict[market_type.SOCCER_GOALS_HOME_TEAM] = hou_dict

    aou_dict = {}
    for i in config.away_ou_line_list:
        aou_dict[str(i)] = doc.over_under_away(i)
    odds_dict[market_type.SOCCER_GOALS_AWAY_TEAM] = aou_dict

    odds_dict[market_type.SOCCER_BOTH_TEAMS_TO_SCORE] = doc.both_scored()
    odds_dict[market_type.SOCCER_ODD_EVEN_GOALS] = doc.odd_even_ttg()

    return odds_dict


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
                     options={'disp': True, 'maxiter': 100}, jac=_grad)
    # model = minimize( _obj, x0=x0,
    #                   args=( c , doc, om_normalized , ha_normalized ),
    #                   options={ 'disp': True, 'maxiter': 300} )
    #

    return model.x


def calculate_base_sup_ttg(mu_now, stage, running_time, ht_add, ft_add, decay):
    """
    @param mu_now : [sup_now, ttg_now]
    @param stage: 4- pre match, 6 - first half, 7 - middle interval , 8 - second half
                    13 - end of regular time, 0 - all end
    """
    sup_now = mu_now[0]
    ttg_now = mu_now[1]

    if stage in [4, 6]:
        t0 = (90 * 60 + ft_add + ht_add - running_time) / (90 * 60 + ft_add + ht_add)

    if stage in [7, 8]:
        t0 = (90 * 60 + ft_add - running_time) / (90 * 60 + ft_add)

    sup_base = sup_now / (t0 ** decay)
    ttg_base = ttg_now / (t0 ** decay)

    return [sup_base, ttg_base]


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

    model = minimize(_obj, x0=c.x0,
                     options={'disp': True}, args=(config))

    return model.x


if __name__ == '__main__':
    config = InferSoccerConfig()
    config.ou_line = 3.5
    config.ahc_line = -0.25
    config.scores = [0, 0]
    config.over_odds = 1.9
    config.under_odds = 1.78
    config.home_odds = 2.1
    config.away_odds = 1.8

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
