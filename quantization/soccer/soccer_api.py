import numpy as np

from quantization.constants import market_type, period
from quantization.soccer.soccer_dynamic_odds_cal import DynamicOddsCal, DocConfig
from quantization.soccer.soccer_inversion import InferSoccerConfig, infer_ttg_sup

def time_checking( stage, running_time, ht_add, ft_add ):
    if stage ==4 and running_time > 0:
        raise Exception( "stage =4, running time > 0 " )
    if stage ==6 and running_time >= 45 + ht_add + 5:
        raise Exception( "stage =6, running time > 45 + ht_add + 5" )
    if stage ==7 and running_time != 45:
        raise Exception( "stage =7, running time != 45 " )
    if stage ==8 and running_time >= 90 + ft_add + 5:
        raise Exception( "stage =8, running time >= 90 + ft_add + 5 " )
    if stage==13 and running_time != 90:
        raise Exception( "stage =13, running time != 90 " )

def sup_ttg_checking( sup, ttg ):
    if abs( sup ) >= ttg:
        raise Exception( " abs(sup) >= ttg ")

    if abs( sup ) >= 10 or ttg >= 16:
        raise Exception( "sup or ttg has extreme value")


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


def INTERFACE_infer_soccer_sup_ttg(c: InferSoccerConfig,
                                   stage,
                                   running_time,
                                   ht_add,
                                   ft_add,
                                   decay):
    """

    Args:
        c:
        stage:
        running_time:
        ht_add:
        ft_add:
        decay:

    Returns: [sup_now, sup_original], [ttg_now, ttg_original]

    """
    time_checking( stage, running_time, ht_add, ft_add )

    sup_now, ttg_now = infer_ttg_sup(c)
    sup_original, ttg_original = calculate_base_sup_ttg( [sup_now, ttg_now],
                                                         stage,
                                                         running_time,
                                                         ht_add,
                                                         ft_add,
                                                         decay )

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

    # checking
    time_checking( stage, running_time, ht_add, ft_add )
    sup_ttg_checking( sup_ttg[0], sup_ttg[1] )

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

