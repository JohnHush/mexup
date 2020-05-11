import math

from pandas import np

from quantization.basketball.basketball_dynamic_odds_cal import DynamicOddsCalBas
from quantization.basketball.basketball_inversion import InferBasketConfig, infer_sup_ttg_bas
from quantization.constants import market_type, period


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
