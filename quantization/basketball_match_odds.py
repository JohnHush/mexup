#!/usr/bin/env python
# coding: utf-8
from quantization.constants import selection_type
import numpy as np
from quantization.basketball_normal import cal_basketball_odds
from quantization.constants import period, market_type
import math


# score = [0,0]
# parameter=[sigma, decay]
# clock = [current_stage, current_sec, exp_total_sec]
# match_format = [period_sec, total_period]
# eps=0.001


class cal_basketball_match_odds(object):
    def __init__(self):
        self.odds_tool_reg_time = cal_basketball_odds()
        self.odds_tool_full_time = cal_basketball_odds()

    # self.stage 0-赛前，1-第一节 ，2-第二节 ，3-第三节，4-第四节, 5-加时，6-结束
    def set_value(self, mu, score, clock, match_format, parameter):
        self.sup = mu[0]
        self.tts = mu[1]
        self.score = score
        self.score_diff = score[0] - score[1]
        self.total_score = score[0] + score[1]
        self.stage = clock[0]
        self.current_sec = clock[1]
        self.exp_total_sec = clock[2]
        self.total_period = match_format[1]
        self.total_reg_sec = match_format[0] * match_format[1]
        self.sec_left = self.exp_total_sec - self.current_sec
        self.sigma_now = parameter[0]*np.sqrt(self.sec_left/self.total_reg_sec)
        self.decay = parameter[1]
        self.eps = eps

        self.sup_now = self.sup * ((self.sec_left / self.exp_total_sec) ** self.decay)
        self.tts_now = self.tts * ((self.sec_left / self.exp_total_sec) ** self.decay)

    def odds_output(self):

        odds = {}
        self.odds_tool_full_time.set_value([self.sup_now, self.tts_now], self.score, [self.sigma_now, self.decay])
        odds[period.BASKETBALL_FULL_TIME] = self.cal_full_time_odds()
        return odds

    def cal_full_time_odds(self):
        full_time_odds = {}
        full_time_odds[market_type.BASKETBALL_2WAY] = self.odds_tool_full_time.match_winner()

        ahc = {}
        ahc_core_line = math.ceil(self.sup + self.score_diff)
        ahc_line_list = np.arange(-ahc_core_line - 10, -ahc_core_line +10.5, 0.5)
        for j in ahc_line_list:
            ahc[str(j)] = self.odds_tool_full_time.asian_handicap_no_draw(j)
        full_time_odds[market_type.BASKETBALL_HANDICAP] = ahc

        over_under = {}
        ou_core_line = math.ceil(self.tts + self.total_score)
        ou_line_list = np.arange(ou_core_line - 10, ou_core_line +10.5, 0.5)
        for j in ou_line_list:
            over_under[str(j)] = self.odds_tool_full_time.over_under(j)
        full_time_odds[market_type.BASKETBALL_TOTALS] = over_under
        return full_time_odds



# mu = [5.0, 199]
# stage = 1
# current_sec = 12*60
# exp_total_sec = 48*60
# period_sec = 12 * 60
# total_period = 4
# score = [0,0]
# parameter=[10, 1.05]
# clock = [1, current_sec, exp_total_sec]
# match_format = [period_sec, total_period]
# eps=0.001
# bk_odds = cal_basketball_match_odds()
# bk_odds.set_value( mu, score, clock, match_format, parameter)
#
#
# print(bk_odds.odds_output())
