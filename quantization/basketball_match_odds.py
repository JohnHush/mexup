#!/usr/bin/env python
# coding: utf-8
from quantization.constants import selection_type
import numpy as np
from quantization.basketball_normal import cal_basketball_odds
from quantization.constants import period


# score = [0,0]
# asian_handicap_market_now_draw=[-0.5,1.84,1.99]
# over_under_market=[2.5,1.84,1.99]
# parameter=[sigma, decay]
# clock = [current_stage, current_sec, exp_total_sec]
# match_format = [period_sec, total_period]
# eps=0.001


class infer_basketball_model_input(object):
    def __init__(self):
        self.odds_tool_reg_time = cal_basketball_odds()
        self.odds_tool_full_time_with_ot = cal_basketball_odds()
        self.odds_tool_1st_half = cal_basketball_odds()
        self.odds_tool_2nd_half = cal_basketball_odds()
        self.odds_tool_1st_quarter = cal_basketball_odds()
        self.odds_tool_2nd_quarter = cal_basketball_odds()
        self.odds_tool_3rd_quarter = cal_basketball_odds()
        self.odds_tool_4th_quarter = cal_basketball_odds()

    def set_value(self, mu, score, clock, match_format, parameter, eps):
        self.sup = mu[0]
        self.tts = mu[1]
        self.score = score
        self.current_stage = clock[0]
        self.current_sec = clock[1]
        self.exp_total_sec = clock[2]
        self.total_period = match_format[1]
        self.total_reg_sec = match_format[0] * match_format[1]
        self.sec_left = self.exp_total_sec - self.current_sec
        self.sigma_now = parameter[0]*np.sqrt(self.sec_left/self.total_reg_sec)
        self.decay = parameter[1]
        self.eps = eps

        if self.total_period = 4:

        if self.current_sec <


    def odds_output(self):


        odds = {}
        if self.stage in [4, 6]:
            self.odds_tool_full_time.set_value(self.mu_full_time_now, self.full_time_score, self.parameter)
            self.odds_tool_1st_half.set_value(self.mu_first_half_now, self.half_time_score, self.parameter)
            self.odds_tool_2nd_half.set_value(self.mu_second_half_now, self.second_half_socore, self.parameter)

            odds[period.SOCCER_FULL_TIME] = self.cal_full_time_odds()
            odds[period.SOCCER_FIRST_HALF] = self.cal_first_half_odds()
        elif self.stage in [7, 8]:
            self.odds_tool_full_time.set_value(self.mu_full_time_now, self.full_time_score, self.parameter)
            self.odds_tool_2nd_half.set_value(self.mu_second_half_now, self.second_half_socore, self.parameter)
            odds[period.SOCCER_FULL_TIME] = self.cal_full_time_odds()
        return odds