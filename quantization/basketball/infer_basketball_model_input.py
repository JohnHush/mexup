#!/usr/bin/env python
# coding: utf-8
from quantization.constants import selection_type, config
import numpy as np
from quantization.basketball.basketball_normal import cal_basketball_odds


# score = [0,0]
# asian_handicap_market_now_draw=[-0.5,1.84,1.99]
# over_under_market=[2.5,1.84,1.99]
# parameter=[sigma, decay]
# clock = [current_stage, current_sec, exp_total_sec]
# match_format = [period_sec, total_period]
# eps=0.001


class infer_basketball_model_input(object):
    def __init__(self):
        self.odds_tool = cal_basketball_odds()
        self.eps = config.infer_eps

    def set_value_ahc(self, score, clock, match_format, asian_handicap_market_no_draw, total_score, parameter):
        self.score = score
        self.current_stage = clock[0]
        self.current_sec = clock[1]
        self.exp_total_sec = clock[2]
        self.total_reg_sec = match_format[0] * match_format[1]
        self.sec_left = self.exp_total_sec - self.current_sec
        self.sigma_now = parameter[0]*np.sqrt(self.sec_left/self.total_reg_sec)
        self.decay = parameter[1]
        self.asian_handicap_market = asian_handicap_market_no_draw
        self.total_score = total_score


    def set_value_ou(self, score, clock, match_format, over_under_market, max_total_score, parameter):
        self.score = score
        self.current_stage = clock[0]
        self.current_sec = clock[1]
        self.exp_total_sec = clock[2]
        self.total_reg_sec = match_format[0] * match_format[1]
        self.sec_left = self.exp_total_sec - self.current_sec
        self.max_total_score = max_total_score
        self.sigma_now = parameter[0]*np.sqrt(self.sec_left/self.total_reg_sec)
        self.decay = parameter[1]
        self.over_under_market = over_under_market

    def infer_total_score(self):
        # 反查total goals
        exp_total_score_left = self.eps
        exp_total_score_right = self.max_total_score - self.eps
        over_market_margin = round(1 / self.over_under_market[1], 5)
        under_market_margin = round(1 / self.over_under_market[2], 5)
        over_under_market_margin = over_market_margin + under_market_margin

        self.over_market_100_margin = round(over_market_margin / (over_under_market_margin), 5)
        self.over_under_line = self.over_under_market[0]

        exp_total_score_middle = (exp_total_score_left + exp_total_score_right) / 2

        count_tts = 0  # 统计迭代次数
        if self.over_under_line >= sum(self.score) + 0.5:
            while abs(self.over_under_cost_function(exp_total_score_middle)) > self.eps:
                exp_total_score_middle = (exp_total_score_left + exp_total_score_right) / 2
                if self.over_under_cost_function(exp_total_score_left) * self.over_under_cost_function(
                        exp_total_score_middle) <= 0:
                    exp_total_score_right = exp_total_score_middle
                else:
                    exp_total_score_left = exp_total_score_middle
                count_tts = count_tts + 1
        exp_total_score_now = round(exp_total_score_middle, 3)
        exp_total_score_base = round(exp_total_score_now / ((self.sec_left/self.exp_total_sec)**self.decay), 3)
        return exp_total_score_now, exp_total_score_base

    def infer_supremacy(self):
        # 反查supremacy
        exp_supremacy_left = -self.total_score + self.eps
        exp_supremacy_right = self.total_score - self.eps
        home_market_margin = round(1 / self.asian_handicap_market[1], 5)
        away_market_margin = round(1 / self.asian_handicap_market[2], 5)
        asian_handicap_market_margin = home_market_margin + away_market_margin

        self.home_market_100_margin = round(home_market_margin / asian_handicap_market_margin, 5)
        self.asian_handicap_line = self.asian_handicap_market[0]
        exp_supremacy_middle = (exp_supremacy_left + exp_supremacy_right) / 2
        count_sup = 0  # 统计迭代次数

        while abs(self.asian_handicap_cost_function(exp_supremacy_middle, self.total_score)) > self.eps:
            exp_supremacy_middle = (exp_supremacy_left + exp_supremacy_right) / 2

            if self.asian_handicap_cost_function(exp_supremacy_left,
                                                 self.total_score) * self.asian_handicap_cost_function(
                    exp_supremacy_middle, self.total_score) <= 0:
                exp_supremacy_right = exp_supremacy_middle
            else:
                exp_supremacy_left = exp_supremacy_middle
            count_sup = count_sup + 1
        exp_sup_now = round(exp_supremacy_middle, 3)
        exp_sup_base = round(exp_sup_now / ((self.sec_left/self.exp_total_sec)**self.decay), 3)
        return exp_sup_now, exp_sup_base

    def over_under_cost_function(self, x):
        self.odds_tool.set_value([0, x], self.score, [self.sigma_now])
        return self.odds_tool.over_under(self.over_under_line)[selection_type.OVER] - self.over_market_100_margin

    def asian_handicap_cost_function(self, x, exp_total_score):
        self.odds_tool.set_value([x, exp_total_score], self.score, [self.sigma_now])
        return self.odds_tool.asian_handicap_no_draw(self.asian_handicap_line)[
                   selection_type.HOME] - self.home_market_100_margin



# eps = 0.005
# score = [6, 0]
# max_total_score = 400
# clock = [1, 0, 60*12*4]
# match_format =[12*60, 4]
# over_under_market = [167.5, 1.9, 1.78]
# asian_handicap_market_no_draw = [-0.5, 2.1, 1.8]
# parameter = [9, 1.05]
#
# para_input = infer_basketball_model_input()
# para_input.set_value_ou(score, clock, match_format, over_under_market, max_total_score, parameter, eps)
# total_score = para_input.infer_total_score()
# para_input.set_value_ahc(score, clock, match_format, asian_handicap_market_no_draw, total_score[0], parameter, eps)
# supremacy = para_input.infer_supremacy()
#
# print(supremacy, total_score)
