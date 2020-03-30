#!/usr/bin/env python
# coding: utf-8

from apps.quantization.soccer_poisson import cal_soccer_odds
#score = [0,0]
# asian_handicap_market=[-0.5,1.84,1.99]
# over_under_market=[2.5,1.84,1.99]
# rho=-0.08
# draw_adj=0.05
# draw_split=0.6
# draw_adj_parameter=[draw_adj,draw_split]
# adj_mode=1
#parameter=[adj_mode, adj_parameter], 当adj_mode=0时，adj_parameter格式为【draw_adj，draw_split】，当adj_mode=1时，adj_parameter格式为rho
# if adj_mode==1:
#     parameter=[1,rho]
# elif adj_mode==0:
#     parameter=[draw_adj,draw_split]
# eps=0.001


class infer_soccer_model_input(object):
    def __init__(self):
        self.odds_tool = cal_soccer_odds()

    def set_value_ahc(self, score, asian_handicap_market, total_goals, parameter, eps):
        self.score = score
        self.parameter = parameter
        self.asian_handicap_market = asian_handicap_market
        self.total_goals = total_goals
        self.eps = eps

    def set_value_ou(self, score, over_under_market, max_total_goals, parameter, eps):
        self.score = score
        self.parameter = parameter
        self.over_under_market = over_under_market
        self.max_total_goals = max_total_goals
        self.eps = eps

    def infer_total_goals(self):
        # 反查total goals
        exp_total_goals_left = self.eps
        exp_total_goals_right = self.max_total_goals-self.eps
        over_market_margin = round(1/self.over_under_market[1], 5)
        under_market_margin = round(1/self.over_under_market[2], 5)
        over_under_market_margin = over_market_margin+under_market_margin

        self.over_market_100_margin = round(over_market_margin/(over_under_market_margin), 5)
        self.over_under_line = self.over_under_market[0]
        
        exp_total_goals_middle = (exp_total_goals_left+exp_total_goals_right)/2

        count_ttg=0 # 统计迭代次数
        if self.over_under_line >= sum(self.score)+0.5:
            while abs(self.over_under_cost_function(exp_total_goals_middle)) > self.eps:
                exp_total_goals_middle = (exp_total_goals_left+exp_total_goals_right)/2
                if self.over_under_cost_function(exp_total_goals_left)*self.over_under_cost_function(exp_total_goals_middle)<=0:
                    exp_total_goals_right = exp_total_goals_middle
                else:
                    exp_total_goals_left = exp_total_goals_middle
                count_ttg = count_ttg+1
        return round(exp_total_goals_middle, 3)

    def infer_supremacy(self):
        #反查supremacy
        exp_supremacy_left = -self.total_goals+self.eps
        exp_supremacy_right = self.total_goals-self.eps
        home_market_margin = round(1/self.asian_handicap_market[1], 5)
        away_market_margin = round(1/self.asian_handicap_market[2], 5)
        asian_handicap_market_margin = home_market_margin+away_market_margin

        self.home_market_100_margin = round(home_market_margin/asian_handicap_market_margin, 5)
        self.asian_handicap_line = self.asian_handicap_market[0]
        exp_supremacy_middle = (exp_supremacy_left + exp_supremacy_right)/2
        count_sup = 0 # 统计迭代次数

        while abs(self.asian_handicap_cost_function(exp_supremacy_middle, self.total_goals)) > self.eps:
            exp_supremacy_middle = (exp_supremacy_left+exp_supremacy_right)/2

            if self.asian_handicap_cost_function(exp_supremacy_left, self.total_goals)*self.asian_handicap_cost_function(exp_supremacy_middle, self.total_goals) <= 0:
                exp_supremacy_right = exp_supremacy_middle
            else:
                exp_supremacy_left = exp_supremacy_middle
            count_sup = count_sup+1
        return round(exp_supremacy_middle, 3)

    def over_under_cost_function(self, x):
        self.odds_tool.set_value([0, x], self.score, self.parameter)
        return self.odds_tool.over_under(self.over_under_line)['over']-self.over_market_100_margin

    def asian_handicap_cost_function(self, x, exp_total_goals_middle):
        self.odds_tool.set_value([x, exp_total_goals_middle], self.score, self.parameter)
        return self.odds_tool.asian_handicap(self.asian_handicap_line)['home']-self.home_market_100_margin


# rho = -0.08
# draw_adj = 0.05
# draw_split = 0.6
# draw_adj_parameter = [draw_adj, draw_split]
# adj_mode = 1
# #
# # parameter=[adj_mode, adj_parameter], 当adj_mode=0时，adj_parameter格式为【draw_adj，draw_split】，当adj_mode=1时，adj_parameter格式为rho
# if adj_mode == 1:
#     parameter = [1, rho]
# elif adj_mode == 0:
#     parameter = [draw_adj, draw_split]
#
#
# eps = 0.005
# score = [0, 0]
# max_total_goals = 16
# over_under_market = [3.5, 1.9, 1.78]
# asian_handicap_market = [-0.75, 2.1, 1.8]
#
# para_input = infer_soccer_model_input()
# para_input.set_value_ou(score, over_under_market, max_total_goals, parameter, eps)
# total_goals = para_input.infer_total_goals()
# para_input.set_value_ahc(score, asian_handicap_market, total_goals, parameter, eps)
# supremacy = para_input.infer_supremacy()
#
#
# print(supremacy, total_goals)
