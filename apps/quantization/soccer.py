#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import poisson
import math


class SoccerOdds(object):
    # 输入参数
    # mu_home 主队期望进球数; mu_away 客队期望进球数; home_score 主队当前进球数; away_score 客队当前进球数;
    # adj_mode 赔率调整模式，0-平局调整模式，1-rho调整模式; adj_parameter, 平局调整模式[draw_adj, draw_split]，rho模式
    def __init__(self, mu, score, parameter):
        self.mu_home = (mu[0] + mu[1]) / 2
        self.mu_away = (mu[1] - mu[0]) / 2
        self.home_score = score[0]
        self.away_score = score[1]
        self.goal_diff = score[0] - score[1]
        self.goal_sum = score[0] + score[1]
        self.prob_matrix = np.zeros([16, 16])
        self.adj_mode = parameter[0]

        for i in range(0, 16):
            for j in range(0, 16):
                self.prob_matrix[i, j] = poisson.pmf(i, self.mu_home) * poisson.pmf(j, self.mu_away)

        if self.adj_mode == 0:
            self.draw_adj = parameter[1][0]
            self.draw_split = parameter[1][1]

            total_draw_adj = np.diag(self.prob_matrix).sum() * self.draw_adj
            total_home_adj = -total_draw_adj * (1 - self.draw_split)
            total_away_adj = -total_draw_adj * (self.draw_split)

            diag_new = np.diag(np.diag(self.prob_matrix * (1 + self.draw_adj)))
            tril_new = np.tril(self.prob_matrix, -1) * (1 + total_home_adj / np.tril(self.prob_matrix, -1).sum())
            triu_new = np.triu(self.prob_matrix, 1) * (1 + total_away_adj / np.triu(self.prob_matrix, 1).sum())
            self.prob_matrix = diag_new + tril_new + triu_new

        elif self.adj_mode == 1:
            self.rho = parameter[1]
            if self.rho >= max(-1 / self.mu_home, -1 / self.mu_away) and self.rho <= min(
                    1 / self.mu_home / self.mu_away, 1):
                self.prob_matrix[0, 0] = self.prob_matrix[0, 0] * (1 - self.mu_home * self.mu_away * self.rho)
                self.prob_matrix[0, 1] = self.prob_matrix[0, 1] * (1 + self.mu_home * self.rho)
                self.prob_matrix[1, 0] = self.prob_matrix[1, 0] * (1 + self.mu_away * self.rho)
                self.prob_matrix[1, 1] = self.prob_matrix[1, 1] * (1 - self.rho)

        # 计算主队进球数概率的向量
        self.home_prob_vector = np.sum(self.prob_matrix, axis=1)
        # 计算客队进球数概率的向量
        self.away_prob_vector = np.sum(self.prob_matrix, axis=0)

    # 输出 home，draw，away的概率
    def had(self):
        home_prob = np.tril(self.prob_matrix, -1 + self.goal_diff).sum()
        draw_prob = np.diag(self.prob_matrix, self.goal_diff).sum()
        away_prob = np.triu(self.prob_matrix, 1 + self.goal_diff).sum()
        return round(home_prob, 5), round(draw_prob, 5), round(away_prob, 5)

    # 输出double change 玩法
    def double_chance(self):
        home_prob = np.tril(self.prob_matrix, -1 + self.goal_diff).sum()
        draw_prob = np.diag(self.prob_matrix, self.goal_diff).sum()
        away_prob = np.triu(self.prob_matrix, 1 + self.goal_diff).sum()

        home_plus_draw_prob = home_prob + draw_prob
        home_plus_away_prob = home_prob + away_prob
        draw_plus_away = draw_prob + away_prob
        return round(home_plus_draw_prob, 5), round(home_plus_away_prob, 5), round(draw_plus_away, 5)

    # 输出 指定亚盘让球线的概率
    def asian_handicap(self, line):
        indicator = math.ceil(line) - line
        slice_line = math.ceil(line) - 1
        if indicator == 0.5:
            # 计算让球线为半球的情况
            home_prob = np.tril(self.prob_matrix, slice_line).sum()
            away_prob = 1 - home_prob
        elif indicator == 0.25:
            # 先计算主队全赢的概率
            # 投 line-0.25的line，主队全赢
            home_prob_1 = np.tril(self.prob_matrix, slice_line).sum()
            # 投line+0.25的line，主队全赢
            home_prob_2 = home_prob_1 / (home_prob_1 + np.triu(self.prob_matrix, 2 + slice_line).sum())
            home_prob = 2 / (1 / home_prob_1 + 1 / home_prob_2)
            away_prob = 1 - home_prob
        elif indicator == 0.75:
            # 先计算客队全赢的概率
            # 投line+0.25的line，客队全赢
            away_prob_1 = np.triu(self.prob_matrix, 1 + slice_line).sum()
            # 投line-0.25的line，客队全赢
            away_prob_2 = away_prob_1 / (away_prob_1 + np.tril(self.prob_matrix, slice_line - 1).sum())
            away_prob = 2 / (1 / away_prob_1 + 1 / away_prob_2)
            home_prob = 1 - away_prob
        elif indicator == 0:
            home_prob_1 = np.tril(self.prob_matrix, slice_line).sum()
            away_prob_1 = np.triu(self.prob_matrix, 2 + slice_line).sum()
            home_prob = home_prob_1 / (home_prob_1 + away_prob_1)
            away_prob = away_prob_1 / (home_prob_1 + away_prob_1)
        return round(home_prob, 5), round(away_prob, 5)

    # 矩阵90度向左翻转
    def flip90_left(self, arr):
        new_arr = np.transpose(arr)
        new_arr = new_arr[::-1]
        return new_arr

    # 输出指定line亚盘大小球的概率
    def over_under(self, line):
        prob_matrix_ou = self.flip90_left(self.prob_matrix)
        rows = prob_matrix_ou.shape[0]
        net_line = line - self.goal_sum
        indicator = math.ceil(net_line) - net_line
        slice_line = math.ceil(net_line) - rows
        if line >= self.goal_sum + 0.5 and indicator in [0.5, 0.25, 0.75, 0]:
            if indicator == 0.5:
                under_prob = np.tril(prob_matrix_ou, slice_line).sum()
                over_prob = 1 - under_prob
            elif indicator == 0.25:
                under_prob_1 = np.tril(prob_matrix_ou, slice_line).sum()
                under_prob_2 = under_prob_1 / (under_prob_1 + np.triu(prob_matrix_ou, slice_line + 2).sum())
                under_prob = 2 / (1 / under_prob_1 + 1 / under_prob_2)
                over_prob = 1 - under_prob
            elif indicator == 0:
                under_prob_1 = np.tril(prob_matrix_ou, slice_line).sum()
                over_prob_1 = np.triu(prob_matrix_ou, slice_line + 2).sum()
                under_prob = under_prob_1 / (under_prob_1 + over_prob_1)
                over_prob = over_prob_1 / (under_prob_1 + over_prob_1)
            elif indicator == 0.75:
                over_prob_1 = np.triu(prob_matrix_ou, slice_line + 1).sum()
                over_prob_2 = over_prob_1 / (over_prob_1 + np.tril(prob_matrix_ou, slice_line - 1).sum())
                over_prob = 2 / (1 / over_prob_1 + 1 / over_prob_2)
                under_prob = 1 - over_prob
        elif self.goal_sum >= line + 0.5:
            over_prob = 1
            under_prob = 0
        else:
            #             print("please in put valid line!")
            over_prob = 0
            under_prob = 0
        return round(over_prob, 5), round(under_prob, 5)

    # 输出正确比分概率
    def correct_score(self, home_target_score, away_target_score):
        home_score_gap = home_target_score - self.home_score
        away_score_gap = away_target_score - self.away_score
        return round(self.prob_matrix[home_score_gap, away_score_gap], 5)

    # double change over under 用到的大小球计算公式，只有.5 的line
    def cal_over_under(self, prob_matrix_input, line):
        prob_matrix_ou = self.flip90_left(prob_matrix_input)
        rows = prob_matrix_ou.shape[0]
        net_line = line - self.goal_sum
        indicator = math.ceil(net_line) - net_line
        slice_line = math.ceil(net_line) - rows

        if indicator == 0.5:
            under_prob = np.tril(prob_matrix_ou, slice_line).sum()
            over_prob = np.triu(prob_matrix_ou, slice_line + 1).sum()
        else:
            #             print("please input valid line!")
            over_prob = 0
            under_prob = 0
        return over_prob, under_prob

    # 输出双重机会大小
    def double_chance_over_under(self, line):
        home_prob_matrix = np.tril(self.prob_matrix, -1 + self.goal_diff)
        away_prob_matrix = np.triu(self.prob_matrix, 1 + self.goal_diff)
        draw_prob_matrix = self.prob_matrix - home_prob_matrix - away_prob_matrix

        home_plus_draw_prob_matrix = home_prob_matrix + draw_prob_matrix
        home_plus_away_prob_matrix = home_prob_matrix + away_prob_matrix
        draw_plus_away_prob_matrix = draw_prob_matrix + away_prob_matrix

        home_plus_draw_over, home_plus_draw_under = self.cal_over_under(home_plus_draw_prob_matrix, line)
        home_plus_away_over, home_plus_away_under = self.cal_over_under(home_plus_away_prob_matrix, line)
        draw_plus_away_over, draw_plus_away_under = self.cal_over_under(draw_plus_away_prob_matrix, line)
        return round(home_plus_draw_over, 5), round(home_plus_draw_under, 5), round(home_plus_away_over, 5), round(
            home_plus_away_under, 5), round(draw_plus_away_over, 5), round(draw_plus_away_under, 5)

    # 准确进球数
    def exact_totals(self, target_goals):
        prob_matrix_total_goals = self.flip90_left(self.prob_matrix)
        rows = prob_matrix_total_goals.shape[0]
        net_target_goals = target_goals - self.goal_sum
        slice_line = net_target_goals - rows + 1
        target_goals_prob = np.diag(prob_matrix_total_goals, slice_line).sum()
        return round(target_goals_prob, 5)

    # 主队进球数over under
    def home_over_under(self, line):

        net_line = line - self.home_score
        indicator = math.ceil(net_line) - net_line
        slice_line = max(math.ceil(net_line), -1)
        if line >= self.home_score + 0.5 and indicator in [0.5, 0.25, 0.75, 0]:
            if indicator == 0.5:
                under_prob = self.home_prob_vector[:slice_line].sum()
                over_prob = 1 - under_prob
            elif indicator == 0.25:
                under_prob_1 = self.home_prob_vector[:slice_line].sum()
                under_prob_2 = under_prob_1 / (1 - self.home_prob_vector[slice_line])
                under_prob = 2 / (1 / under_prob_1 + 1 / under_prob_2)
                over_prob = 1 - under_prob
            elif indicator == 0.75:
                over_prob_1 = 1 - self.home_prob_vector[:slice_line].sum()
                over_prob_2 = over_prob_1 / (1 - self.home_prob_vector[slice_line - 1])
                over_prob = 2 / (1 / over_prob_1 + 1 / over_prob_2)
                under_prob = 1 - over_prob
            elif indicator == 0:
                under_prob = self.home_prob_vector[:slice_line].sum() / (1 - self.home_prob_vector[slice_line])
                over_prob = self.home_prob_vector[slice_line + 1:].sum() / (1 - self.home_prob_vector[slice_line])
        elif self.home_score >= 0.5 + line:
            over_prob = 1
            under_prob = 0
        else:
            over_prob = 0
            under_prob = 0
        return round(over_prob, 5), round(under_prob, 5)

    # 客队进球数over under
    def away_over_under(self, line):

        net_line = line - self.away_score
        indicator = math.ceil(net_line) - net_line
        slice_line = math.ceil(net_line)
        if line >= self.away_score + 0.5 and indicator in [0.5, 0.25, 0.75, 0]:
            if indicator == 0.5:
                under_prob = self.away_prob_vector[:slice_line].sum()
                over_prob = 1 - under_prob
            elif indicator == 0.25:
                under_prob_1 = self.away_prob_vector[:slice_line].sum()
                under_prob_2 = under_prob_1 / (1 - self.away_prob_vector[slice_line])
                under_prob = 2 / (1 / under_prob_1 + 1 / under_prob_2)
                over_prob = 1 - under_prob
            elif indicator == 0.75:
                over_prob_1 = 1 - self.away_prob_vector[:slice_line].sum()
                over_prob_2 = over_prob_1 / (1 - self.away_prob_vector[slice_line - 1])
                over_prob = 2 / (1 / over_prob_1 + 1 / over_prob_2)
                under_prob = 1 - over_prob
            elif indicator == 0:
                under_prob = self.away_prob_vector[:slice_line].sum() / (1 - self.away_prob_vector[slice_line])
                over_prob = self.away_prob_vector[slice_line + 1:].sum() / (1 - self.away_prob_vector[slice_line])
        elif self.away_score >= line + 0.5:
            over_prob = 1
            under_prob = 0
        else:
            over_prob = 0
            under_prob = 0
        return round(over_prob, 5), round(under_prob, 5)

    # 主队准确进球数
    def home_exact_totals(self, home_target_goals):
        net_home_target_goals = home_target_goals - self.home_score
        home_target_goals_prob = self.home_prob_vector[net_home_target_goals]
        return round(home_target_goals_prob, 5)

    # 客队准确进球数
    def away_exact_totals(self, away_target_goals):
        net_away_target_goals = away_target_goals - self.away_score
        away_target_goals_prob = self.away_prob_vector[net_away_target_goals]
        return round(away_target_goals_prob, 5)

    # 主队净胜n球胜出
    def home_winning_by(self, target_winning_by):
        net_target_winning_by = target_winning_by - self.goal_diff
        home_winning_by_prob = np.diag(self.prob_matrix, -net_target_winning_by).sum()
        return round(home_winning_by_prob, 5)

    # 客队净胜n球胜出
    def away_winning_by(self, target_winning_by):
        net_target_winning_by = target_winning_by + self.goal_diff
        away_winning_by_prob = np.diag(self.prob_matrix, net_target_winning_by).sum()
        return round(away_winning_by_prob, 5)

    # 双方球队都进球
    def both_scored(self):
        yes_prob = self.home_over_under(0.5)[0] * self.away_over_under(0.5)[0]
        no_prob = 1 - yes_prob
        return round(yes_prob, 5), round(no_prob, 5)

    # odd even玩法
    def odd_even(self):
        rows = self.prob_matrix.shape[0]
        odd_prob = 0
        even_prob = 0
        for i in range(0, rows):
            for j in range(0, rows):
                if (i + j + self.goal_sum) % 2 == 0:
                    even_prob += self.prob_matrix[i, j]
                elif (i + j + self.goal_sum) % 2 == 1:
                    odd_prob += self.prob_matrix[i, j]
        return round(odd_prob, 5), round(even_prob, 5)


# In[1]:



# input home exp goals, away expt goals, current home score, current away score, adj mode, adj parameter (0-[draw_adj, draw_split])
#
# start = time.time()
#
# examp = SoccerOdds([0, 2.7], [0, 0], [1, -0.08])
#
# print("ms 1:", time.time() - start)
# start = time.time()
#
# print(examp.had())
#
# print("ms 2:", time.time() - start)
# start = time.time()
#
# print(examp.asian_handicap(-0.5))
# print("ms 3:", time.time() - start)
# start = time.time()
#
# print(examp.over_under(2.5))
#
# print("ms 4:", time.time() - start)
# start = time.time()
#
# print(examp.exact_totals(5))
#
# print("ms 5:", time.time() - start)
# start = time.time()
#
# print(examp.correct_score(3, 1))
#
# print("ms 6:", time.time() - start)
# start = time.time()
#
# print(examp.double_chance_over_under(2.5))
#
# print("ms 7:", time.time() - start)
# start = time.time()
#
# examp.home_over_under(0.75)
# examp.away_over_under(1.5)
# print(examp.home_exact_totals(3))
#
# print("ms 8:", time.time() - start)
# start = time.time()
#
# print(examp.away_exact_totals(0))
#
# print("ms 9:", time.time() - start)
# start = time.time()
#
# print(examp.home_winning_by(2))
#
# print("ms 10:", time.time() - start)
# start = time.time()
#
# print(examp.away_winning_by(1))
#
# print("ms 11:", time.time() - start)
# start = time.time()
#
# examp.both_scored()
# print(examp.odd_even())
#
# print("ms 12:", time.time() - start)
# start = time.time()
#
# # In[ ]:


# In[ ]:




