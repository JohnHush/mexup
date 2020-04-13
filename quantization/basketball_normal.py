from scipy.stats import norm
from quantization.constants import selection_type
import numpy as np
import math
class cal_basketball_odds(object):
    #输入参数
    #mu: [期望得分差, 期望得分合]
    #score: [当前主队得分，当前客队得分]
    #parameter: [sigma] 标注差
    def set_value(self, mu, score, parameter):

        self.mu = mu
        self.sup = mu[0]
        self.ttg = mu[1]
        self.score = score
        self.sigma = parameter[0]*np.sqrt(2)
        self.mu_home = (self.sup+self.ttg)/2
        self.mu_away = (self.ttg-self.sup)/2
        self.home_score = self.score[0]
        self.away_score = self.score[1]
        self.score_diff = self.score[0]-self.score[1]
        self.score_sum = self.score[0]+self.score[1]
    #输出亚盘让分玩法概率，没有平局的情况
    def asian_handicap_no_draw(self, line):
        #净让球线
        net_line = line + self.score_diff

        draw_prob = norm(loc = self.sup, scale = self.sigma).cdf(-self.score_diff + 0.5) - norm(loc = self.sup, scale = self.sigma).cdf(-self.score_diff - 0.5)
        if math.ceil(net_line)-net_line == 0.5:
            tmp = round(norm(loc=self.sup, scale=self.sigma).cdf(-net_line), 5)
            if net_line < self.score_diff:
                home_prob_1 = 1- tmp
                away_prob_1 = tmp - draw_prob
            elif net_line > self.score_diff:
                home_prob_1 = 1 - tmp - draw_prob
                away_prob_1 = tmp
        elif math.ceil(net_line)-net_line == 0:
            tmp_1 = round(norm(loc=self.sup, scale=self.sigma).cdf(-net_line + 0.5), 5)
            tmp_2 = round(norm(loc=self.sup, scale=self.sigma).cdf(-net_line - 0.5), 5)
            if net_line < self.score_diff:
                home_prob_1 = 1- tmp_1
                away_prob_1 = tmp_2 - draw_prob
            elif net_line > self.score_diff:
                home_prob_1 = 1 - tmp_1 - draw_prob
                away_prob_1 = tmp_2
            elif net_line == self.score_diff:
                home_prob_1 = 1 - tmp_1
                away_prob_1 = tmp_2
        home_prob = home_prob_1 / (home_prob_1 + away_prob_1)
        away_prob = away_prob_1 / (home_prob_1 + away_prob_1)
        return {selection_type.HOME: round(home_prob, 5), selection_type.AWAY: round(away_prob, 5)}

    #输出亚盘让分玩法概率，有平局
    def asian_handicap(self, line):
        # 净让球线
        net_line = line + self.score_diff
        if math.ceil(net_line)-net_line == 0.5:
            away_prob = round(norm(loc=self.sup, scale=self.sigma).cdf(-net_line), 5)
            home_prob = 1 - away_prob
        elif math.ceil(net_line)-net_line == 0:
            home_prob_1 = 1- round(norm(loc=self.sup, scale=self.sigma).cdf(-net_line + 0.5), 5)
            away_prob_1 = round(norm(loc=self.sup, scale=self.sigma).cdf(-net_line - 0.5), 5)
            home_prob = home_prob_1 / (home_prob_1 + away_prob_1)
            away_prob = away_prob_1 / (home_prob_1 + away_prob_1)
        return {selection_type.HOME: round(home_prob, 5), selection_type.AWAY: round(away_prob, 5)}

    #输出亚盘大小玩法概率
    def over_under(self, line):
        # 净大小球线
        net_line = line - self.score_sum
        if math.ceil(net_line)-net_line == 0.5:
            under_prob = round(norm(loc=self.sup, scale=self.sigma).cdf(net_line), 5)
            over_prob = 1- under_prob
        elif math.ceil(net_line)-net_line == 0:
            over_prob_1 = 1- round(norm(loc=self.sup, scale=self.sigma).cdf(net_line + 0.5), 5)
            under_prob_1 = round(norm(loc=self.sup, scale=self.sigma).cdf(net_line - 0.5), 5)
            over_prob = over_prob_1 / (over_prob_1 + under_prob_1)
            under_prob = under_prob_1 / (over_prob_1 + under_prob_1)
        return {selection_type.HOME: round(over_prob, 5), selection_type.AWAY: round(under_prob, 5)}

    #输出三项盘概率
    def had(self):
        away_prob = norm(loc=self.sup, scale=self.sigma).cdf(-self.score_diff - 0.5)
        draw_prob = norm(loc = self.sup, scale = self.sigma).cdf(-self.score_diff + 0.5) - norm(loc = self.sup, scale = self.sigma).cdf(-self.score_diff - 0.5)
        home_prob = 1- away_prob - draw_prob
        return {selection_type.HOME: round(home_prob, 5), selection_type.DRAW: round(draw_prob, 5), selection_type.AWAY: round(away_prob, 5)}

    #输出两项盘概率
    def match_winner(self):
        return self.asian_handicap(-0.5)


# bk_odds = cal_basketball_odds()
# bk_odds.set_value([0.5, 175.5], [5, 0], [11])
#
# print(bk_odds.had())