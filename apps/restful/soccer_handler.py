import json
from abc import ABC
from concurrent.futures.thread import ThreadPoolExecutor

import tornado
from tornado.concurrent import run_on_executor

from apps.base.base_handler import BaseHandler

#反查 totoal goal
from quantization.constants import config
from quantization.match_odds import cal_match_odds


class SoccerInferTotalGoalsHandler(BaseHandler, ABC):

    @run_on_executor
    def getData(self):

        # 解析参数
        query_score = [float(self.get_argument("home_score")), float(self.get_argument("away_score"))]
        over_under_market = [float(self.get_argument("ou_line")), float(self.get_argument("ou_over_odds")),
                             float(self.get_argument("ou_under_odds"))]
        max_total_goals = 16

        adjModel = int(self.get_argument("adj_mode"))

        parameter = None
        if adjModel == 1:
            parameter = [adjModel, float(self.get_argument("rho"))]
        else:
            parameter = [adjModel, [float(self.get_argument("draw_adj")), float(self.get_argument("draw_split"))]]

        # 实例化
        inferSoccer = self.getInferSoccer()
        inferSoccer.set_value_ou(query_score,over_under_market,max_total_goals,parameter,config.infer_eps)

        #计算
        return inferSoccer.infer_total_goals()

#反查 supremacy
class SoccerInferSupremacyHandler(BaseHandler, ABC):

    @run_on_executor
    def getData(self):
        # 解析参数
        query_score = [float(self.get_argument("home_score")), float(self.get_argument("away_score"))]
        asian_handicap_market = [float(self.get_argument("ahc_line")), float(self.get_argument("ahc_home_odds")),
                                 float(self.get_argument("ahc_away_odds"))]

        total_goals = float(self.get_argument("total_goals"))

        eps = config.infer_eps
        adjModel = int(self.get_argument("adj_mode"))

        parameter = None
        if adjModel == 1:
            parameter = [adjModel, float(self.get_argument("rho"))]
        else:
            parameter = [adjModel, [float(self.get_argument("draw_adj")), float(self.get_argument("draw_split"))]]

        # 实例化
        inferSoccer = self.getInferSoccer()
        inferSoccer.set_value_ahc(query_score, asian_handicap_market, total_goals, parameter, eps)

        #计算
        return inferSoccer.infer_supremacy()

class SoccerOddsHandler(BaseHandler, ABC) :

    @run_on_executor
    def getData(self):
        # 解析参数
        mu = [float(self.get_argument("supremacy")), float(self.get_argument("total_goals"))]

        score = [[int(self.get_argument("half_time_score_home")), int(self.get_argument("half_time_score_away"))],
                 [int(self.get_argument("full_time_score_home")), int(self.get_argument("full_time_score_away"))]]

        clock = [int(self.get_argument("stage")), int(self.get_argument("running_time")),
                 int(self.get_argument("ht_add")), int(self.get_argument("ft_ad"))]

        decay = float(self.get_argument("decay"))

        adjModel = int(self.get_argument("adj_mode"))

        parameter = None
        if adjModel == 1:
            parameter = [adjModel, float(self.get_argument("rho"))]
        else:
            parameter = [adjModel, [float(self.get_argument("draw_adj")), float(self.get_argument("draw_split"))]]

        #实例化
        soccerMatchOdds = cal_match_odds()
        soccerMatchOdds.set_value(mu, score, clock, decay, parameter)

        #计算
        return  soccerMatchOdds.odds_output()