from abc import ABC

from tornado.concurrent import run_on_executor

from apps.base.base_handler import BaseHandler

#反查 totoal goal
from quantization.old.basketball.basketball_match_odds import cal_basketball_match_odds


class BasketballInferTotalGoalsHandler(BaseHandler, ABC):


    # @run_on_executor
    def getData(self):

        # 解析参数
        query_score = [float(self.get_argument("home_score")), float(self.get_argument("away_score"))]
        clock = [int(self.get_argument("current_stage")), int(self.get_argument("current_sec")), int(self.get_argument("exp_total_sec"))]
        match_format = [int(self.get_argument("period_sec")), int(self.get_argument("total_period"))]
        over_under_market = [float(self.get_argument("ou_line")), float(self.get_argument("ou_over_odds")),
                             float(self.get_argument("ou_under_odds"))]

        max_total_goals = 400
        parameter = [float(self.get_argument("sigma")), float(self.get_argument("decay"))]

        # 实例化
        inferBasketball = self.getInferBasketball()
        inferBasketball.set_value_ou(query_score,clock,match_format,over_under_market,max_total_goals,parameter)

        #计算
        return inferBasketball.infer_total_score()

#反查 supremacy
class BasketballInferSupremacyHandler(BaseHandler, ABC):


    # @run_on_executor
    def getData(self):


        # 解析参数
        query_score = [float(self.get_argument("home_score")), float(self.get_argument("away_score"))]
        clock = [int(self.get_argument("current_stage")), int(self.get_argument("current_sec")),
                 int(self.get_argument("exp_total_sec"))]
        match_format = [int(self.get_argument("period_sec")), int(self.get_argument("total_period"))]

        asian_handicap_market_no_draw = [float(self.get_argument("ahc_line")), float(self.get_argument("ahc_home_odds")),
                                 float(self.get_argument("ahc_away_odds"))]

        totalScore = float(self.get_argument("total_goals"))

        parameter = [float(self.get_argument("sigma")), float(self.get_argument("decay"))]

        # 实例化
        inferBasketball = self.getInferBasketball()
        inferBasketball.set_value_ahc(query_score, clock, match_format, asian_handicap_market_no_draw, totalScore, parameter)

        # 计算
        return inferBasketball.infer_supremacy()

class BasketballOddsHandler(BaseHandler, ABC) :

    # @run_on_executor
    def getData(self):


        # 解析参数
        mu = [float(self.get_argument("supremacy")), float(self.get_argument("total_goals"))]
        score = [int(self.get_argument("home_score")), int(self.get_argument("away_score"))]

        clock = [int(self.get_argument("stage")), int(self.get_argument("current_sec")),
                 int(self.get_argument("exp_total_sec"))]

        match_format = [int(self.get_argument("period_sec")), int(self.get_argument("total_period"))]

        parameter = [float(self.get_argument("sigma")), float(self.get_argument("decay"))]

        # 实例化
        basketballMatchOdds = cal_basketball_match_odds()
        basketballMatchOdds.set_value(mu, score, clock, match_format, parameter)

        # 计算
        return basketballMatchOdds.odds_output()