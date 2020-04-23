from abc import ABC

from apps.base.base_handler import BaseHandler

#反查 totoal goal
from quantization.basketball.basketball_match_odds import cal_basketball_match_odds
from quantization.constants import config


class BasketballInferTotalGoalsHandler(BaseHandler, ABC):
    def get(self):
        inferBasketball = self.getInferBasketball();
        self.setValue(inferBasketball)

        self.write({"code":0,"data":inferBasketball.infer_total_score()})

    def setValue(self,inferBasketball):

        query_score = [float(self.get_argument("home_score")), float(self.get_argument("away_score"))]
        clock = [int(self.get_argument("current_stage")), int(self.get_argument("current_sec")), int(self.get_argument("exp_total_sec"))]
        match_format = [int(self.get_argument("period_sec")), int(self.get_argument("total_period"))]
        over_under_market = [float(self.get_argument("ou_line")), float(self.get_argument("ou_over_odds")),
                             float(self.get_argument("ou_under_odds"))]

        max_total_goals = 400

        parameter = [float(self.get_argument("sigma")), float(self.get_argument("decay"))]

        inferBasketball.set_value_ou(query_score,clock,match_format,over_under_market,max_total_goals,parameter)

#反查 supremacy
class BasketballInferSupremacyHandler(BaseHandler, ABC):
    def get(self):
        inferBasketball = self.getInferBasketball();
        self.setValue(inferBasketball)
        self.write( {"code":0,"data":inferBasketball.infer_supremacy()} )

    def setValue(self,inferBasketball):

        query_score = [float(self.get_argument("home_score")), float(self.get_argument("away_score"))]
        clock = [int(self.get_argument("current_stage")), int(self.get_argument("current_sec")),
                 int(self.get_argument("exp_total_sec"))]
        match_format = [int(self.get_argument("period_sec")), int(self.get_argument("total_period"))]

        asian_handicap_market_no_draw = [float(self.get_argument("ahc_line")), float(self.get_argument("ahc_home_odds")),
                                 float(self.get_argument("ahc_away_odds"))]

        totalScore = float(self.get_argument("total_goals"))

        parameter = [float(self.get_argument("sigma")), float(self.get_argument("decay"))]

        inferBasketball.set_value_ahc(query_score, clock, match_format, asian_handicap_market_no_draw, totalScore, parameter)

class BasketballOddsHandler(BaseHandler, ABC) :

    def get(self):
        basketballMatchOdds = cal_basketball_match_odds();
        self.setValue(basketballMatchOdds)
        self.write( {"code":0,"data":basketballMatchOdds.odds_output()} )

    def setValue(self,basketballMatchOdds):
        mu = [float(self.get_argument("supremacy")), float(self.get_argument("total_goals"))]
        score = [int(self.get_argument("home_score")), int(self.get_argument("away_score"))]

        clock = [int(self.get_argument("stage")), int(self.get_argument("current_sec")),
                 int(self.get_argument("exp_total_sec"))]

        match_format = [int(self.get_argument("period_sec")), int(self.get_argument("total_period"))]

        parameter = [float(self.get_argument("sigma")), float(self.get_argument("decay"))]

        basketballMatchOdds.set_value(mu, score, clock, match_format, parameter)