import json
from abc import ABC

from apps.base.base_handler import BaseHandler

#反查 totoal goal
from quantization.constants import config
from quantization.match_odds import cal_match_odds


class SoccerInferTotalGoalsHandler(BaseHandler, ABC):
    def get(self):
        inferSoccer = self.getInferSoccer()
        self.setValue(inferSoccer)

        self.write({"code":0,"data":inferSoccer.infer_total_goals()})

    def setValue(self,inferSoccer):

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

        inferSoccer.set_value_ou(query_score,over_under_market,max_total_goals,parameter,config.infer_eps)

#反查 supremacy
class SoccerInferSupremacyHandler(BaseHandler, ABC):
    def get(self):
        inferSoccer = self.getInferSoccer()
        self.setValue(inferSoccer)
        self.write( {"code":0,"data":inferSoccer.infer_supremacy()} )

    def setValue(self,inferSoccer):

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

        inferSoccer.set_value_ahc(query_score, asian_handicap_market, total_goals, parameter, eps)


class SoccerOddsHandler(BaseHandler, ABC) :
    def get(self):
        soccerMatchOdds = cal_match_odds()
        self.setValue(soccerMatchOdds)
        self.write({"code": 0, "data":  soccerMatchOdds.odds_output()})

    def setValue(self,soccerMatchOdds):
        mu = [float(self.get_argument("supremacy")), float(self.get_argument("total_goals"))]

        # cal_match_odds([0.5,2.7],[[0,0],[0,0]],[0,0,1,3],0.88,[1,-0.08])

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

        soccerMatchOdds.set_value(mu, score, clock, decay, parameter)