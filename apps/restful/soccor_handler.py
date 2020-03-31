import json
from abc import ABC
#
# import tornado
# from apps.quantization.match_odds import cal_match_odds
from apps.base.base_handler import BaseHandler

#反查 totoal goal
class InferSoccerTotalGoalsHandler(BaseHandler, ABC):
    def get(self):
        inferSoccer = self.getInferSoccer();
        self.setValue(inferSoccer)

        self.write(json.dumps(inferSoccer.infer_total_goals()))

    def setValue(self,inferSoccer):

        # eps = 0.005
        # score = [0, 0]
        # max_total_goals = 16
        # over_under_market = [3.5, 1.9, 1.78]


        # para_input = infer_soccer_model_input()
        # para_input.set_value_ou(score, over_under_market, max_total_goals, parameter, eps)

        query_score = [float(self.get_argument("home_score")), float(self.get_argument("away_score"))]
        over_under_market = [float(self.get_argument("ou_line")), float(self.get_argument("ou_home_odds")),
                             float(self.get_argument("ou_away_odds"))]
        max_total_goals = 16
        eps = float(self.get_argument("eps"))
        adjModel = int(self.get_argument("adj_mode"))

        parameter = None
        if adjModel == 1:
            parameter = [adjModel, float(self.get_argument("rho"))];
        else:
            parameter = [adjModel, [float(self.get_argument("draw_adj")), float(self.get_argument("draw_split"))]]

        inferSoccer.set_value_ou(query_score,over_under_market,max_total_goals,parameter,eps)

#反查 supremacy
class InferSoccerSupremacyHandler(BaseHandler, ABC):
    def get(self):
        inferSoccer = self.getInferSoccer();
        self.setValue(inferSoccer)

        self.write( json.dumps( (inferSoccer.infer_supremacy()) ) )

    def setValue(self,inferSoccer):

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

        query_score = [float(self.get_argument("home_score")), float(self.get_argument("away_score"))]
        asian_handicap_market = [float(self.get_argument("ahc_line")), float(self.get_argument("ahc_home_odds")),
                                 float(self.get_argument("ahc_away_odds"))]

        total_goals = float(self.get_argument("total_goals"))

        eps = float(self.get_argument("eps"))
        adjModel = int(self.get_argument("adj_mode"))

        parameter = None
        if adjModel == 1:
            parameter = [adjModel, float(self.get_argument("rho"))];
        else:
            parameter = [adjModel, [float(self.get_argument("draw_adj")), float(self.get_argument("draw_split"))]]

        inferSoccer.set_value_ahc(query_score, asian_handicap_market, total_goals, parameter, eps);


# class InferSoccerMuHandler(BaseHandler, ABC):
#     def get(self):
#         inferSoccer = self.getInferSoccer();
#         self.write( json.dumps((inferSoccer.infer_supremacy_total_goals())) )


class ScocorFullTimeHandicapHandler(BaseHandler, ABC) :

    def get(self):

        matchOdds = self.getMatchOdds();

        self.write( (matchOdds.full_time()) )

#
# class HadHandler(BaseHandler, ABC) :
#
#     def get(self):
#         soccerOdds = self.getSoccerOdds()
#         result = soccerOdds.had()
#
#         # soccerOdds.asian_handicap(-0.5)
#         # soccerOdds.over_under(2.5)
#         # soccerOdds.exact_totals(5)
#         # soccerOdds.correct_score(3, 1)
#         # soccerOdds.double_chance_over_under(2.5)
#         # soccerOdds.home_over_under(0.75)
#         # soccerOdds.away_over_under(1.5)
#         # soccerOdds.home_exact_totals(3)
#         # soccerOdds.away_exact_totals(0)
#         # soccerOdds.home_winning_by(2)
#         # soccerOdds.away_winning_by(1)
#         # soccerOdds.both_scored()
#         # soccerOdds.odd_even()
#
#         self.write(','.join(str(i) for i in result))
#
#
# class AsianHandicapHandler(BaseHandler, ABC) :
#
#     def get(self):
#         soccerOdds = self.getSoccerOdds();
#         line = self.get_argument("line")
#         self.write((soccerOdds.asian_handicap(line)))

