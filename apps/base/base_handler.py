from abc import ABC

import tornado.web
#
# from apps.quantization.soccer import SoccerOdds

from quantization.match_odds import cal_match_odds
from quantization.infer_soccer_model_input import infer_soccer_model_input

#
# import logging
# logger = logging.getLogger(__name__)
#


class NoResultError(Exception):
    pass


class BaseHandler(tornado.web.RequestHandler, ABC):


    def getInstanceKey(self,mu,score,adjModel,adjPara0,adjPara1):
        if adjModel == 1:
            return "{}-{}-{}-{}-{}-{}".format(mu[0],mu[1],score[0],score[1],adjModel,adjPara0)
        else:
            return "{}-{}-{}-{}-{}-{}-{}".format(mu[0],mu[1],score[0],score[1],adjModel,adjPara0,adjPara1)

    def getInferSoccer(self):

        cal = infer_soccer_model_input();
        return cal;


    def getMatchOdds(self):

        mu = [float(self.get_argument("supremacy")), float(self.get_argument("total_goals"))]

        #cal_match_odds([0.5,2.7],[[0,0],[0,0]],[0,0,1,3],0.88,[1,-0.08])

        score = [ [int(self.get_argument("half_time_score_home")), int(self.get_argument("half_time_score_away"))],
                 [int(self.get_argument("full_time_score_home")), int(self.get_argument("full_time_score_away"))] ]

        clock = [ int(self.get_argument("stage")), int(self.get_argument("running_time")),
                  int(self.get_argument("ht_add")), int(self.get_argument("ft_ad")) ]

        decay = float(self.get_argument("decay"))

        adjModel = int(self.get_argument("adj_mode"))

        parameter = None
        if adjModel == 1:
            parameter = [adjModel,float(self.get_argument("rho"))];
        else:
            parameter = [adjModel, [float(self.get_argument("draw_adj")), float(self.get_argument("draw_split"))]]



        # para_input = cal_soccer_model_input()
        # para_input.set_value(score, asian_handicap_market, over_under_market, max_total_goals, parameter, eps)
        #
        # print(para_input.infer_supremacy_total_goals())






        #
        # match=cal_match_odds([0.5,2.7],[[0,0],[0,0]],[0,0,1,3],[0.88,0.88],[1,-0.08])
        #
        # print(match.full_time())
        # mu_home 主队期望进球数; mu_away 客队期望进球数; home_score 主队当前进球数; away_score 客队当前进球数;
        # adj_mode 赔率调整模式，0-平局调整模式，1-rho调整模式; adj_parameter, 平局调整模式[draw_adj, draw_split]，rho模式

        # mu: [supremacy, total goals]
        # score=[half_time_score,full_time_score]
        # decay=[decay_home, decay_away]
        # parameter=[adj_mode, rho]或者[adj_mode,[draw_adj, draw_split]]
        # clock=[stage, running_time, ht_add, ft_ad]

        matchOdds = cal_match_odds()
        matchOdds.set_value(mu,score, clock, decay, parameter)

        return matchOdds


    # def getSoccerOdds(self):
    #     global soccerIntance
    #
    #     if not soccerIntance == None:
    #         return soccerIntance
    #
    #     mu = [float(self.get_argument("mu0")), float(self.get_argument("mu1"))]
    #     score = [int(self.get_argument("score0")), int(self.get_argument("score1"))]
    #     adjModel = int(self.get_argument("adjModel"))
    #     adjPara0 = float(self.get_argument("adjPara0"))
    #
    #     if adjModel == 1:
    #         soccerIntance = SoccerOdds(mu, score, [adjModel, adjPara0])
    #     else:
    #         adjPara1 = float(self.get_argument("adjPara1"))
    #         soccerIntance = SoccerOdds(mu, score, [adjModel, [adjPara0, adjPara1]])
    #
    #     return soccerIntance;


    #
    #
    # def load_json(self):
    #     """Load JSON from the request body and store them in
    #     self.request.arguments, like Tornado does by default for POSTed form
    #     parameters.
    #
    #     If JSON cannot be decoded, raises an HTTPError with status 400.
    #     """
    #     try:
    #         self.request.arguments = json.loads(self.request.body)
    #     except ValueError:
    #         msg = "Could not decode JSON: %s" % self.request.body
    #         logger.debug(msg)
    #         raise tornado.web.HTTPError(400, msg)
    #
    # def get_json_argument(self, name, default=None):
    #     """Find and return the argument with key 'name' from JSON request data.
    #     Similar to Tornado's get_argument() method.
    #     """
    #     if default is None:
    #         default = self._ARG_DEFAULT
    #     if not self.request.arguments:
    #         self.load_json()
    #     if name not in self.request.arguments:
    #         if default is self._ARG_DEFAULT:
    #             msg = "Missing argument '%s'" % name
    #             logger.debug(msg)
    #             raise tornado.web.HTTPError(400, msg)
    #         logger.debug("Returning default argument %s, as we couldn't find "
    #                 "'%s' in %s" % (default, name, self.request.arguments))
    #         return default
    #     arg = self.request.arguments[name]
    #     logger.debug("Found '%s': %s in JSON arguments" % (name, arg))
    #     return arg
    #
    # def row_to_obj(self, row, cur):
    #     """Convert a SQL row to an object supporting dict and attribute access."""
    #     obj = tornado.util.ObjectDict()
    #     for val, desc in zip(row, cur.description):
    #         obj[desc.name] = val
    #     return obj
