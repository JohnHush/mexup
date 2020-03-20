from abc import ABC

import tornado.web

from apps.quantization.soccer import SoccerOdds

#
# import logging
# logger = logging.getLogger(__name__)
#


class NoResultError(Exception):
    pass

#todo 待优化
soccerIntance = None

class BaseHandler(tornado.web.RequestHandler, ABC):


    def getInstanceKey(self,mu,score,adjModel,adjPara0,adjPara1):
        if adjModel == 1:
            return "{}-{}-{}-{}-{}-{}".format(mu[0],mu[1],score[0],score[1],adjModel,adjPara0)
        else:
            return "{}-{}-{}-{}-{}-{}-{}".format(mu[0],mu[1],score[0],score[1],adjModel,adjPara0,adjPara1)


    def getSoccerOdds(self):
        global soccerIntance

        if not soccerIntance == None:
            return soccerIntance

        mu = [float(self.get_argument("mu0")), float(self.get_argument("mu1"))]
        score = [int(self.get_argument("score0")), int(self.get_argument("score1"))]
        adjModel = int(self.get_argument("adjModel"))
        adjPara0 = float(self.get_argument("adjPara0"))

        if adjModel == 1:
            soccerIntance = SoccerOdds(mu, score, [adjModel, adjPara0])
        else:
            adjPara1 = float(self.get_argument("adjPara1"))
            soccerIntance = SoccerOdds(mu, score, [adjModel, [adjPara0, adjPara1]])

        return soccerIntance;


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
