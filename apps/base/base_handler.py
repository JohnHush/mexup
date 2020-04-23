from abc import ABC

import tornado.web

from quantization.infer_soccer_model_input import infer_soccer_model_input
from quantization.basketball.infer_basketball_model_input import infer_basketball_model_input


#
# import logging
# logger = logging.getLogger(__name__)
#


class NoResultError(Exception):
    pass


class BaseHandler(tornado.web.RequestHandler, ABC):

    def getInferSoccer(self):
        cal = infer_soccer_model_input();
        return cal;

    def getInferBasketball(self):
        cal = infer_basketball_model_input();
        return cal;
