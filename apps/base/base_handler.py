import abc
import tornado.web
from abc import ABC
# from concurrent.futures.thread import ThreadPoolExecutor
from quantization.old.basketball.infer_basketball import infer_basketball_model_input
from quantization.old.soccer.infer_soccer import infer_soccer_model_input


#
# import logging
# logger = logging.getLogger(__name__)
#



class NoResultError(Exception):
    pass


class BaseHandler(tornado.web.RequestHandler, ABC):
    # executor = ThreadPoolExecutor(30)

    # @tornado.gen.coroutine
    def get(self):
        # data = yield self.getData()
        data = self.getData()
        result = {"code": 0, "data": data}
        self.write(result)

    @abc.abstractmethod
    def getData(self):
        pass


    def getInferSoccer(self):
        cal = infer_soccer_model_input();
        return cal;

    def getInferBasketball(self):
        cal = infer_basketball_model_input();
        return cal;
