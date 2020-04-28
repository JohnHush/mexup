import abc
import json
import multiprocessing
import threading
from abc import ABC
from concurrent.futures.thread import ThreadPoolExecutor

import tornado.web


from quantization.infer_soccer_model_input import infer_soccer_model_input
from quantization.basketball.infer_basketball_model_input import infer_basketball_model_input


#
# import logging
# logger = logging.getLogger(__name__)
#



json.encoder.FLOAT_REPR = lambda x: format(x, '.4f')

class NoResultError(Exception):
    pass

def pretty_floats(obj):
    if isinstance(obj, float):
        return round(obj, 5)
    elif isinstance(obj, dict):
        return dict((k, pretty_floats(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return map(pretty_floats, obj)
    return obj

class BaseHandler(tornado.web.RequestHandler, ABC):
    executor = ThreadPoolExecutor(30)

    @tornado.gen.coroutine
    def get(self):
        data = yield self.getData()
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
