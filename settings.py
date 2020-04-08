# encoding: utf-8
import os


from tornado.options import define
from configs import initialize_logging


# 编码设置
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

define("port", default=8001, help="Run server on a specific port", type=int)
define("host", default="localhost", help="Run server on a specific host")

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
LOG_BASE_PATH = os.path.join(ROOT_PATH, "logs")

# the application settings
MEDIA_PATH = os.path.join(ROOT_PATH, 'media')
TEMPLATE_PATH = os.path.join(ROOT_PATH, 'templates')


# 日志初始化
initialize_logging(LOG_BASE_PATH)

# environment setting : develop or production
environment = "develop"
if environment == "develop":
    print('生产环境！！！')
    DEBUG = False

else:
    print('开发环境...')
    DEBUG = True

settings = {
    "debug": DEBUG,
    "static_path": MEDIA_PATH,
    "template_loader": TEMPLATE_PATH,
    "cookie_secret": "xhIBBR2lSp2Pfpx4iiIyX/X6K9j7VUB9oNeA+YdS+ng=",
    # "xsrf_cookies": True,
}