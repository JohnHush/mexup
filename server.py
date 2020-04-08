import tornado.ioloop
import tornado.web
import tornado.httpserver
import sys
import settings

from urls import urls_patterns as url_handlers
from tornado.options import options
import multiprocessing;

class Application(tornado.web.Application):
    def __init__(self, handlers, **setting):
        tornado.web.Application.__init__(self, handlers, **setting)


application = Application(url_handlers, **settings.settings)


def main():
    try:
        print(f"Starting server at: {'http://' + options.host + ':' + str(options.port)}")
        options.parse_command_line()

        http_server = tornado.httpserver.HTTPServer(application)
        http_server.bind(options.port)
        http_server.start(multiprocessing.cpu_count())

        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("Server got to stop.")
    except:
        import traceback
        print(traceback.print_exc())
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()

