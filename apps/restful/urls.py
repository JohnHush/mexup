from apps.restful import soccor_handler

url_patterns = [
    (r"/had", soccor_handler.HadHandler),
    (r"/asian_handicap", soccor_handler.AsianHandicapHandler),
]