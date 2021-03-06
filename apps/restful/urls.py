from apps.restful import soccer_handler

url_patterns = [
    (r"/getAllOdds", soccer_handler.SoccerFullTimeHandicapHandler),
    (r"/getTotalGoals", soccer_handler.InferSoccerTotalGoalsHandler),
    (r"/getSupremacy", soccer_handler.InferSoccerSupremacyHandler),
]