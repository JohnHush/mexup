from apps.restful import soccor_handler

url_patterns = [
    # (r"/had", soccor_handler.HadHandler),
    # (r"/asian_handicap", soccor_handler.AsianHandicapHandler),
    (r"/fullTime", soccor_handler.ScocorFullTimeHandicapHandler),
    #(r"/getMu", soccor_handler.InferSoccerMuHandler),
    (r"/getTotalGoals", soccor_handler.InferSoccerTotalGoalsHandler),
    (r"/getSupremacy", soccor_handler.InferSoccerSupremacyHandler),
]