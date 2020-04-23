from apps.restful import soccer_handler
from apps.restful import basketball_handler

url_patterns = [
    (r"soccer/getAllOdds", soccer_handler.SoccerOddsHandler),
    (r"soccer/getTotalGoals", soccer_handler.SoccerInferTotalGoalsHandler),
    (r"soccer/getSupremacy", soccer_handler.SoccerInferSupremacyHandler),

    (r"basketball/getAllOdds", basketball_handler.BasketballOddsHandler),
    (r"basketball/getTotalGoals", basketball_handler.BasketballInferTotalGoalsHandler),
    (r"basketball/getSupremacy", basketball_handler.BasketballInferSupremacyHandler),
]