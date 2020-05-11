from apps.restful import soccer_handler
from apps.restful import basketball_handler

url_patterns = [
    (r"soccer/getAllOdds", soccer_handler.SoccerOddsHandler),
    (r"soccer/getTotalGoals", soccer_handler.SoccerInferTotalGoalsHandler),
    (r"soccer/getSupremacy", soccer_handler.SoccerInferSupremacyHandler),

    (r"basketball/getAllOdds", basketball_handler.BasketballOddsHandler),
    (r"basketball/getTotalGoals", basketball_handler.BasketballInferTotalGoalsHandler),
    (r"basketball/getSupremacy", basketball_handler.BasketballInferSupremacyHandler),


    (r"v2/soccer/getSupTtg", soccer_handler.V2SoccerSupTtgHandler),
    (r"v2/soccer/getAllOdds", soccer_handler.V2SoccerOddsHandler),

    (r"v2/basketball/getSupTtg", basketball_handler.V2BasketballSupTtgHandler),
    (r"v2/basketball/getAllOdds", basketball_handler.V2BasketballOddsHandler),

]