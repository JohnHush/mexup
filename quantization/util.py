
def infer_ttg_exp_from_market( **kwargs ):
    try:
        line = kwargs['line']
        over_odds  = kwargs['over_odds']
        under_odds = kwargs['under_odds']

        eps = kwargs.get( 'eps', 0.005 )

    except KeyError:
        return

    over_margin  = 1. / over_odds
    under_margin = 1. / under_odds

    # target 1 to be matched
    om_normalized = over_margin / ( over_margin+under_margin )

    def _obj( ):
