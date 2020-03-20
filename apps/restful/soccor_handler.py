from abc import ABC

from apps.base.base_handler import BaseHandler


class HadHandler(BaseHandler, ABC) :

    def get(self):
        soccerOdds = self.getSoccerOdds()
        result = soccerOdds.had()

        # soccerOdds.asian_handicap(-0.5)
        # soccerOdds.over_under(2.5)
        # soccerOdds.exact_totals(5)
        # soccerOdds.correct_score(3, 1)
        # soccerOdds.double_chance_over_under(2.5)
        # soccerOdds.home_over_under(0.75)
        # soccerOdds.away_over_under(1.5)
        # soccerOdds.home_exact_totals(3)
        # soccerOdds.away_exact_totals(0)
        # soccerOdds.home_winning_by(2)
        # soccerOdds.away_winning_by(1)
        # soccerOdds.both_scored()
        # soccerOdds.odd_even()

        self.write(','.join(str(i) for i in result))


class AsianHandicapHandler(BaseHandler, ABC) :

    def get(self):
        soccerOdds = self.getSoccerOdds();
        line = self.get_argument("line")
        self.write((soccerOdds.asian_handicap(line)))

