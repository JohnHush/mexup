from quantization.model.soccer.dixon_coles_model import DixonColesModel, UrlData

class TestClass( object ):
    def test_infer_team_strength(self):
        ds = UrlData(url=  [ './1920_E0.csv' , './1819_E0.csv', './1718_E0.csv' ] )
        dcm = DixonColesModel(ds)

        dcm.load_model('./EnglandPremierLeague_17181920_dcm.model')

        # print( ds.encoder.categories_[0] )
        # dcm.infer_team_strength( 'Arsenal' )

        team_strength_dict = {}

        for team in ds.encoder.categories_[0]:
            team_strength_dict[team] = dcm.infer_team_strength( team )

        team_strength_dict_sorted = sorted( team_strength_dict.items(), key = lambda kv: ( kv[1], kv[0]) , reverse=True)

        for item in team_strength_dict_sorted:
            print( item[0], item[1] )

        # print( team_strength_dict )
