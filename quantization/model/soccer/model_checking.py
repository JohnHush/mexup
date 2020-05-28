import pandas as pd
import numpy as np
import os
from quantization.soccer.soccer_inversion import *
from quantization.model.soccer.dixon_coles_model import *

pd.set_option('display.max_columns', None )

def collect_data():
    csv_list = [
        '1011_E0.csv',
        '1112_E0.csv',
        '1213_E0.csv',
        '1314_E0.csv',
        '1415_E0.csv',
        '1516_E0.csv',
        '1617_E0.csv',
        '1718_E0.csv',
        '1819_E0.csv',
        '1920_E0.csv'
    ]

    df_list = []
    for c in csv_list:
        p = os.path.join( os.path.abspath('../..'), c )
        df = pd.read_csv( p )
        df_list.append( df )

    df = pd.concat( df_list )

    df = df[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG',
             'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS',
             'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY',
             'HR', 'AR', 'B365H', 'B365D', 'B365A', 'BbAv>2.5',
             'BbAv<2.5', 'BbAHh', 'BbAvAHH', 'BbAvAHA', 'Avg>2.5',
             'Avg<2.5', 'AHh', 'AvgAHH', 'AvgAHA' ] ]

    df.loc[ np.isnan( df['BbAv>2.5'] ), 'BbAv>2.5' ] = df.loc[ np.isnan( df['BbAv>2.5'] ), 'Avg>2.5' ]
    df.loc[ np.isnan( df['BbAv<2.5'] ), 'BbAv<2.5' ] = df.loc[ np.isnan( df['BbAv<2.5'] ), 'Avg<2.5' ]
    df.loc[ np.isnan( df['BbAHh'] ), 'BbAHh' ] = df.loc[ np.isnan( df['BbAHh'] ), 'AHh' ]
    df.loc[ np.isnan( df['BbAvAHH'] ), 'BbAvAHH' ] = df.loc[ np.isnan( df['BbAvAHH'] ), 'AvgAHH' ]
    df.loc[ np.isnan( df['BbAvAHA'] ), 'BbAvAHA' ] = df.loc[ np.isnan( df['BbAvAHA'] ), 'AvgAHA' ]

    df.drop( ['Avg>2.5', 'Avg<2.5', 'AHh', 'AvgAHH', 'AvgAHA'] , axis=1, inplace=True )
    df.dropna( axis=0 , inplace=True )

    df.to_csv( '../../england_10years_data.csv' , index=False )


def infer_sup_ttg_from_market():
    df = pd.read_csv( '../../england_10years_data.csv')
    # df = df[0:20]

    def sup_ttg_apply_fn( series ):
        config = InferSoccerConfig()
        config.ou_line = 2.5
        config.ahc_line = series['BbAHh']
        config.scores = [0, 0]
        config.over_odds = series['BbAv>2.5']
        config.under_odds = series['BbAv<2.5']
        config.home_odds = series['BbAvAHH']
        config.away_odds = series['BbAvAHA']

        return infer_ttg_sup(config)

    df['tmp'] = df.apply( sup_ttg_apply_fn , axis=1 )
    df['sup_market'] = df['tmp'].map( lambda s: round(s[0], 5) )
    df['ttg_market'] = df['tmp'].map( lambda s: round(s[1], 5) )

    df.drop( ['tmp'], axis=1, inplace=True )
    df.to_csv( '../../england_10years_data_sup_ttg_market.csv', index=False )


def check_dixon_coles_model_with_without_gradient():
    ds = UrlData(url=['../../1920_E0.csv'])
    dcm = DixonColesModel(ds)

    dcm.solve()
    # dcm.save_model('./england_league0_1920_dcm2.model')

    # dcm.load_model( './england_league0_1920_dcm.model' )
    # home_exp, away_exp, rho = dcm.infer_exp_rho('Arsenal', 'Southampton')
    # print( home_exp, away_exp, rho )
    # home_exp, away_exp, rho = dcm.infer_exp_rho('Chelsea', 'Watford')
    # print( home_exp, away_exp, rho )
    # home_exp, away_exp, rho = dcm.infer_exp_rho('Chelsea', 'Man City')
    # print( home_exp, away_exp, rho )
    #
    print(dcm.model.x)


# collect_data()
# infer_sup_ttg_from_market()
check_dixon_coles_model_with_without_gradient()