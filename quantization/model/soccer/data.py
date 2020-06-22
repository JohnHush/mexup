from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
from quantization.soccer.soccer_inversion import *

class Data( metaclass=ABCMeta ):
    '''
    base class for data
    '''

    @abstractmethod
    def from_raw_data(self, **kwargs):
        pass

    @abstractmethod
    def from_csv(self, **kwargs):
        pass

    @abstractmethod
    def to_csv(self, **kwargs):
        pass

    @abstractmethod
    def get_df(self, **kwargs):
        pass

    @abstractmethod
    def get_train_df(self, **kwargs):
        pass


class E0Data(Data):
    '''
    England League Division 0 matches
    '''
    def __init__(self):
        self._df = None

    def get_df(self, **kwargs):
        if self._df is not None:
            return self._df

        return None

    def get_train_df(self, **kwargs):
        '''
        this train df will be returned as the same as get_df
        Args:
            **kwargs:

        Returns:

        '''
        if self._df is not None:
            return self._df

        return None

    def to_csv(self, **kwargs):
        caches = kwargs.pop( 'caches', None )
        if caches is None:
            raise Exception('must specify caches name')
        if kwargs:
            raise TypeError('unexpected kwargs')

        if self._df is not None:
            self._df.to_csv( caches, index=False)

    def from_csv(self, **kwargs):
        caches = kwargs.pop( 'caches', None )
        if caches is None:
            raise Exception('must specify caches name')
        if kwargs:
            raise TypeError('unexpected kwargs')

        self._df = pd.read_csv(caches,
                               date_parser=lambda s: pd.datetime.strptime(s, '%Y-%m-%d'),
                               infer_datetime_format=True,
                               parse_dates=['Date', ])

    def from_raw_data(self, **kwargs):
        raw_files = kwargs.pop( 'raw_files', None )

        if raw_files is None:
            raise Exception('need to specify the raw files name!')

        if kwargs:
            raise TypeError('unexpected **kwargs: %r' % kwargs )

        def date_parser(s):
            try:
                r = pd.datetime.strptime(s, '%d/%m/%y')
            except:
                r = pd.datetime.strptime(s, "%d/%m/%Y")
            else:
                pass
            return r

        df_list = [pd.read_csv(c,
                               date_parser=date_parser,
                               infer_datetime_format=True,
                               parse_dates=['Date', ]
                               ) for c in raw_files ]

        df = pd.concat(df_list, sort=True)
        df = df[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG',
                 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS',
                 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY',
                 'HR', 'AR', 'B365H', 'B365D', 'B365A', 'BbAv>2.5',
                 'BbAv<2.5', 'BbAHh', 'BbAvAHH', 'BbAvAHA', 'Avg>2.5',
                 'Avg<2.5', 'AHh', 'AvgAHH', 'AvgAHA']]

        df.loc[np.isnan(df['BbAv>2.5']), 'BbAv>2.5'] = df.loc[np.isnan(df['BbAv>2.5']), 'Avg>2.5']
        df.loc[np.isnan(df['BbAv<2.5']), 'BbAv<2.5'] = df.loc[np.isnan(df['BbAv<2.5']), 'Avg<2.5']
        df.loc[np.isnan(df['BbAHh']), 'BbAHh'] = df.loc[np.isnan(df['BbAHh']), 'AHh']
        df.loc[np.isnan(df['BbAvAHH']), 'BbAvAHH'] = df.loc[np.isnan(df['BbAvAHH']), 'AvgAHH']
        df.loc[np.isnan(df['BbAvAHA']), 'BbAvAHA'] = df.loc[np.isnan(df['BbAvAHA']), 'AvgAHA']

        df.drop(['Avg>2.5', 'Avg<2.5', 'AHh', 'AvgAHH', 'AvgAHA'], axis=1, inplace=True)
        df.dropna(axis=0, inplace=True)
        df = df.sort_values(by='Date')

        def _infer_market_sup_ttg_fn(series):
            config = InferSoccerConfig()
            config.ou_line = 2.5
            config.ahc_line = series['BbAHh']
            config.scores = [0, 0]
            config.over_odds = series['BbAv>2.5']
            config.under_odds = series['BbAv<2.5']
            config.home_odds = series['BbAvAHH']
            config.away_odds = series['BbAvAHA']

            return infer_ttg_sup(config)

        df['tmp'] = df.apply(_infer_market_sup_ttg_fn, axis=1)
        df['sup_market'] = df['tmp'].map(lambda s: round(s[0], 5))
        df['ttg_market'] = df['tmp'].map(lambda s: round(s[1], 5))

        df.drop(['tmp'], axis=1, inplace=True)
        self._df = df


if __name__=='__main__':
    data = E0Data()
    # data.from_csv( caches='../../data_e0_cache.csv')
    raw_files = [
        '../../1011_E0.csv',
        '../../1112_E0.csv',
        '../../1213_E0.csv',
        '../../1314_E0.csv',
        '../../1415_E0.csv',
        '../../1516_E0.csv',
        '../../1617_E0.csv',
        '../../1718_E0.csv',
        '../../1819_E0.csv',
        '../../1920_E0.csv'
    ]
    data.from_raw_data( raw_files=raw_files )
    df = data.get_df()
    print( df.head() )

