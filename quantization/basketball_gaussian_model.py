import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import norm
pd.set_option( 'display.max_columns', None )

class BasGaussianModel( object ):
    def __init__(self):
        pass

    def prepareDataset(self, *, xls=None, random_seed=0xCAFFE ):
        if not xls:
            raise ValueError('must specify xls files')
        self.df = pd.read_excel(xls)

        def map_with_draw( s ):
            if not 'OT' in s:
                return s
            return s.split('(')[0].strip()

        def map_no_draw( s ):
            if not 'OT' in s:
                return s
            return s.split( ' ' )[-1][:-1]

        self.df['Score_with_draw'] = self.df.Score.map( map_with_draw )
        self.df['Score_no_draw']   = self.df.Score.map( map_no_draw )
        self.df = self.df.drop( ['Score'], axis=1 )

        # split the whole data into train and test part
        X_train, X_test , y_train, y_test = train_test_split( self.df,
                                                              np.ones(len(self.df)),
                                                              test_size=0.25,
                                                              random_state=random_seed )

        # split data into team-specified category
        self.train_data = self._collect_score_dict( X_train )
        self.test_data  = self._collect_score_dict( X_test )

    def _collect_score_dict( self, df: pd.DataFrame ):
        key_set = set( self.df['Home Team'] )

        d = {}
        for k in key_set:
            home_score = df[ df['Home Team']==k ]['Score_with_draw'].map(\
                lambda s: int(s.split(':')[0]) ).to_numpy()
            away_score = df[ df['Away Team']==k ]['Score_with_draw'].map(\
                lambda s: int(s.split(':')[1]) ).to_numpy()

            k_score = np.append( home_score , (away_score,) )
            d[k] = k_score

        return d

    def paramsEvaluation(self):
        self.params = {}

        for k, v in self.train_data.items():
            mean_evaluated = np.mean( v )
            std_evaluated  = np.std( v )

            self.params[k] = [ mean_evaluated, std_evaluated ]

    def log_likelihood_category(self, ds='test' ):
        if ds == 'train':
            dataset = self.train_data
        else:
            dataset = self.test_data

        llk_dict = {}
        for k, v in dataset.items():
            llk = 0.
            count = 0
            miu = self.params[k][0]
            std = self.params[k][1]

            for vv in v:
                # iterate every item in value
                llk += np.log( norm.pdf( vv, loc=miu, scale=std ) )
                count += 1

            llk_dict[k] = llk / count

        return llk_dict

    def log_likelihood(self, ds='test' ):
        if ds == 'train':
            dataset = self.train_data
        else:
            dataset = self.test_data

        llk = 0.
        count = 0
        for k, v in dataset.items():
            miu = self.params[k][0]
            std = self.params[k][1]

            for vv in v:
            # iterate every item in value
                llk += np.log( norm.pdf( vv, loc=miu, scale=std ) )
                count += 1

        return llk/count

if __name__ == '__main__':
    M = BasGaussianModel()
    M.prepareDataset( xls='StartClosePrices-5.xls' )
    M.paramsEvaluation()

    llk_test = M.log_likelihood()
    llk_train = M.log_likelihood( ds='train' )
    print('test : ', llk_test)
    print('train : ', llk_train)

    llk_test_dict = M.log_likelihood_category()
    llk_train_dict = M.log_likelihood_category( ds='train' )
    for k,v in llk_test_dict.items():
        print( k,llk_train_dict[k] , v )
    # for k,v in llk_train_dict.items():
    #     print( k,v )
