from quantization.model.soccer.model import DynamicDixonModelV2
from quantization.model.soccer.data import E0Data
import numpy as np
import pymongo

class BackTesting():
    @staticmethod
    def ddm_back_testing():
        data = E0Data()
        # data.from_csv( caches='../../data_e0_cache.csv' )
        data.from_sql()

        ddm = DynamicDixonModelV2()
        ddm.data_preprocessing(data=data,
                               aux_tar=[['HST', 'AST']])

        id = 900
        epsilon_list = np.arange( 0., 0.003, 0.0001 )
        duration_list = range( 1080 , 1120 , 10 )
        league_weight_list = np.arange( 4.5, 5.5, 0.1 )
        aux_weight_list = np.arange( 0.4, 0.6, 0.02 )
        # epsilon = 0.0018571,

        # attack_weight_list = [ 0., 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30. ]
        attack_weight_list = np.arange( 0.00003, 0.0001, 0.00003 )

        # for epsilon in epsilon_list:
        # for duration in duration_list:
        # for league_weight in league_weight_list:
        # for aux_weight in aux_weight_list:
        for attack_weight in attack_weight_list:
        # for x in [1,]:
            ddm.inverse( window_size=1,
                         epsilon=0.0015,
                         duration=1100,
                         aux_weight=0.52,
                         league_weight=[4.7, 1.],
                         l2_attack_weight=0.00009,
                         l2_defend_weight=.03 )

            ddm.save_to_mongodb( id=id,
                                 db_name='all_models_db',
                                 collection_name='dynamic_dixon_coles_model' )

            id += 1

    @staticmethod
    def compute_rmse():
        data = E0Data()
        # data.from_csv( caches='../../data_e0_cache.csv' )
        data.from_sql()

        ddm = DynamicDixonModelV2()
        ddm.data_preprocessing(data=data,
                               aux_tar=[['HST', 'AST']])

        client = pymongo.MongoClient(host='localhost', port=27017 )
        dbs = client['all_models_db']
        ddm_collection = dbs['dynamic_dixon_coles_model']

        cursor = ddm_collection.find( { 'model_metrics' : {'$exists' : False } } )

        for c in cursor:
            ddm.load_from_dict( c )

            l1e, l2e, pick_list = ddm._rmse_model()
            # print( ddm._rmse_model() )
            print(c['model_id'])
            print(l1e, l2e)
            print(ddm._rmse_naive_mean_model())
            print(ddm._rmse_market(pick_list=pick_list))

            model_metrics = {}
            model_metrics['lambda1_rmse'] = l1e
            model_metrics['lambda2_rmse'] = l2e

            c['model_metrics'] = model_metrics
            ddm_collection.update_one( { '_id' : c['_id'] },
                                       { '$set' : c })

if __name__ =='__main__':
    # BackTesting.ddm_back_testing()
    BackTesting.compute_rmse()