import matplotlib.pyplot as plt
from quantization.model.soccer.data import E0Data
from quantization.model.soccer.model import StaticDixonModel
import pandas as pd
import numpy as np

class PLOT(object):
    @staticmethod
    def plot_sdm_backtesting():
        data = E0Data()
        data.from_csv(caches='../../data_e0_cache.csv')
        sdm = StaticDixonModel()
        sdm.data_preprocessing(data=data, split_time=pd.Timestamp(2012, 2, 15))
        # sdm.inverse()
        # sdm.save_model( fn='tt2.model')
        sdm.load_model(fn='tt.model')

        df = sdm.back_testing( train_or_test='test',
                               lo=0.,
                               hi=100. )

        missing_ratio = len(df[ df.pnl_model != 0 ]) / len(df)

        model_sum_list = []
        random_sum_list = []
        index_list = np.arange( 0.93, 1.2, 0.03 )
        index_list = list( map( lambda s: round( s,2 ), index_list) )

        bet_ratio_list = []

        for lo in index_list:
            df = sdm.back_testing(train_or_test='test',
                                  lo=lo,
                                  hi=100.)

            tmp = len(df[ df.pnl_model != 0 ]) / len(df)
            bet_ratio = tmp / missing_ratio
            print(bet_ratio)

            bet_ratio_list.append( round(bet_ratio, 2) )

            sum_val_list = []
            for _ in range(100):
                sum_val = sdm.back_testing_random_model( train_or_test='test',
                                                         ratio=bet_ratio )
                sum_val_list.append( sum_val )

            random_sum_list.append( sum_val_list )
            model_sum_list.append( df.pnl_model.sum() / len(df[ df.pnl_model != 0 ]) )

        index_show_list = []

        for i,j in zip( index_list, bet_ratio_list ):
            index_show_list.append( str(i) + ' / ' + str(j) )

        fig, ax = plt.subplots()
        ax.violinplot( random_sum_list, positions=range( len(index_list) ) )
        ax.plot( range( len(index_list) ), model_sum_list , label='Static Dixon Model' )
        plt.xticks( range( len(index_list) ), index_show_list )
        plt.xlabel('low limit of betting / bet sample ratio')
        plt.ylabel('bet value mean')
        plt.legend(loc=2)
        plt.show()

    @staticmethod
    def plot_something():
        data = E0Data()
        data.from_csv(caches='../../data_e0_cache.csv')
        sdm = StaticDixonModel()

        # time_start = pd.Timestamp(2011, 7, 15)
        time_start = pd.Timestamp(2012, 2, 15)
        time_delta = pd.Timedelta(days=365)

        t = time_start

        l1_model = []
        l2_model = []
        l1_market = []
        l2_market = []
        l1_nmm = []
        l2_nmm = []

        time_list = []
        time_list.append(t)

        for _ in range(2):
            sdm.data_preprocessing(data=data, split_time=t)
            sdm.inverse()

            l1, l2 = sdm._rmse_naive_mean_model(train_or_test='test')
            l1_nmm.append(l1)
            l2_nmm.append(l2)

            l1, l2 = sdm._rmse_model(train_or_test='test')
            l1_model.append(l1)
            l2_model.append(l2)

            l1, l2 = sdm._rmse_market(train_or_test='test')
            l1_market.append(l1)
            l2_market.append(l2)

            t = t + time_delta
            time_list.append(t)

        time_list.pop()

        fig, ax = plt.subplots()
        ax.plot(time_list, l1_nmm, label='naive mean model', marker='o', linestyle=':')
        ax.plot(time_list, l1_market, label='bookie model', marker='+', linestyle=':')
        ax.plot(time_list, l1_model, label='Dixon model', marker='*', linestyle=':')

        plt.xlabel('time')
        plt.ylabel('rmse')

        plt.legend(loc=2)
        plt.show()

if __name__=='__main__':
    # PLOT.plot_back_testing_result()
    PLOT.plot_sdm_backtesting()
    # PLOT.plot_something()
