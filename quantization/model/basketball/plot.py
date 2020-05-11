from quantization.model.basketball.basketball_binomial_model import BasBinomialModel
from quantization.model.basketball.basketball_gaussian_model import BasGaussianModel
from quantization.model.basketball.basketball_poisson_model import BasPoissonModel

import matplotlib.pyplot as plt

def plot_basketball_model_comparison():
    M1 = BasBinomialModel()
    M2 = BasGaussianModel()
    M3 = BasPoissonModel()

    M1_train = []
    M2_train = []
    M3_train = []

    M1_test = []
    M2_test = []
    M3_test = []

    for i in range( 50 ):
        print( "iteration : ", i )
        M1.prepareDataset(xls='StartClosePrices-5.xls', random_seed=i )
        M1.paramsEvaluation()
        M2.prepareDataset(xls='StartClosePrices-5.xls', random_seed=i )
        M2.paramsEvaluation()
        M3.prepareDataset(xls='StartClosePrices-5.xls', random_seed=i )
        M3.paramsEvaluation()

        M1_train.append( M1.log_likelihood(ds='train') )
        M1_test.append ( M1.log_likelihood() )
        M2_train.append( M2.log_likelihood(ds='train') )
        M2_test.append ( M2.log_likelihood() )
        M3_train.append( M2.log_likelihood(ds='train') )
        M3_test.append ( M3.log_likelihood() )

    fig, ax = plt.subplots()
    ax.violinplot( [M1_train, M1_test, M2_train, M2_test, M3_train, M3_test ],
                   showmeans=False,
                   showmedians=True )
    plt.show()

    # llk_test_dict = M.log_likelihood_category()
    # llk_train_dict = M.log_likelihood_category(ds='train')
    # for k, v in llk_test_dict.items():
    #     print(k, llk_train_dict[k], v)


if __name__=='__main__':
    plot_basketball_model_comparison()