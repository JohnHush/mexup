#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:48:50 2020

@author: frank

模型结果的展示和比较

第一步：数据分区，包括训练和测试
第二步：比赛得分和胜负
    模型需要的参数确定之后，输入每场比赛的结果，然后记录在队伍的输出结果里面
    队伍的输出结果包含比分的似然函数，胜负的似然函数
第三部：结果记录输出和比较
第四部：优化代码
"""
#导入模块
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import basketball_gamma_model as GT
from scipy.stats import binom,poisson,norm,gamma

lams = [] #参数
Teams = {} #队伍
df = pd.read_csv('/Users/frank/Documents/篮球Gamma模型/数据/NBA-PBP_2016-2017.csv')
game_num = df[df['GameType']=='regular']['URL'].value_counts()
for _ in range(0,int(len(game_num)*0.5)):
    game_k = df[df['URL'].isin([game_num.index[_]])]
    df_game = GT.Match.data_process(game_k)
    match = GT.Match(df_game)
    # match.count_lam()
    lams.append(match.lam)
    if match.home_team not in Teams.keys():
        #没有队伍类的时候，新建一个队伍实例
        key = match.home_team
        print(f'主队是{key}')
        value = GT.Team(key)
        value.fresh_data(df_game)
        Teams[key]=value
    else:#有队伍类的时候，对队伍实例进行数据更新
        key = match.home_team
        print(f'主队是{key},更新数据')
        Teams[key].fresh_data(df_game)
    if match.away_team not in Teams.keys():
        key = match.away_team
        print(f'客队是{key}')
        value = GT.Team(key)
        value.fresh_data(df_game)
        Teams[key]=value
    else:
        key = match.away_team
        print(f'客队是{key},更新数据')
        Teams[key].fresh_data(df_game)

#计算lam和各类参数
lam_mean = np.mean(lams)
lam_var = np.var(lams)
alpha = lam_mean/lam_var
beta = (lam_mean**2)/lam_var
#估算参数
for _ in Teams:
    Teams[_].para_est(alpha,beta)
    Teams[_].para_other()
#生成似然函数值：Cohen版
def likelihood(name,score,team,home=1):
    alpha = 94.8084478383912
    beta = 122.27217645214549
    para_bino = team.para_bino
    para_pois = team.para_pois
    para_gaus = team.para_gaus

    if home == 1:#主场
        para_gamma = [team.home_mu,alpha/beta]
    else:
        para_gamma = [team.away_mu,alpha/beta]
    
    vv = score
    llk_binom = binom.logpmf(vv,para_bino[1],para_bino[0])
    llk_poisson = poisson.logpmf(vv,para_pois[0])
    llk_gaus = norm.logpdf(vv,loc=para_gaus[0],scale = np.sqrt(para_gaus[1]))
    llk_gamma = gamma.logpdf(vv,a=48*para_gamma[0],scale = para_gamma[1])

    try:
        team.result.append([llk_binom,llk_poisson,llk_gaus,llk_gamma])
    except :
        team.result = [[llk_binom,llk_poisson,llk_gaus,llk_gamma]]
#计算胜负预测和真实结果
def winratio(fake_home,fake_away):
    return ((fake_home>fake_away).sum())/len(fake_home)

def model_winratio(team,team2,monte_carlo_num = 10000):
    mu_home = team.home_mu
    mu_away = team2.away_mu
    alpha = 94.8084478383912
    beta = 122.27217645214549
    lam_bayes = beta/alpha
    fake_home_gamma=[]
    fake_away_gamma=[]
    for _ in range(monte_carlo_num):
        _home = np.rint(np.random.gamma(mu_home,1/lam_bayes,48))
        _away = np.rint(np.random.gamma(mu_away,1/lam_bayes,48))
        fake_home_gamma.append(_home.sum())
        fake_away_gamma.append(_away.sum())
    gamma_ratio = winratio(np.array(fake_home_gamma),np.array(fake_away_gamma))
    
    fake_home_binom = np.random.binomial(team.para_bino[1],team.para_bino[0],monte_carlo_num)
    fake_away_binom = np.random.binomial(team2.para_bino[1],team2.para_bino[0],monte_carlo_num)
    binom_ratio = winratio(fake_home_binom,fake_away_binom)
    
    fake_home_pois = np.random.poisson(team.para_pois[0],monte_carlo_num)
    fake_away_pois = np.random.poisson(team2.para_pois[0],monte_carlo_num)
    pois_ratio = winratio(fake_home_pois,fake_away_pois)

    fake_home_gaus = np.random.normal(team.para_gaus[0],
                                  np.sqrt(team.para_gaus[1]),monte_carlo_num)
    fake_away_gaus = np.random.normal(team2.para_gaus[0],
                                  np.sqrt(team2.para_gaus[1]),monte_carlo_num)
    guas_ratio = winratio(fake_home_gaus,fake_away_gaus)

    return [gamma_ratio,binom_ratio,pois_ratio,guas_ratio]



count = 1
df_home = []
df_away = []
df_win = []
df_is_homewin = []
df_gamma =[]
df_binm =[]
df_pois =[]
df_gaus = []
for _ in range(int(len(game_num)*0.5)+1,int(len(game_num))):
    game_k = df[df['URL'].isin([game_num.index[_]])]
    df_game = GT.Match.data_process(game_k)
    match = GT.Match(df_game)

    # 分主队，客队的情况
    Team_Home = df_game.iloc[0]['HomeTeam']
    Team_Away = df_game.iloc[0]['AwayTeam']
    Score_Home =df_game.iloc[-1]['HomeScore']
    Score_Away = df_game.iloc[-1]['AwayScore']
    Team_Win = df_game.iloc[0]['WinningTeam']
    df_home.append(Team_Home)
    df_away.append(Team_Away)
    df_win.append(Team_Win)
    if Team_Home == Team_Win:
        df_is_homewin.append(1)
    else:
        df_is_homewin.append(0)
    _ratio = model_winratio(Teams[Team_Home],Teams[Team_Away])
    df_gamma.append(_ratio[0])
    df_binm.append(_ratio[1])
    df_pois.append(_ratio[2])
    df_gaus.append(_ratio[3])
    #各个模型
    #二项分布：binomial，参数llk += np.log( binom.pmf( vv, self._N, p_ ) )
    #柏松分布：poisson,参数llk += np.log( poisson.pmf( vv, lambda_ ) )
    #高斯分布：gaussion,参数llk += np.log( norm.pdf( vv, loc=miu, scale=std ) )std=np.sqrt(10)
    #gamma分布：gamma.logpdf(vv,mu_home,scale=lam_bayes)
    #入参:para_bino,para_pois,para_gaus,home_mu,away_mu,lam_bayes = beta/alpha
    likelihood(Team_Home,Score_Home,Teams[Team_Home],home=1)
    likelihood(Team_Away,Score_Away,Teams[Team_Away],0)
    #dataframe生成
    
    
df_win_matrix = pd.DataFrame({'home_team': df_home,
                              'away_team': df_away,
                              'win_team': df_win,
                              'is_homewin': df_is_homewin,
                              'gamma': df_gamma,
                              'binom': df_binm,
                              'poisson':df_pois,
                              'guassi':df_gaus})
#统计特征1,猜中的概率，2.猜中的似然函数
# df_win_matrix.loc[:,['is_homewin','gamma']]\
#               .apply(lambda x: 1 if (x[0])==1 & x[1]>0.5)|(x[0]==0 & x[1]<0.5) else 0)
#               .sum()
#猜中的概率radio,似然函数likehood
#计算每个模型猜中次数
def fo(x,column='gamma'):
    count = 0
    if x['is_homewin'].sum() == len(x['is_homewin']):
        count += (x[column]>0.5).sum()
    else:
        count += (x[column]<0.5).sum()
    return count
#计算每个模型似然函数
def fo2(x,column = 'gamma'):
    likelihood = np.power(df_win_matrix[column],df_win_matrix['is_homewin'])\
        *np.power(1-df_win_matrix[column],1-df_win_matrix['is_homewin'])
    likelihood = np.power(likelihood.prod(),1/len(likelihood))
    return likelihood

## 计算猜中的概率
df_win_matrix.groupby('is_homewin').apply(fo,column = 'gamma').sum()
df_win_matrix.groupby('is_homewin').apply(fo,column = 'binom').sum()
df_win_matrix.groupby('is_homewin').apply(fo,column = 'poisson').sum()
df_win_matrix.groupby('is_homewin').apply(fo,column = 'guassi').sum()
## 计算胜负的似然函数
df_win_matrix.apply(fo2)
df_win_matrix.apply(fo2,column='binom')
df_win_matrix.apply(fo2,column='poisson')
df_win_matrix.apply(fo2,column='guassi')



Final_result = {}
for i in Teams:
    Final_result[i] = np.mean(Teams[i].result,axis=0)
    
print(Final_result)

#可视化结果
df_result = pd.DataFrame(Final_result,index=['binom','poisson','norm','gamma'])
plt.bar(range(len(df_result['ATL'])),list(df_result['IND']))

