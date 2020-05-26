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
    match.count_lam()
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
for _ in Teams:
    Teams[_].para_est(alpha,beta)
    Teams[_].para_other()

#生成似然函数值
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
    llk_gaus = norm.logpdf(vv,loc=para_gaus[0],scale = para_gaus[1])
    llk_gamma = gamma.logpdf(vv,a=48*para_gamma[0],scale = para_gamma[1])

    try:
        team.result.append([llk_binom,llk_poisson,llk_gaus,llk_gamma])
    except :
        team.result = [[llk_binom,llk_poisson,llk_gaus,llk_gamma]]



count = 1
for _ in range(int(len(game_num)*0.5)+1,int(len(game_num))):
    game_k = df[df['URL'].isin([game_num.index[_]])]
    df_game = GT.Match.data_process(game_k)
    match = GT.Match(df_game)

    # 分主队，客队的情况
    Team_Home = df_game.iloc[0]['HomeTeam']
    Team_Away = df_game.iloc[0]['AwayTeam']
    Score_Home =df_game.iloc[-1]['HomeScore']
    Score_Away = df_game.iloc[-1]['AwayScore']
    #各个模型
    #二项分布：binomial，参数llk += np.log( binom.pmf( vv, self._N, p_ ) )
    #柏松分布：poisson,参数llk += np.log( poisson.pmf( vv, lambda_ ) )
    #高斯分布：gaussion,参数llk += np.log( norm.pdf( vv, loc=miu, scale=std ) )
    #gamma分布：gamma.logpdf(vv,mu_home,scale=lam_bayes)
    #入参:para_bino,para_pois,para_gaus,home_mu,away_mu,lam_bayes = beta/alpha
    likelihood(Team_Home,Score_Home,Teams[Team_Home],home=1)
    
    likelihood(Team_Away,Score_Away,Teams[Team_Away],0)
    

Final_result = {}
for i in Teams:
    Final_result[i] = np.mean(Teams[i].result,axis=0)
    
print(Final_result)

#可视化结果
df_result = pd.DataFrame(Final_result,index=['binom','poisson','norm','gamma'])


plt.bar(range(len(df_result['ATL'])),list(df_result['IND']))

