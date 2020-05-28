import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

from random import randint
'''
优化点：
1.类的初始化
'''
# 如何构建队伍类，
class Team():


    def __init__(self,name):
        self.name = name

    #计算mu参数:得分能力
    def para_est(self,alpha,beta):
        home_d = np.mean(self.home_his_march.mean())
        away_d = np.mean(self.away_his_march.mean())
        self.home_mu = (beta-1)/alpha*home_d
        self.away_mu = (beta-1)/alpha*away_d

    #计算其他模式的参数
    def para_other(self):
        
        _data = np.cumsum(self.home_his_march).iloc[-1]
        _data.append(np.cumsum(self.away_his_march).iloc[-1])
        mean_other = np.mean(_data)
        #bino参数
        para_bino = [mean_other/(12 * 4 * 60 // 5),12 * 4 * 60 // 5]
        var_other = np.var(_data)
        #gaus参数
        para_gaus = [mean_other, var_other]
        #pois参数
        para_pois = [mean_other]
        self.para_bino = para_bino
        self.para_gaus = para_gaus
        self.para_pois = para_pois
        


    #格式化数据
    def fresh_data(self,df):
        if self.name in df['HomeTeam'].iloc[0]:#主场作战
            try:
                self.home_his_march
            except:
                self.home_his_march = pd.DataFrame()
            #需要对名字进行字符串的添加和匹配+i，in
            _insert_name = df['AwayTeam'].iloc[0]+str(len(self.home_his_march.columns)+1)
            _insert_value = df['HomeMinScore'].values

            self.home_his_march.insert(len(self.home_his_march.columns),_insert_name,_insert_value)#主队交战记录
        elif self.name in df['AwayTeam'].iloc[0]:#客场作战
            try:
                self.away_his_march
            except:
                self.away_his_march = pd.DataFrame()

            _insert_name = df['HomeTeam'].iloc[0]+str(len(self.away_his_march.columns)+1)
            _insert_value = df['AwayMinScore'].values

            self.away_his_march.insert(len(self.away_his_march.columns),_insert_name,_insert_value)#主队交战记录
        else:#异常数据
            raise NameError('没有找到比赛数据')

#实例和比塞类别
class Match():


    def __init__(self,df):
        #后面用于计算不同队伍的参数值
        #df = data_process(df)
        self.df = df
        self.home_team = df['HomeTeam'].iloc[0]
        self.away_team = df['AwayTeam'].iloc[0]
        self.home_score = df['HomeMinScore']
        self.away_score = df['AwayMinScore']
        self.winteam = df.iloc[0]['WinningTeam']
        self.home_total_score = df.iloc[-1]['HomeScore']
        self.away_total_score = df.iloc[-1]['AwayScore']
        self.count_lam()
        #
    @staticmethod
    def data_process(df):
        ls1 = [x for x in range(720,-1,-1) if x%60 ==0]
        data_index = ['GameType','WinningTeam','Quarter','SecLeft','HomeTeam','HomeScore','AwayTeam','AwayScore']
        #先不考虑加时赛
        df_rdgame = df[df['Quarter']<=4][data_index]
        df_rdgame['Istrue'] = ''
        for group in df_rdgame['Quarter'].unique():
            df_group = df_rdgame[df_rdgame['Quarter']==group]
            for _ in range(len(ls1)):
                if _ != 0:
                    _serices = df_group['SecLeft'][(df_group["SecLeft"]>=ls1[_])&(df_group['SecLeft']<ls1[_-1])]
                    if _serices.empty:
                        _serices = df_group['SecLeft'][df['SecLeft']>=ls1[_]]
                        df_rdgame['Istrue'][_serices.idxmin()]=True
                    else:
                        
                        df_rdgame['Istrue'][_serices.idxmin()]=True
#一场比赛的数据整理之后如下
        df_mindata = df_rdgame[df_rdgame['Istrue']==True]
        #分钟数据不足的情况怎么办
        df_mindata = min_process(df_mindata)
        if len(df_mindata) != 48:
            raise NameError('Minutes data not equal to 48')
        df_mindata['HomeMinScore'] = df_mindata['HomeScore']-df_mindata['HomeScore'].shift(1)
        df_mindata['HomeMinScore'].fillna(df_mindata['HomeScore'],inplace=True)
        df_mindata['AwayMinScore'] = df_mindata['AwayScore']-df_mindata['AwayScore'].shift(1)
        df_mindata['AwayMinScore'].fillna(df_mindata['AwayScore'],inplace=True)
        return df_mindata

    def count_lam(self):
        # 估计lam,mean_home,
        df_mindata = self.df
        home_mean = df_mindata['HomeMinScore'].mean()
        home_var = df_mindata['HomeMinScore'].std(ddof=0)
        away_mean =  df_mindata['AwayMinScore'].mean()
        away_var = df_mindata['AwayMinScore'].std(ddof=0)
        #lam
        self.lam = 0.5*(home_mean/home_var+away_mean/away_var)


#处理分钟数据：4节12分钟，如果有数据缺失，就用上一分钟去补
def min_process(df):
    ls = [x for x in range(720,-1,-1) if x%60 ==0]
    for i in df['Quarter'].unique():
        _data = df[df['Quarter']==i]
        for j in range(1,len(ls)):
            _df = _data[(_data['SecLeft']>=ls[j])&(_data['SecLeft']<ls[j-1])]
            if _df.empty:
                _index = _data[_data['SecLeft']>=ls[j-1]]['SecLeft'].idxmin()
                df.loc[_index+1] = np.nan
                df.loc[_index+1,'SecLeft'] = ls[j]
                df.sort_index(inplace = True)
                df.fillna(method='ffill', axis = 0,inplace = True)
    return df

#随机挑选出一场测试集
def rand_match(df):
    game_num = df['URL'].value_counts()
    #随机挑选出一场比赛
    rd_index = randint(int(len(game_num)*0.1)+1,len(game_num))
    df_rd_game = df[df['URL'].isin([game_num.index[rd_index]])]
    return df_rd_game



def main():
    lams = []
    Teams = {}
    df = pd.read_csv('/Users/frank/Documents/篮球Gamma模型/数据/NBA-PBP_2016-2017.csv')
    game_num = df[df['GameType']=='regular']['URL'].value_counts()
    for _ in range(0,int(len(game_num)*0.5)):
        print(f'第{_}场比赛'*5)
        game_k = df[df['URL'].isin([game_num.index[_]])]
        df_game = Match.data_process(game_k)
        match = Match(df_game)
        lams.append(match.lam)
        if match.home_team not in Teams.keys():
        #没有队伍类的时候，新建一个队伍实例
            key = match.home_team
            print(f'主队是{key}')
            value = Team(key)
            value.fresh_data(df_game)
            Teams[key]=value
        else:#有队伍类的时候，对队伍实例进行数据更新
            key = match.home_team
            print(f'主队是{key},更新数据')
            Teams[key].fresh_data(df_game)


        if match.away_team not in Teams.keys():
            key = match.away_team
            print(f'客队是{key}')
            value = Team(key)
            value.fresh_data(df_game)
            Teams[key]=value
        else:
            key = match.away_team
            print(f'客队是{key},更新数据')
            Teams[key].fresh_data(df_game)

    print(lams)
    print(Teams)

    lam_mean = np.mean(lams)
    lam_var = np.var(lams)
    alpha = lam_mean/lam_var
    beta = (lam_mean**2)/lam_var
    
    
    a_test = rand_match(df)
    
    game_test = Match.data_process(df)
    match_test = Match(df_game)
    
    home_team = Teams[match_test.home_team]
    away_team = Teams[match_test.away_team]
    
    home_team.para_est(alpha,beta)
    away_team.para_est(alpha,beta)
    lam_bayes = beta/alpha
    
    #实际上lam = ((home_team.home_mu+away_team.away_mu)+beta)/((awaw_score+home_score)+alpha)
    
if __name__ == '__main__':
    main()
