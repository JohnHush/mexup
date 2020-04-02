from apps.quantization.soccer_poisson import cal_soccer_odds
from apps.quantization.constans import market_type, period, match_states


import numpy as np
#输入参数
#mu_home 主队期望进球数; mu_away 客队期望进球数; home_score 主队当前进球数; away_score 客队当前进球数;
#adj_mode 赔率调整模式，0-平局调整模式，1-rho调整模式; adj_parameter, 平局调整模式[draw_adj, draw_split]，rho模式

# mu: [supremacy, total goals]
# score=[half_time_score,full_time_score,score_0_15, score_15_30, score_30_45, score_45_60, score_60_75, score_75_90]
# decay 衰减系数
# parameter=[adj_mode, rho]或者[adj_mode,[draw_adj, draw_split]]
# clock=[stage, running_time, ht_add, ft_ad, et_ht_add, et_ft_add]

class cal_match_odds(object):
    def __init__(self):
        self.odds_tool_full_time = cal_soccer_odds()
        self.odds_tool_1st_half = cal_soccer_odds()
        self.odds_tool_2nd_half = cal_soccer_odds()
        # self.set_value(mu, score, clock, decay, parameter)

    def set_value(self, mu, score, clock, decay, parameter):

        self.mu = mu
        self.clock = clock
        self.parameter = parameter

        self.half_time_score = score[0]
        self.full_time_score = score[1]
        self.second_half_socore=[self.full_time_score[i]-self.half_time_score[i] for i in [0, 1]]

        self.stage = clock[0]
        self.running_time = clock[1]
        self.ht_add = clock[2]
        self.ft_add = clock[3]

        self.decay = decay
        self.parameter = parameter

        # self.stage 4-赛前，6-上半场 ，7-中场 ，8-下半场，13-常规时间结束, 0-结束(包含加时和点球)

        self.time_remain_2nd_half = (45*60 + self.ft_add) / (90*60 + self.ft_add)

        if self.stage in [4, 6]:
            self.time_remain_now = (90*60 + self.ft_add + self.ht_add - self.running_time) / (
                        90*60 + self.ft_add + self.ht_add)

            self.mu_full_time_now = [self.mu[i] * (self.time_remain_now ** self.decay) for i in [0, 1]]
            self.mu_second_half = [self.mu[i] * (self.time_remain_2nd_half ** self.decay) for i in [0, 1]]
            self.mu_first_half_now = [self.mu_full_time_now[i] - self.mu_second_half[i] for i in [0, 1]]
            self.mu_second_half_now = self.mu_second_half

        elif self.stage in [7, 8]:
            self.time_remain_now = (90*60 + self.ft_add - self.running_time) / (90*60 + self.ft_add)
            self.mu_1st_half_now = [0, 0]
            self.mu_full_time_now = [self.mu[i] * (self.time_remain_now ** self.decay) for i in [0, 1]]
            self.mu_second_half_now = self.mu_full_time_now

        # elif self.stage == 13:
        #     self.mu_1st_half_now = [0,0]
        #     self.mu_full_time_now = [0,0]
        #     self.mu_2nd_half_now = [0,0]


    def odds_output(self):

        odds = {}
        if self.stage in [4, 6]:
            self.odds_tool_full_time.set_value(self.mu_full_time_now, self.full_time_score, self.parameter)
            self.odds_tool_1st_half.set_value(self.mu_first_half_now, self.half_time_score, self.parameter)
            self.odds_tool_2nd_half.set_value(self.mu_second_half_now, self.second_half_socore, self.parameter)

            odds[period.SOCCER_FULL_TIME] = self.cal_full_time_odds()
            odds[period.SOCCER_FIRST_HALF] = self.cal_first_half_odds()
        elif self.stage in [7, 8]:
            self.odds_tool_full_time.set_value(self.mu_full_time_now, self.full_time_score, self.parameter)
            self.odds_tool_2nd_half.set_value(self.mu_second_half_now, self.second_half_socore, self.parameter)
            odds[period.SOCCER_FULL_TIME] = self.cal_full_time_odds()
        return odds
    #计算全场玩法的赔率
    def cal_full_time_odds(self):

        full_time_odds={}

        #输出SOCCER_3WAY玩法
        full_time_odds[market_type.SOCCER_3WAY]=self.odds_tool_full_time.had()

        #输出亚盘让球玩法、 SOCCER_ASIAN_HANDICAP
        asian_handicap={}
        ahc_line_list = np.arange(-15, 12.25, 0.25)
        for i in ahc_line_list:
            asian_handicap[str(i)]=self.odds_tool_full_time.asian_handicap(i)
        full_time_odds[market_type.SOCCER_ASIAN_HANDICAP]=asian_handicap

        #输出亚盘大小玩法 SOCCER_ASIAN_TOTALS
        over_under = {}
        hilo_line_list = np.arange(0.5, 20.25, 0.25)
        for j in hilo_line_list:
            over_under[str(j)] = self.odds_tool_full_time.over_under(j)
        full_time_odds[market_type.SOCCER_ASIAN_TOTALS]=over_under

        #输出正确比分玩法赔率 SOCCER_CORRECT_SCORE
        correct_score={}
        for i in range(0,6):
            for j in range(0,6):
                correct_score[ str(i)+'_'+str(j)] = self.odds_tool_full_time.correct_score(i,j)
        full_time_odds[market_type.SOCCER_CORRECT_SCORE]=correct_score

        # #输出双重机会大小玩法赔率
        # double_chance_over_under={}
        # for k in [1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]:
        #     double_chance_over_under[str(k)] = self.odds_tool_full_time.double_chance_over_under(k)
        # first_half_odds['double_chance_over_under']=double_chance_over_under

        #输出主队亚盘大小 SOCCER_TOTAL_HOME_TEAM
        home_over_under={}
        home_ou_line_list = np.arange(0.5, 15.25, 0.25)
        for j in home_ou_line_list:
            home_over_under[str(j)] = self.odds_tool_full_time.home_over_under(j)
        full_time_odds[market_type.SOCCER_GOALS_HOME_TEAM]=home_over_under

        #输出客队亚盘大小 SOCCER_TOTAL_AWAY_TEAM
        away_over_under={}
        away_ou_line_list = np.arange(0.5, 15.25, 0.25)
        for j in away_ou_line_list:
            away_over_under[str(j)] = self.odds_tool_full_time.away_over_under(j)
        full_time_odds[market_type.SOCCER_GOALS_AWAY_TEAM]=away_over_under

        # #输出主队净胜
        # home_winning_by={}
        # for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
        #     home_winning_by[str(i)] = self.odds_tool_full_time.home_winning_by(i)
        # full_time_odds['home_winning_by']=home_winning_by
        #
        # # 输出客队净胜
        # away_winning_by = {}
        # for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        #     away_winning_by[str(i)] = self.odds_tool_full_time.away_winning_by(i)
        # full_time_odds['away_winning_by']=away_winning_by

        #输出是否都进球玩法赔率 SOCCER_BOTH_TEAMS_TO_SCORE
        full_time_odds[market_type.SOCCER_BOTH_TEAMS_TO_SCORE] = self.odds_tool_full_time.both_scored()

        #输出奇偶玩法赔率 SOCCER_ODD_EVEN_GOALS
        full_time_odds[market_type.SOCCER_ODD_EVEN_GOALS] = self.odds_tool_full_time.odd_even()

        return full_time_odds

    #输出上半场玩法赔率
    def cal_first_half_odds(self):

        first_half_odds = {}

        #输出SOCCER_3WAY玩法
        first_half_odds[market_type.SOCCER_3WAY] = self.odds_tool_1st_half.had()

        #输出亚盘让球玩法、 SOCCER_ASIAN_HANDICAP
        asian_handicap={}
        ahc_line_list = np.arange(-10, 10.25, 0.25)
        for i in ahc_line_list:
            asian_handicap[str(i)]=self.odds_tool_1st_half.asian_handicap(i)
        first_half_odds[market_type.SOCCER_ASIAN_HANDICAP]=asian_handicap

        #输出亚盘大小玩法 SOCCER_ASIAN_TOTALS
        over_under = {}
        hilo_line_list = np.arange(0.5, 15.25, 0.25)
        for j in hilo_line_list:
            over_under[str(j)] = self.odds_tool_1st_half.over_under(j)
        first_half_odds[market_type.SOCCER_ASIAN_TOTALS]=over_under

        #输出正确比分玩法赔率 SOCCER_CORRECT_SCORE
        correct_score={}
        for i in range(0,6):
            for j in range(0,6):
                correct_score[ str(i)+'_'+str(j)] = self.odds_tool_1st_half.correct_score(i,j)
        first_half_odds[market_type.SOCCER_CORRECT_SCORE]=correct_score

        # #输出双重机会大小玩法赔率
        # double_chance_over_under={}
        # for k in [1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]:
        #     double_chance_over_under[str(k)] = self.odds_tool_full_time.double_chance_over_under(k)
        # first_half_odds['double_chance_over_under']=double_chance_over_under

        #输出主队亚盘大小 SOCCER_TOTAL_HOME_TEAM
        home_over_under={}
        home_ou_line_list = np.arange(0.5, 10.25, 0.25)
        for j in home_ou_line_list:
            home_over_under[str(j)] = self.odds_tool_1st_half.home_over_under(j)
        first_half_odds[market_type.SOCCER_GOALS_HOME_TEAM]=home_over_under

        #输出客队亚盘大小 SOCCER_TOTAL_AWAY_TEAM
        away_over_under={}
        away_ou_line_list = np.arange(0.5, 10.25, 0.25)
        for j in away_ou_line_list:
            away_over_under[str(j)] = self.odds_tool_1st_half.away_over_under(j)
        first_half_odds[market_type.SOCCER_GOALS_AWAY_TEAM]=away_over_under

        # #输出主队净胜
        # home_winning_by={}
        # for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
        #     home_winning_by[str(i)] = self.odds_tool_1st_half.home_winning_by(i)
        # first_half_odds['home_winning_by']=home_winning_by
        #
        # # 输出客队净胜
        # away_winning_by = {}
        # for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        #     away_winning_by[str(i)] = self.odds_tool_1st_half.away_winning_by(i)
        # first_half_odds['away_winning_by']=away_winning_by

        #输出是否都进球玩法赔率 SOCCER_BOTH_TEAMS_TO_SCORE
        first_half_odds[market_type.SOCCER_BOTH_TEAMS_TO_SCORE] = self.odds_tool_1st_half.both_scored()

        #输出奇偶玩法赔率 SOCCER_ODD_EVEN_GOALS
        first_half_odds[market_type.SOCCER_ODD_EVEN_GOALS]=self.odds_tool_1st_half.odd_even()

        return first_half_odds

# match=cal_match_odds()
# match.set_value([0.5,2.7],[[0,0],[0,0]],[8,45*60,1*60,3*60],0.88,[1,-0.08])
#
# print(match.odds_output())

