from apps.quantization.soccer_poisson import cal_soccer_odds
from apps.quantization.constans import market_type,period


import numpy as np
#输入参数
#mu_home 主队期望进球数; mu_away 客队期望进球数; home_score 主队当前进球数; away_score 客队当前进球数;
#adj_mode 赔率调整模式，0-平局调整模式，1-rho调整模式; adj_parameter, 平局调整模式[draw_adj, draw_split]，rho模式

# mu: [supremacy, total goals]
# score=[half_time_score,full_time_score,score_0_15, score_15_30, score_30_45, score_45_60, score_60_75, score_75_90]
# decay=[decay_home, decay_away]
# parameter=[adj_mode, rho]或者[adj_mode,[draw_adj, draw_split]]
# clock=[stage, running_time, ht_add, ft_ad, et_ht_add, et_ft_add]

class cal_match_odds(object):
    def __init__(self, mu, score, clock, decay, parameter):
        self.odds_tool_full_time = cal_soccer_odds()
        self.odds_tool_1st_half=cal_soccer_odds()
        self.odds_tool_2nd_half = cal_soccer_odds()
        self.set_value(mu, score, clock, decay, parameter)

    def set_value(self, mu, score, clock, decay, parameter):

        self.mu = mu
        self.clock = clock
        self.parameter = parameter
        self.mu_home = (self.mu[0] + self.mu[1]) / 2
        self.mu_away = (self.mu[1] - self.mu[0]) / 2

        self.half_time_score = score[0]
        self.full_time_score = score[1]
        self.second_half_socore=[self.full_time_score[i]-self.half_time_score[i] for i in range(len(self.half_time_score))]

        self.stage = clock[0]
        self.running_time = clock[1]
        self.ht_add = clock[2]
        self.ft_add = clock[3]

        self.decay_home = decay[0]
        self.decay_away = decay[1]
        self.parameter = parameter

        # self.stage 0-赛前，1-上半场 ，2-中场 ，3-下半场，4-全场结束

        self.time_remain_2nd_half = (45 + self.ft_add) / (90 + self.ft_add)

        if self.stage in [0, 1]:
            self.time_remain_now = (90 + self.ft_add + self.ht_add - self.running_time) / (
                        90 + self.ft_add + self.ht_add)

            self.mu_home_now = self.mu_home * (self.time_remain_now ** self.decay_home)
            self.mu_away_now = self.mu_away * (self.time_remain_now ** self.decay_away)

            self.exp_home_goals_2nd_half = self.mu_home * (self.time_remain_2nd_half ** self.decay_home)
            self.exp_away_goals_2nd_half = self.mu_away * (self.time_remain_2nd_half ** self.decay_away)

            self.sup_2nd_half_now= self.exp_home_goals_2nd_half-self.exp_away_goals_2nd_half
            self.ttg_2nd_half_now = self.exp_home_goals_2nd_half + self.exp_away_goals_2nd_half
            self.mu_2nd_half_now=[self.sup_2nd_half_now,self.ttg_2nd_half_now]


            self.sup_full_time_now = self.mu_home_now - self.mu_away_now
            self.ttg_full_time_now = self.mu_home_now + self.mu_away_now
            self.mu_full_time_now = [self.sup_full_time_now, self.ttg_full_time_now]

            self.sup_1st_half_now = self.sup_full_time_now - (
                        self.exp_home_goals_2nd_half - self.exp_away_goals_2nd_half)
            self.ttg_1st_half_now = self.ttg_full_time_now - (
                        self.exp_home_goals_2nd_half + self.exp_away_goals_2nd_half)
            self.mu_1st_half_now = [self.sup_1st_half_now, self.ttg_1st_half_now]

        elif self.stage in [2, 3]:
            self.time_remain_now = (90 + self.ft_add - self.running_time) / (90 + self.ft_add)
            self.mu_1st_half_now = [0,0]


            self.mu_home_now = self.mu_home * (self.time_remain_now ** self.decay_home)
            self.mu_away_now = self.mu_away * (self.time_remain_now ** self.decay_away)

            self.sup_full_time_now = self.mu_home_now - self.mu_away_now
            self.ttg_full_time_now = self.mu_home_now + self.mu_away_now
            self.mu_full_time_now = [self.sup_full_time_now, self.ttg_full_time_now]

            self.mu_2nd_half_now =self.mu_full_time_now

        elif self.stage == 4:
            self.mu_1st_half_now = [0,0]
            self.mu_full_time_now = [0,0]
            self.mu_2nd_half_now = [0,0]


    def odds_output(self):
        self.odds_tool_full_time.set_value(self.mu_full_time_now, self.full_time_score, self.parameter)
        self.odds_tool_1st_half.set_value(self.mu_1st_half_now, self.half_time_score, self.parameter)
        self.odds_tool_2nd_half.set_value(self.mu_2nd_half_now, self.second_half_socore, self.parameter)

        odds={}
    #计算上半场玩法的赔率
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

        #输出半全场玩法赔率
        odds[period.SOCCER_FIRST_HALF]= full_time_odds

    #输出上半场玩法赔率
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

        odds[period.SOCCER_FIRST_HALF] = first_half_odds
        return odds

# match=cal_match_odds([0.5,2.7],[[0,0],[0,0]],[0,0,1,3],[0.88,0.88],[1,-0.08])
#
# print(match.odds_output())


