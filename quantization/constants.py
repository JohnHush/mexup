#配置类

class config(  ):
    #反查精度
    infer_eps = 0.001


#市场类型
class market_type(  ):
    # 足球亚盘让球
    SOCCER_ASIAN_HANDICAP=1001
    # 足球让球
    SOCCER_EUROPEAN_HANDICAP=1002
    # 欧盘大小球
    SOCCER_TOTALS=1003
    # 足球波胆=小于6球)
    SOCCER_CORRECT_SCORE=1029
    # 足球独赢
    SOCCER_3WAY=1005
    # 足球平局退款
    SOCCER_DRAW_NO_BET=1006
    # 足球亚盘大小球
    SOCCER_ASIAN_TOTALS=1007
    # 足球单双
    SOCCER_ODD_EVEN_GOALS=1008
    # 足球角球胜平负
    SOCCER_CORNERS_MATCH_BET=1009
    # 足球角球大小球
    SOCCER_TOTAL_CORNERS=1010
    # 足球角球让球
    SOCCER_CORNERS_HANDICAP=1011
    # 足球双重机会
    SOCCER_DOUBLE_CHANCE=1012
    # 足球第一个进球球队
    SOCCER_FIRST_TEAM_TO_SCORE=1013
    # 足球角球数=精确)
    SOCCER_CORNER_COUNT_EXACT=1014
    # 足球角球数单双
    SOCCER_CORNER_ODD_EVEN=1015
    # 足球总进球数（范围）
    SOCCER_TOTAL_GOALS_AGGREGATED=1020
    # 足球主队大小球
    SOCCER_TOTAL_HOME_TEAM=1021
    # 足球客队大小球
    SOCCER_TOTAL_AWAY_TEAM=1022
    # 足球主队进球数
    SOCCER_GOALS_HOME_TEAM=1023
    # 足球客队进球数
    SOCCER_GOALS_AWAY_TEAM=1024
    # 足球双方均有进球
    SOCCER_BOTH_TEAMS_TO_SCORE=1027
    # 足球总进球数（精確）
    SOCCER_TOTAL_GOALS_EXACTLY=1081
    # 足球主队单双
    SOCCER_HOME_ODD_EVEN=1082
    # 足球客队单双
    SOCCER_AWAY_ODD_EVEN=1083

    SOCCER_HALFTIME_FULLTIME=1033
    SOCCER_BOTH_HALVES_OVER1_5=1034
    SOCCER_BOTH_HALVES_UNDER1_5=1035
    SOCCER_HOME_TO_SCORE_IN_BOTH_HALVES=1036
    SOCCER_AWAY_TO_SCORE_IN_BOTH_HALVES=1037
    SOCCER_HOME_TO_WIN_BOTH_HALVES=1038
    SOCCER_HOME_TO_WIN_EITHER_HALVE=1039
    SOCCER_AWAY_TO_WIN_BOTH_HALVES=1040
    SOCCER_AWAY_TO_WIN_EITHER_HALVE=1041


    # 篮球胜平负（常规时间）
    BASKETBALL_3WAY = 3001 # "Basketball3Way", Arrays.asList(SelectionType.HOME, SelectionType.DRAW, SelectionType.AWAY), "3Way"
    # 篮球让分
    BASKETBALL_HANDICAP = 3002 # "BasketballHandicap", Arrays.asList(SelectionType.HOME, SelectionType.AWAY), "Handicap"),
    # 篮球总得分
    BASKETBALL_TOTALS = 3003 # "BasketballTotals", Arrays.asList(SelectionType.OVER, SelectionType.UNDER), "Totals"),
    # 篮球胜负（包括加时）
    BASKETBALL_2WAY = 3004 # "Basketball2Way", Arrays.asList(SelectionType.HOME, SelectionType.AWAY), "2Way"),
    # 篮球得分单双
    BASKETBALL_ODD_EVEN = 3005 # "BasketballOddEven", Arrays.asList(SelectionType.ODD, SelectionType.EVEN), "OddEven"),
    # 篮球欧盘让分
    BASKETBALL_EUROPEAN_HANDICAP =3006 # "BasketballEuropeanHandicap", Arrays.asList = SelectionType.HOME, SelectionType.AWAY), "EuropeanHandicap"),
    #篮球独赢分数范围（包括加时）
    BASKETBALL_WINNING_MARGINS = 3007 # "BasketballWinningMargins", null, "WinningMargins"),
    #篮球第一个得到X分的球队
    BASKETBALL_1ST_TO_X_POINTS = 3008 # "Basketball1stToXPoints", Collections.singletonList = SelectionType.YES), "1stToXPoints"),
    BASKETBALL_TOTAL_AAMS = 3009 # "BasketballTotalAAMS", null, "TotalAAMS"),
    #篮球NBA总分范围（不包括加时）
    BASKETBALL_TOTAL_MARGINS_NBA_EXC_LOT = 3010 # "BasketballTotalMarginsNBAExclOT", Collections.singletonList = SelectionType.YES), "TotalMarginsNBAExclOT"),
    #篮球非NBA总分范围（不包括加时）
    BASKETBALL_TOTAL_MARGINS_NONE_NBA_EXC_LOT = 3011 # "BasketballTotalMarginsNonNBAExclOT", Collections.singletonList = SelectionType.YES), "TotalMarginsNonNBAExclOT"),
    #篮球主队总分大小
    BASKETBALL_TOTALS_HOME_TEAM = 3012 # "BasketballTotalsHomeTeam", Arrays.asList = SelectionType.OVER, SelectionType.UNDER), "TotalsHomeTeam"),
    #篮球客队总分大小
    BASKETBALL_TOTALS_AWAY_TEAM = 3013 # "BasketballTotalsHomeTeam", Arrays.asList = SelectionType.OVER, SelectionType.UNDER), "TotalsHomeTeam"),
    #篮球独赢和大小（包括加时）
    BASKETBALL_MATCH_BET_AND_TOTALS_INC_LOT = 3014 # "BasketballMatchbetAndTotalsInclOT", null, "MatchbetAndTotalsInclOT"),
    #篮球会有加时
    BASKETBALL_WILL_THERE_BE_AN_OVERTIME = 3015 # "BasketballWillThereBeAnOvertime", Collections.singletonList = SelectionType.YES), "WillThereBeAnOvertime"),
    #篮球分数最高的一节（不包括加时）
    BASKETBALL_HIGHEST_SCORING_QUARTER = 3016 # "BasketballHighestScoringQuarter", Collections.singletonList = SelectionType.YES), "HighestScoringQuarter"),
    #篮球上下半场胜平负
    BASKETBALL_HAL_TIME_FULL_TIME = 3017 # "BasketballHaltimeFulltime", Collections.singletonList = SelectionType.YES), "HaltimeFulltime"),
    BASKETBALL_US_TOTAL = 3018 # "BasketballUSTotal", null, "USTotal"),
    BASKETBALL_US_SPREAD = 3019 # "BasketballUSSpread", null, "USSpread"),
    #篮球平局退款
    BASKETBALL_DRAW_NO_BET = 3020 # "BasketballDrawNoBet", Arrays.asList = SelectionType.HOME, SelectionType.AWAY), "DrawNoBet"),
    BASKETBALL_X_TH_QUARTER_COMPETITOR1_TOTAL = 3021 # "BasketballXthQuarterCompetitor1Total", Arrays.asList = SelectionType.OVER, SelectionType.UNDER), "XthQuarterCompetitor1Total"),
    #主队总得分单双
    BASKETBALL_TEAM1_TOTAL_ODD_EVEN = 3022 # "BasketballTeam1TotalOddEven", Arrays.asList = SelectionType.OVER, SelectionType.UNDER), "Team1TotalOddEven"),
    #客队总得分单双
    BASKETBALL_TEAM2_TOTAL_ODD_EVEN = 3023 # "BasketballTeam2TotalOddEven", Arrays.asList = SelectionType.OVER, SelectionType.UNDER), "Team2TotalOddEven"),
    #主队总得分尾数
    BASKETBALL_TEAM1_TOTAL_LAST_DIGIT = 3024 # "BasketballTeam1TotalLastDigit", Collections.singletonList = SelectionType.YES), "Team1TotalLastDigit"),
    #篮球客队总得分尾数
    BASKETBALL_TEAM2_TOTAL_LAST_DIGIT = 3025 # "BasketballTeam2TotalLastDigit", Collections.singletonList = SelectionType.YES), "Team2TotalLastDigit"),
    #篮球全场大小（包含精确）
    BASKETBALL_TOTALS_EXACT = 3026 # "BasketBallTotalsExact", Arrays.asList = SelectionType.OVER, SelectionType.UNDER, SelectionType.EXACT), "TotalsExact"),
    #篮球全场大小（包含加時）
    BASKETBALL_TOTALS_INCL_OVERTIME = 3027 # "BasketBallTotalsInclOvertime", Arrays.asList = SelectionType.OVER, SelectionType.UNDER), "TotalsInclOvertime"),
    #篮球主队总分大小
    BASKETBALLR_HOEM_TOTAL = 3028 # "BasketballrHoemTotal", Arrays.asList = SelectionType.OVER, SelectionType.UNDER), "HoemTotal"),
    #篮球客队总分大小
    BASKETBALL_AWAY_TOTAL = 3029 # "BasketballAwayTotal", Arrays.asList = SelectionType.OVER, SelectionType.UNDER), "AwayTotal"),
    #籃球独赢（不包含加時）
    BASKETBALL_HALF_WINNER = 3030 # "BasketballHalfWinner", Arrays.asList = SelectionType.HOME, SelectionType.AWAY), "HalfWinner"),
    #篮球单双（包含加時）
    BASKETBALL_ODD_EVEN_INCL_OVERTIME = 3032 # "BasketballOddEvenInclOvertime", Arrays.asList = SelectionType.ODD, SelectionType.EVEN), "OddEvenInclOvertime")


#选型类型
class selection_type(  ):
    #主队
    HOME=1
    #客队
    AWAY=2
    #平局
    DRAW=3
    #大于
    OVER=4
    #小于
    UNDER=5
    #双
    EVEN=6
    #单
    ODD=7
    #是
    YES=8
    #否
    NO=9
    #胜
    W1=10
    #负
    W2=11
    HOME_OR_DRAW=12
    HOME_OR_AWAY=13
    AWAY_OR_DRAW=14

    #精确
    EXACT=15
    #主胜
    HOME_WIN=16
    #客胜
    AWAY_WIN=17

    # composite type
    HOME_AND_OVER=18
    HOME_AND_UNDER=19
    AWAY_AND_OVER=20
    AWAY_AND_UNDER=21
    DRAW_AND_OVER=22
    DRAW_AND_UNDER=23

    HOME_AND_YES=24
    HOME_AND_NO=25
    AWAY_AND_YES=26
    AWAY_AND_NO=27
    DRAW_AND_YES=28
    DRAW_AND_NO=29

    # half time / full time composite type
    DRAW_AND_HOME=30
    DRAW_AND_AWAY=31
    DRAW_AND_DRAW=32
    HOME_AND_HOME=33
    HOME_AND_AWAY=34
    HOME_AND_DRAW=35
    AWAY_AND_HOME=36
    AWAY_AND_AWAY=37
    AWAY_AND_DRAW=38


class period(  ):
    # 足球全场
    SOCCER_FULL_TIME = 1001
    # 足球上半场
    SOCCER_FIRST_HALF = 1002
    # 足球下半场
    SOCCER_SECOND_HALF = 1003
    # 足球第一个加时
    SOCCER_EXTRA_FIRST_HALF = 1004
    # 足球第二个加时
    SOCCER_EXTRA_SECOND_HALF = 1005
    # 足球点球/罚球
    SOCCER_PENALTY_KICK = 1006
    # 足球0-15min
    SOCCER_0_15_MIN = 1007
    # 足球15-30min
    SOCCER_15_30_MIN = 1008
    # 足球30-45min
    SOCCER_30_45_MIN = 1009
    # 足球45-60min
    SOCCER_45_60_MIN = 1010
    # 足球60-75min
    SOCCER_60_75_MIN = 1011
    # 足球75-90min
    SOCCER_75_90_MIN = 1012
    # 足球加时
    SOCCER_EXTRA = 1013

    # 篮球全场（包含加时）
    BASKETBALL_FULL_TIME =3001 # "BasketballFullTime", "Full Time"),
    # 常规时间
    BASKETBALL_REGULAR_TIME = 3002 # "BasketBallRegularTime", "Regular Time"),
    # 蓝球上半场
    BASKETBALL_FIRST_HALF = 3003 # "BasketballFirstHalf", "First Half"),
    #篮球下半场
    BASKETBALL_SECOND_HALF = 3004 # "BasketballSecondHalf", "Second Half"),
    #篮球第一节
    BASKETBALL_FIRST_QUARTER = 3005 # "BasketballFirstQuarter", "First Quarter"),
    #篮球第二节
    BASKETBALL_SECOND_QUARTER = 3006 # "BasketballSecondQuarter", "Second Quarter"),
    #篮球第三节
    BASKETBALL_THIRD_QUARTER = 3007 # "BasketballThirdQuarter", "Third Quarter"),
    #篮球第四节
    BASKETBALL_FOURTH_QUARTER = 3008 # "BasketballFourthQuarter", "Fourth Quarter"),
    #篮球第一个加时
    BASKETBALL_FIRST_OVERTIME = 3009 # "BasketballFirstOvertime", "First Overtime"),
    #篮球第二个加时
    BASKETBALL_SECOND_OVERTIME = 3010 # "BasketballSecondOvertime", "Second Overtime"),
    #篮球第三个加时
    BASKETBALL_THIRD_OVERTIME = 3012 # "BasketballThirdOvertime", "Third Overtime"),
    #篮球第四个加时
    BASKETBALL_FOURTH_OVERTIME = 3013 # "BasketballFourthOvertime", "Fourth Overtime"),
    #篮球第五个加时
    BASKETBALL_FIFTH_OVERTIME = 3014 # "BasketballFifthOvertime", "Fifth Overtime"),
    #篮球第六个加时
    BASKETBALL_SIXTH_OVERTIME = 3015 # "BasketballSixthOvertime", "Sixth Overtime"),
    #篮球第七个加时
    BASKETBALL_SEVENTH_OVERTIME = 3016 # "BasketballSeventhOvertime", "Seventh Overtime"),
    #篮球第八个加时
    BASKETBALL_EIGHTH_OVERTIME = 3017 # "BasketballEighthOvertime", "Eighth Overtime");

class match_states():
    # 已结束
    ENDED = 0
    # 推迟
    POSTPONE = 1
    # 中断
    INTERRUPTED = 2
    # 取消
    CANCELED =3
    # 即将开赛
    UPCOMING = 4
    # 滚球
    LIVE = 5
    # 上半场 - 全称
    SOCCER_FIRST_HALF = 6
    # 半场时间 - 全称
    SOCCER_HALF_TIME = 7
    # 下半场 - 全称
    SOCCER_SECOND_HALF = 8
    # ET(加时赛) - 上半场 - 全称
    SOCCER_ET_FIRST_HALF = 9
    # ET(加时赛) - 半场时间 - 全称
    SOCCER_ET_HALF_TIME = 10
    # ET(加时赛) - 后半场
    SOCCER_ET_SECOND_HALF = 11
    # 常规时间结束
    SOCCER_FULL_TIME = 13
    # ET(加时赛) - 结束
    SOCCER_ET_ENDED = 14
    # 点球开始
    SOCCER_PENALTY = 15
    # 点球结束
    SOCCER_PENALTY_ENDED = 16

