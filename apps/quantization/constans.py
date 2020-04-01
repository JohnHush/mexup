#市场类型
class market_type():
    # 足球亚盘让球
    SOCCER_ASIAN_HANDICAP=1001
    # 足球让球
    SOCCER_EUROPEAN_HANDICAP=1002
    # 欧盘大小球
    SOCCER_TOTALS=1003
    # 足球波胆=小于6球)
    SOCCER_CORRECT_SCORE=1004
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

#选型类型
class selection_type():
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

class period():
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