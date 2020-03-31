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