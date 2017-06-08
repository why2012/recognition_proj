# coding: utf-8

choiceDirection = {"horizontal": 1, "vertical": 2}
questionType = {"choice": 2, "multiChoice": 3}

CHOICE = 1
JUDGE = 2
SUBJECT = 3
MULTI_CHOICE = 4

BLOCK_CHOICE = 5
BLOCK_MULTI_CHOICE = 6

MAX_CHOICE_NUM = 7
SCORE_BAR = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 0.5]
SCORE_BAR_AREA0_INDEX = 0
SCORE_BAR_AREA1 = range(1, 11)
SCORE_BAR_AREA2 = range(11, 16)
SCORE_BAR_AREA3_INDEX = 16
SCORE_BAR_RATIO = 0.0304

# 左侧留白，选择区域宽度，高度
MARK_CHOICE_RATIO = {"a3": [0.12311804, 0.05127883, 0.02863094]} # 0.09084939