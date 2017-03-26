# coding: utf-8
from ScantronRecogController import *
import numpy as np
import conf.Config as conf

class ScoreMarkingController(ScantronRecogController):
	def execute(self):
		ScoreMarkingController.checkParams02(self)
		ScantronRecogController.execute(self)
		self.marking(self.getResult()["ans"], self.standAns, self.quesType, self.direction, self.totalScore)

	# standAns:A,B|C|D|A, direction:horizontal: 1|vertical: 2, type: choice: 2|multiChoice: 3, 
	# return [0, 1, 2, 3], "A,B|C|D|-"
	def marking(self, mat, standAns, quesType, direction, totalScore):
		ansMat = np.array(mat)
		standAns = standAns.lower().split("|")
		totalQuesCount = len(standAns)
		singleAnsScore = totalScore / totalQuesCount
		# 每题分数
		stuScore = []
		# 每题选项
		stuSelection = []
		for ansIndex, answerSingles in enumerate(standAns):
			stuSelection.append(["-"] * len(standAns))
			answerSingles = answerSingles.split(",")
			# 正确选项个数
			ansCount = len(answerSingles)
			# 正确选择个数
			stuAnsTotal = 0
			# 学生答案
			if direction == conf.choiceDirection["horizontal"]:
				cardAns =  ansMat[ansIndex]
			elif direction == conf.choiceDirection["vertical"]:
				cardAns =  ansMat[:, ansIndex]
			# 对应answerSingles的序号
			choiceIndex = 0
			for cardAnsIndex, cardAnsNumber in enumerate(cardAns):
				if choiceIndex < len(answerSingles):
					ansNumber = answerSingles[choiceIndex];
				else:
					ansNumber = -1
				if cardAnsNumber != 0:
					stuSelection[ansIndex][cardAnsIndex] = chr(cardAnsIndex + ord("a")).upper()
				# 将字母选项映射为数字序号
				if ansNumber != -1:
					ansNumber = ord(ansNumber) - ord("a")
					# 匹配answerSingles与cardAns的序号
					if ansNumber != cardAnsIndex:
						# 匹配失败, 此选项不属于此填涂区域
						ansNumber = -1
					else:
						# 匹配成功, 此选项属于此填涂区域
						choiceIndex += 1
				# 错选, 0分
				if ansNumber == -1 and cardAnsNumber == 1:
					stuAnsTotal = 0
					break
				# 没有选此选项，继续判定
				if ansNumber == -1:
					continue
				# 选正确  
				if cardAnsNumber == 1:
					stuAnsTotal += 1
			# 全对, 部分对
			if stuAnsTotal != 0:
				if stuAnsTotal == ansCount:
					stuScore.append(singleAnsScore)
				elif stuAnsTotal > 0 and stuAnsTotal < ansCount:
					stuScore.append(singleAnsScore * 0.5)
			else:
				stuScore.append(0.0)
			stuSelection[ansIndex] = ",".join(stuSelection[ansIndex])
		self.setResult((stuScore, "|".join(stuSelection)), STATUS_OK);

	@staticmethod
	def checkParams02(self):
		standAns = self.getArg("standAns", None)
		quesType = self.getIntArg("quesType", None)
		direction = self.getIntArg("direction", None)
		totalScore = self.getFloatArg("totalScore", None)
		if standAns is None:
			raise ErrorStatusException("standAns must be string indicate standard answer", STATUS_PARAM_ERROR)
		if quesType is None:
			raise ErrorStatusException("quesType must be " + str(conf.questionType["choice"]) + ":choice|" + str(conf.questionType["multiChoice"]) + ":multichoice", STATUS_PARAM_ERROR)
		if quesType != conf.questionType["choice"] and quesType != conf.questionType["multiChoice"]:
			raise ErrorStatusException("quesType must be " + str(conf.questionType["choice"]) + ":choice|" + str(conf.questionType["multiChoice"]) + ":multichoice", STATUS_PARAM_ERROR)
		if direction is None:
			raise ErrorStatusException("direction must be " + str(conf.choiceDirection["horizontal"]) + ":horizontal|" + str(conf.choiceDirection["vertical"]) + ":vertical", STATUS_PARAM_ERROR)
		if direction != conf.choiceDirection["horizontal"] and direction != conf.choiceDirection["vertical"]:
			raise ErrorStatusException("direction must be " + str(conf.choiceDirection["horizontal"]) + ":horizontal|" + str(conf.choiceDirection["vertical"]) + ":vertical", STATUS_PARAM_ERROR)
		if totalScore is None:
			raise ErrorStatusException("totalScore must be number", STATUS_PARAM_ERROR)
		self.standAns = standAns
		self.quesType = quesType
		self.direction = direction
		self.totalScore = totalScore

