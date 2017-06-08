# coding: utf-8
import numpy as np
from conf.Config import *
from lib.ScantronAnalyzeCV import *
from lib.CustomMarking import *
from lib.SiftMatch import *
from lib.PreProcessing import *
from BaseController import *
import urllib2 as url

# quesType: [1, 2, 3, 4], 选择, 判断, 主观, 多选
# todo: BLOCK_CHOICE BLOCK_MULTI_CHOICE
class MultiTypeScoreMarkController(BaseController):
	def execute(self):
		MultiTypeScoreMarkController.checkParams(self)
		if not self.imgUrl:
			img = self.processUpFile("img")
		else:
			# 从其他地方获取图片
			res = url.urlopen(self.imgUrl)
			img = res.read()
		img = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)# IMREAD_GRAYSCALE
		# if self.quesType != SUBJECT: 
		# 	details = {}
		# 	details["area"] = None
		# 	if self.quesType == JUDGE:
		# 		details["questionCount"] = self.col
		# 	else:
		# 		details["questionCount"] = self.MAX_CHOICE_NUM
		# 	details["answerCount"] = self.row
		# 	details["groupCount"] = 1
		# 	resultMat = readCard(img, details).T
		# 	resultArray = resultMat[0]
		# else:
		# 	# 主观题需要裁剪出打分条
		# 	# imgFeature = cv2.imread("resources/scoreBar.png")
		# 	# boundingBox = siftMatchVertical(imgFeature, img)
		# 	# if len(boundingBox) == 0:
		# 	# 	raise ErrorStatusException("failed to extract score bar", STATUS_SCAN_ERROR)
		# 	# boundingBox = boundingBox[0]
		# 	# img = img[boundingBox[0][1]:boundingBox[3][1], boundingBox[0][0]:boundingBox[1][0]]
		# 	H, W, _ = img.shape
		# 	# 相对于宽度的高度
		# 	img = img[: int(self.SCORE_BAR_RATIO * W)]
		# 	# 划线
		# 	H, W, _ = img.shape
		# 	img = cv2.resize(img, (W * 7, H * 7))
		# 	centroid = lineMarking(img)
		# 	resultArray = centroidMarkingX(centroid, self.col, W * 7)
		
		# 所有题型都使用划线
		# 过滤出黑色区域
		if self.isColor == 1:
			img = filterBlack(img, color2 = [180, 255, 110])
		H, W, _ = img.shape
		# 主观题，截取打分条
		if self.quesType == SUBJECT: 
			# 相对于宽度的高度
			img = img[: int(self.SCORE_BAR_RATIO * W)]
		# 选择题，截取选择区域
		if self.quesType == CHOICE:
			choiceRatio = MARK_CHOICE_RATIO[self.paperType]
			leftArea = int(choiceRatio[0] * W)
			choiceWidth = int(choiceRatio[1] * W) * self.col
			choiceHeight = int(choiceRatio[2] * W)
			img = img[0: choiceHeight, leftArea: leftArea + choiceWidth]
		# cv2.imshow("img", img);cv2.waitKey(10)
		# 划线
		H, W, _ = img.shape
		img = cv2.resize(img, (W * 7, H * 7))
		centroid = lineMarking(img, drawImg = False, isColor = self.isColor)
		# 多次划线操作判断
		resultArray = centroidMarkingX(centroid, self.col, W * 7)
		self.setResult({"score": self.markingScore(resultArray), "anslist": resultArray}, STATUS_OK)

	def markingScore(self, resultArray):
		resultArray = np.array(resultArray)
		# 去掉多余的部分
		resultArray = resultArray[0: self.col]
		if self.quesType == CHOICE or self.quesType == JUDGE  or self.quesType == MULTI_CHOICE:
			# 正确答案里选择了的
			correctChoosed = resultArray[self.correctAns]
			# 不正确答案里选择了的
			uncorrectChoosed = resultArray[np.delete([i for i, item in enumerate(resultArray)], self.correctAns)]
			if np.sum(uncorrectChoosed) != 0:
				return 0
		if self.quesType == CHOICE:
			# 选择题
			if np.sum(correctChoosed) == 1:
				return self.totalScore
			else:
				return 0
		elif self.quesType == JUDGE:
			# 判断题
			if np.sum(correctChoosed) == 1:
				return self.totalScore
			else:
				return 0
		elif self.quesType == SUBJECT:
			# 主观题
			resultArray = np.array(resultArray)
			scoreArray = np.where(resultArray == 1)[0]
			if len(scoreArray) == 1:
				scoreIndex = scoreArray[0]
				if SCORE_BAR[scoreIndex] == -1:
					return self.totalScore
				else:
					singleScore = SCORE_BAR[scoreIndex]
					if singleScore > self.totalScore:
						return -1
					return singleScore
			elif len(scoreArray) == 2 or len(scoreArray) == 3:
				scoreIndex = scoreArray[0]
				plusScoreIndex = scoreArray[1]
				# 异常判定
				# 包含满分区域
				if scoreIndex == SCORE_BAR_AREA0_INDEX or plusScoreIndex == SCORE_BAR_AREA0_INDEX:
					return -1
				# 同时选中多个个位数区域
				elif (scoreIndex in SCORE_BAR_AREA1 and plusScoreIndex in SCORE_BAR_AREA1):
					return -1
				totalScore = SCORE_BAR[scoreIndex] + SCORE_BAR[plusScoreIndex]
				if len(scoreArray) == 3:
					plusplusScoreIndex = scoreArray[2]
					if plusplusScoreIndex != SCORE_BAR_AREA3_INDEX:
						return -1
					totalScore += SCORE_BAR[plusplusScoreIndex]
				if totalScore > self.totalScore:
					return -1
				return totalScore
			else:
				# 异常情况
				return -1
		elif self.quesType == MULTI_CHOICE:
			# 多选题
			if resultArray[self.correctAns].all() == 1:
				return self.totalScore
			else:
				return (np.sum(correctChoosed) / float(len(resultArray))) * self.totalScore

	@staticmethod
	def checkParams(self):
		quesType = self.getIntArg("quesType")
		totalScore = self.getIntArg("totalScore")
		# 正确答案的序号, 从0开始
		correctAns = self.getIntArgs("correctAns")
		self.isColor = self.getIntArg("isColor")
		col = self.getIntArg("col")
		if quesType not in [CHOICE, JUDGE, SUBJECT, MULTI_CHOICE]:
			raise ErrorStatusException("quesType must be a positive number in [1, 2, 3, 4]", STATUS_PARAM_ERROR)
		if totalScore == -1:
			raise ErrorStatusException("totalScore must be a positive number", STATUS_PARAM_ERROR)
		# 选择和判断题需要有总分
		if quesType == CHOICE or quesType == JUDGE or quesType == MULTI_CHOICE:
			if correctAns == []:
				raise ErrorStatusException("correctAns must be a list of non-negative number", STATUS_PARAM_ERROR)
			if (quesType == CHOICE or quesType == MULTI_CHOICE) and col == -1:
				raise ErrorStatusException("col must be a non-negative number", STATUS_PARAM_ERROR)
			if quesType == JUDGE and col == -1:
				col = 2
			if np.max(correctAns) >= col or np.min(correctAns) < 0:
				 raise ErrorStatusException("item of correctAns must be a correct", STATUS_PARAM_ERROR)
			self.paperType = "a3"
			# self.paperType = self.getStrArg("paperType", "a3").lower()
			# if self.paperType not in MARK_CHOICE_RATIO:
			# 	raise ErrorStatusException("paperType must be " + str(MARK_CHOICE_RATIO.keys()), STATUS_PARAM_ERROR)
		elif quesType == SUBJECT:
			col = len(SCORE_BAR)
		self.MAX_CHOICE_NUM = self.getIntArg("MAX_CHOICE_NUM", MAX_CHOICE_NUM)
		self.SCORE_BAR_RATIO = self.getFloatArg("SCORE_BAR_RATIO", SCORE_BAR_RATIO)
		self.quesType = quesType
		self.totalScore = totalScore
		self.correctAns = correctAns
		self.col = col
		if not self.fileExist("img"):
			self.imgUrl = self.getStrArg("img")
		else:
			self.imgUrl = None
		self.col = col
		self.row = 1
		


