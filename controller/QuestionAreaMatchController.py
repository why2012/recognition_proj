# coding: utf-8
from BaseController import *
from lib.SiftMatch import *
import numpy as np
import conf.Config as conf
import urllib2 as url
import os

class QuestionAreaMatchController(BaseController):
	def execute(self):
		QuestionAreaMatchController.checkParams(self)
		if not self.imgDestUrl:
			rawData = self.processUpFile("imgDest")
		else:
			# 从其他地方获取图片
			rawData = url.urlopen(self.imgDestUrl).read()
		imgDest = cv2.imdecode(np.fromstring(rawData, np.uint8), cv2.IMREAD_COLOR)# IMREAD_COLOR
		H, W, _ = imgDest.shape
		boundingBoxes = []
		octaveLayerConst = {"choice": 8, "judge": 5, "subject": 8}
		for imgFeaturePath in self.imgFeaturePathList:
			imgFeature = cv2.imread("resources/" + imgFeaturePath)
			# [[[], [], [], []], []]
			if imgFeaturePath.find("choice") != -1:
				octaveLayers = octaveLayerConst["choice"]
			elif imgFeaturePath.find("judge") != -1:
				octaveLayers = octaveLayerConst["judge"]
			elif imgFeaturePath.find("subject") != -1:
				octaveLayers = octaveLayerConst["subject"]
			else:
				octaveLayers = 7
			# boundingBox = siftMatchVertical(imgFeature, imgDest, showImg = False, windowHeightRate = 0.04, octaveLayers = octaveLayers)
			# windowHeightRate = 0.06
			boundingBox = siftMatchVertical(imgFeature, imgDest, windowHeightRate = 0.02, method = "SIFT", resizeScale = 1.0, showImg = True)
			if boundingBox != []:
				boundingBox[:, :, 0] = boundingBox[:, :, 0] / float(W)
				boundingBox[:, :, 1] = boundingBox[:, :, 1] / float(H)
			boundingBoxes.append(boundingBox.tolist())
		self.setResult(boundingBoxes, STATUS_OK)

	@staticmethod
	def checkParams(self):
		imgFeatures = self.getStrArg("imgFeatures").strip()
		if imgFeatures != "":
			self.imgFeaturePathList = imgFeatures.split(",")
			for path in self.imgFeaturePathList:
				if not os.path.exists("resources/" + path):
					raise ErrorStatusException("imgFeatures must be a valid resource name", STATUS_PARAM_ERROR)
		else:
			raise ErrorStatusException("imgFeatures must not be null", STATUS_PARAM_ERROR)
		if not self.fileExist("imgDest"):
			self.imgDestUrl = self.getStrArg("imgDest")
		else:
			self.imgDestUrl = None
		self.imgFeatures = imgFeatures

