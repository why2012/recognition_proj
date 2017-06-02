# coding: utf-8
from BaseController import *
from lib.SiftMatch import *
import numpy as np
import conf.Config as conf
import urllib2 as url
import os

class AreaExceptionDetectController(BaseController):
	def execute(self):
		AreaExceptionDetectController.checkParams(self)
		if not self.imgDestUrl:
			rawData = self.processUpFile("imgDest")
		else:
			# 从其他地方获取图片
			rawData = url.urlopen(self.imgDestUrl).read()
		imgDest = cv2.imdecode(np.fromstring(rawData, np.uint8), cv2.IMREAD_COLOR)# IMREAD_COLOR
		imgFeature = cv2.imread("resources/" + self.imgFeature)
		H, W, _ = imgDest.shape
		# boundingBox = siftMatchVertical(imgFeature, imgDest)
		boundingBox = siftMatchVertical(imgFeature, imgDest, windowHeightRate = 1.0, method = "SIFT", octaveLayers = 15, resizeScale = 2, useFlann = False, showImg = False)
		if boundingBox is not None and len(boundingBox) != 0:
			boundingBox[:, :, 0] = boundingBox[:, :, 0] / float(W)
			boundingBox[:, :, 1] = boundingBox[:, :, 1] / float(H)
			boundingBox = boundingBox[0]
			W = boundingBox[1][0] - boundingBox[0][0]
			H = boundingBox[3][1] - boundingBox[0][1]
			whRatio = float(W) / H
			exception = not (whRatio >= (self.whRatio - self.thresh) and whRatio <= (self.whRatio + self.thresh))
			self.setResult({"whRatio": whRatio, "exception": int(exception)}, STATUS_OK)
		else:
			self.setResult({"whRatio": -1, "exception": 1}, STATUS_SCAN_ERROR)

	@staticmethod
	def checkParams(self):
		whRatio = self.getIntArg("whRatio")
		self.thresh = self.getIntArg("thresh", 4)
		imgFeature = self.getStrArg("imgFeature").strip()
		if whRatio == -1:
			raise ErrorStatusException("whRatio must be a float number", STATUS_PARAM_ERROR)
		if imgFeature == "" or not os.path.exists("resources/" + imgFeature):
			raise ErrorStatusException("imgFeature must be a valid resource name", STATUS_PARAM_ERROR)
		if not self.fileExist("imgDest"):
			self.imgDestUrl = self.getStrArg("imgDest")
		else:
			self.imgDestUrl = None
		self.imgFeature = imgFeature
		self.whRatio = whRatio

