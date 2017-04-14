# coding: utf-8
from BaseController import *
from lib.SiftMatch import *
import numpy as np
import conf.Config as conf
import urllib2 as url
import os

class SiftMatchController(BaseController):
	def execute(self):
		SiftMatchController.checkParams(self)
		if not self.imgDestUrl:
			rawData = self.processUpFile("imgDest")
		else:
			# 从其他地方获取图片
			rawData = url.urlopen(self.imgDestUrl).read()
		imgDest = cv2.imdecode(np.fromstring(rawData, np.uint8), cv2.IMREAD_COLOR)# IMREAD_COLOR
		imgFeature = cv2.imread("resources/" + self.imgFeature)
		boundingBox = siftMatchVertical(imgFeature, imgDest)
		self.setResult(boundingBox.tolist(), STATUS_OK)

	@staticmethod
	def checkParams(self):
		imgFeature = self.getStrArg("imgFeature").strip()
		if imgFeature == "" and os.path.exists("resources/" + imgFeature):
			raise ErrorStatusException("imgFeature must be a valid resource name", STATUS_PARAM_ERROR)
		if not self.fileExist("imgDest"):
			self.imgDestUrl = self.getStrArg("imgDest")
		else:
			self.imgDestUrl = None
		self.imgFeature = imgFeature

