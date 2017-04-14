# coding: utf-8
from BaseController import *
from lib.CustomMarking import *
import numpy as np
import conf.Config as conf
import urllib2 as url

class CustomMarkingController(BaseController):
	def execute(self):
		CustomMarkingController.checkParams(self)
		if not self.imgDestUrl:
			rawData = self.processUpFile("img")
		else:
			# 从其他地方获取图片
			rawData = url.urlopen(self.imgDestUrl).read()
		imgDest = cv2.imdecode(np.fromstring(rawData, np.uint8), cv2.IMREAD_COLOR)# IMREAD_COLOR
		centroid = lineMarking(imgDest)
		self.setResult(centroid, STATUS_OK)

	@staticmethod
	def checkParams(self):
		if not self.fileExist("img"):
			self.imgDestUrl = self.getStrArg("img")
		else:
			self.imgDestUrl = None
