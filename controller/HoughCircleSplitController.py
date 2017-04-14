# coding: utf-8
from BaseController import *
from lib.HoughCircleSplit import *
import numpy as np
import conf.Config as conf
import urllib2 as url

class HoughCircleSplitController(BaseController):
	def execute(self):
		HoughCircleSplitController.checkParams(self)
		if not self.paperUrl:
			rawData = self.processUpFile("paper")
		else:
			# 从其他地方获取图片
			res = url.urlopen(self.paperUrl)
			rawData = res.read()
		img = cv2.imdecode(np.fromstring(rawData, np.uint8), cv2.IMREAD_COLOR)# IMREAD_COLOR
		imgList = circleSplit(img, self.paperW, self.paperH)
		if len(imgList) > 0:
			retval, buf = cv2.imencode(".jpg", imgList[0])
			if retval:
				if int(self.version[2]) >= 10:
					rawData = buf.tobytes()
				else:
					rawData = buf.tostring()
				self.set_header('Content-Type', 'image/jpeg')
				self.write(rawData)
				self.flush()
			else:
				self.setResult([], STATUS_ENCODE_ERROR)
		else:
			self.setResult([], STATUS_SCAN_ERROR)

	@staticmethod
	def checkParams(self):
		paperW = self.getIntArg("paperW")
		paperH = self.getIntArg("paperH")
		if paperW <= 0:
			raise ErrorStatusException("paperW must be a positive number", STATUS_PARAM_ERROR)
		if paperH <= 0:
			raise ErrorStatusException("paperH must be a positive number", STATUS_PARAM_ERROR)
		if not self.fileExist("paper"):
			self.paperUrl = self.getStrArg("paper")
		else:
			self.paperUrl = None
		self.paperW = paperW
		self.paperH = paperH
