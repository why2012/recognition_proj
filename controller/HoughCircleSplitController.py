# coding: utf-8
from BaseController import *
from lib.HoughCircleSplit import *
import numpy as np
import conf.Config as conf
import urllib2 as url
from scipy import ndimage
from lib.PreProcessing import *

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
		# 二维码识别
		
		# 二维码
		QRCodeData = {"paperW": 1200, "paperH": 2000}
		if self.paperW == -1:
			self.paperW = QRCodeData["paperW"]
		if self.paperH == -1:
			self.paperH = QRCodeData["paperH"]
		imgH, imgW, _ = img.shape
		print (imgW, imgH), (self.paperW, self.paperH), self.isMobile, self.paperUrl
		# 过滤出黑色区域
		originImg = img
		# img = filterBlue(img)
		if self.isMobile == -1:
			img = filterBlack(img)
			if imgW >= 2000:
				resizeScale = 0.5		
			else:
				resizeScale = 0.8
			resizeW, resizeH = (int(imgW * resizeScale), int(imgH * resizeScale))
			# 缩放至固定尺寸，方便调参
			# resizeW, resizeH = (int(self.paperW / 3.36), int(self.paperH / 3.46))# (1476, 1011) # 1300, 2000
			img = cv2.resize(img, (resizeW, resizeH))
			# (circles, imgList) = circleSplit(img, self.paperW, self.paperH, scaleThresh = 1.0, showImg = False)
			(circles, imgList) = circleSplitPlus(img, self.paperW, self.paperH, colorImg = originImg, resizeScale = resizeScale, scaleThresh = 1.0, showImg = False)
		else:
			# img = filterBlack(img, [0, 0, 0], [180, 255, 90])
			img = filterBlack(img)
			if imgW >= 2000:
				resizeW, resizeH = (int(imgW * 0.5), int(imgH * 0.5))
			else:
				resizeW, resizeH = (int(imgW * 0.8), int(imgH * 0.8))
			originImg = cv2.resize(originImg, (resizeW, resizeH))
			img = cv2.resize(img, (resizeW, resizeH))
			(circles, imgList) = circleSplitMobile(img, self.paperW, self.paperH, scaleThresh = 1.0, colorImg = originImg, showImg = False)
		if len(imgList) > 0 and self.opType == 0:
			# cv2.imwrite("resources/tmp/tmp.png", imgList[0])
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
		elif self.opType == 1:
			for i, item in enumerate(circles):
				if item is not None:
					item[0] = item[0] * imgW / resizeW
					item[1] = item[1] * imgH / resizeH
					circles[i] = item.tolist()
			self.setResult({"coords": circles, "qr": QRCodeData}, STATUS_OK)
		else:
			self.setResult([], STATUS_SCAN_ERROR)

	@staticmethod
	def checkParams(self):
		paperW = self.getIntArg("paperW")
		paperH = self.getIntArg("paperH")
		opType = self.getIntArg("opType")
		self.isMobile = self.getIntArg("isMobile")
		if opType < 0:
			opType = 0 # 返回图片
		if opType != 0 and opType !=1:
			raise ErrorStatusException("opType must be a positive number(0: 返回图片, 1: 返回坐标点和二维码)", STATUS_PARAM_ERROR)
		self.opType = opType
		# if paperW <= 0:
		# 	raise ErrorStatusException("paperW must be a positive number", STATUS_PARAM_ERROR)
		# if paperH <= 0:
		# 	raise ErrorStatusException("paperH must be a positive number", STATUS_PARAM_ERROR)
		if not self.fileExist("paper"):
			self.paperUrl = self.getStrArg("paper")
		else:
			self.paperUrl = None
		self.paperW = paperW
		self.paperH = paperH
