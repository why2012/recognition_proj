# coding: utf-8
from BaseController import *
from util.ErrorCode import *
from lib.ScantronAnalyzeCV import *
import PIL.Image as Image
import numpy as np
import zbar
import cv2
import urllib2 as url
import platform
from lib.PreProcessing import *

class BarCodeRecogController(BaseController):
	def execute(self):
		BarCodeRecogController.checkParams(self)
		if not self.barcodeUrl:
			rawData = self.processUpFile("barcode")
		else:
			# 从其他地方获取图片
			res = url.urlopen(self.barcodeUrl)
			rawData = res.read()
		img = cv2.imdecode(np.fromstring(rawData, np.uint8), cv2.IMREAD_GRAYSCALE)
		# img = cv2.imdecode(np.fromstring(rawData, np.uint8), cv2.IMREAD_COLOR)
		# img = filterBlack(img, [0, 0, 0], [180, 255, 150])
		# cv2.imshow("img", img)
		# cv2.waitKey(10)
		img = Image.fromarray(img)
		w, h = img.size
		version = platform.python_version_tuple()
		if int(version[2]) >= 10:
			zbarImg = zbar.Image(w, h, 'Y800', img.tobytes())
		else:
			zbarImg = zbar.Image(w, h, 'Y800', img.tostring())
		scanner = zbar.ImageScanner()
		barCodeCount = scanner.scan(zbarImg)
		resultCode = -1
		for scanResult in zbarImg:
			resultCode = scanResult.data
			break
		del zbarImg
		self.setResult(resultCode, STATUS_OK)

	@staticmethod
	def checkParams(self):
		if not self.fileExist("barcode"):
			self.barcodeUrl = self.getStrArg("barcode")
		else:
			self.barcodeUrl = None