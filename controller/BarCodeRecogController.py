# coding: utf-8
from BaseController import *
from util.ErrorCode import *
from lib.ScantronAnalyzeCV import *
import PIL.Image as Image
import numpy as np
import zbar
import cv2

class BarCodeRecogController(BaseController):
	def execute(self):
		rawData = self.processUpFile("barcode")
		img = cv2.imdecode(np.fromstring(rawData, np.uint8), cv2.IMREAD_GRAYSCALE)# IMREAD_COLOR
		img = Image.fromarray(img)
		w, h = img.size
		zbarImg = zbar.Image(w, h, 'Y800', img.tostring())
		scanner = zbar.ImageScanner()
		barCodeCount = scanner.scan(zbarImg)
		resultCode = -1
		for scanResult in zbarImg:
			resultCode = scanResult.data
			break
		del zbarImg
		self.setResult(resultCode, STATUS_OK)