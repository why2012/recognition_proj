# coding: utf-8
from BaseController import *
from lib.HoughCircleSplit import *
import numpy as np
import conf.Config as conf
import urllib2 as url
from scipy import ndimage
from lib.PreProcessing import *
from lib.imagemagick import detectAndGetImage
import urlparse
import json
import os
import sys
reload(sys).setdefaultencoding( "utf-8" )

class HoughCircleSplitWithRotateController(BaseController):
	def execute(self):
		HoughCircleSplitWithRotateController.checkParams(self)
		if self.opType == 0:
			if not self.paperUrl:
				rawData = self.processUpFile("paper")
				img = cv2.imdecode(np.fromstring(rawData, np.uint8), cv2.IMREAD_COLOR)# IMREAD_COLOR
			else:
				# 从其他地方获取图片
				# url
				if self.paperUrl.lower().startswith("http"):
					res = url.urlopen(self.paperUrl)
					rawData = res.read()
					img = cv2.imdecode(np.fromstring(rawData, np.uint8), cv2.IMREAD_COLOR)# IMREAD_COLOR
				else:
				# local
					if os.path.exists(self.paperUrl):
						img = cv2.imread(self.paperUrl)
					else:
						if self.writeJson is not None:
							with open(self.writeJson, "w") as jsonfile:
								jsonfile.write(json.dumps({"status": STATUS_SCAN_ERROR, "ans": [], "msg": "文件不存在: %s" % self.paperUrl}, ensure_ascii=False));
						raise ErrorStatusException("文件不存在: %s" % self.paperUrl, STATUS_PARAM_ERROR)
			# 二维码
			QRCodeData = {"paperW": 1476, "paperH": 1011, "id": -1, "pageNumber": 1}
			img, qrcode = detectAndGetImage(img, self.imgFeature, "tmp/image/")
			if img is None or qrcode == -1:
				if self.writeJson is not None:
					with open(self.writeJson, "w") as jsonfile:
						jsonfile.write(json.dumps({"status": STATUS_SCAN_ERROR, "ans": [], "msg": "二维码无法识别"}, ensure_ascii=False));
				raise ErrorStatusException("二维码无法识别", STATUS_SCAN_ERROR)
				return
			imgH, imgW, _ = img.shape

			# queryData = urlparse.urlparse(qrcode).query
			# queryData = urlparse.parse_qs(queryData)

			print qrcode
			qrcode = qrcode.split("?")[1]
			qrcode = qrcode.split("&")
			queryData = {}
			queryData['paperType'] = qrcode[0]
			queryData['pageNumber'] = qrcode[1]
			queryData['id'] = qrcode[2]
			# 不读取二维码
			queryData['paperType'] = self.paperType
			print queryData
			# # ---test---
			# queryData['paperType'] = ['a3']
			# queryData['id'] = ['111']
			# queryData['pageNumber'] = ['111']
			# # -----------
			if 'paperType' not in queryData or 'pageNumber' not in queryData or 'id' not in queryData:
				self.setResult([], STATUS_SCAN_ERROR)
				return
			QRCodeData['paperType'] = queryData['paperType'].lower()
			QRCodeData['pageNumber'] = queryData['pageNumber']
			QRCodeData['id'] = queryData['id']

			# 单开
			if QRCodeData['paperType'] in ["a4", "16k", "b5"]:
				QRCodeData["paperW"] = 1300
				QRCodeData["paperH"] = 2000
			# 双开
			else:
				QRCodeData["paperW"] = 1476
				QRCodeData["paperH"] = 1011
			originImg = img
			# img = filterBlue(img)
			if self.isMobile == -1:
				img = filterBlack(img, [0, 0, 0], [180, 255, 100])
				# img = filterBlack(img)
				if imgW >= 2000:
					resizeScale = 0.5		
				else:
					resizeScale = 0.8
				resizeW, resizeH = (int(imgW * resizeScale), int(imgH * resizeScale))
				img = cv2.resize(img, (resizeW, resizeH))
				# (circles, imgList) = circleSplitMobile(img, QRCodeData["paperW"], QRCodeData["paperH"], scaleThresh = 1.0, colorImg = originImg, resizeScale = resizeScale, records = True, showImg = False)
				(circles, imgList) = circleSplitPlus(img, QRCodeData["paperW"], QRCodeData["paperH"], colorImg = originImg, resizeScale = resizeScale, scaleThresh = 1.0, showImg = False)
			else:
				img = filterBlack(img, [0, 0, 0], [180, 255, 100])
				# img = filterBlack(img)
				if imgW >= 2000:
					resizeScale = 0.5		
				else:
					resizeScale = 0.8
				resizeW, resizeH = (int(imgW * resizeScale), int(imgH * resizeScale))
				img = cv2.resize(img, (resizeW, resizeH))
				# (circles, imgList) = circleSplitMobile(img, QRCodeData["paperW"], QRCodeData["paperH"], scaleThresh = 1.0, colorImg = originImg, resizeScale = resizeScale, records = True, showImg = False)
				(circles, imgList) = circleSplitMobilePlus(img, QRCodeData["paperW"], QRCodeData["paperH"], scaleThresh = 1.0, colorImg = originImg, resizeScale = resizeScale, records = True, showImg = False)
			if len(imgList) > 0 and self.opType == 0:
				# cv2.imwrite("resources/tmp/tmp.png", imgList[0])
				retval, buf = cv2.imencode(".jpg", imgList[0])
				if retval:
					# with open("tmp/data/%s.qrdata" % self.qrid, "w") as qrfile:
					# 	qrfile.write(json.dumps(queryData, ensure_ascii=False));
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
				if self.writeJson is not None:
					with open(self.writeJson, "w") as jsonfile:
						jsonfile.write(json.dumps({"status": STATUS_SCAN_ERROR, "ans": [], "msg": "识别错误"}, ensure_ascii=False));
				self.setResult([], STATUS_SCAN_ERROR)
		# 返回二维码数据
		if self.opType == 1:
			filename = "tmp/data/%s.qrdata" % self.qrid
			suffix = "read"
			if not os.path.isfile(filename):
				filename = "tmp/data/%s.qrdata.%s" % (self.qrid, suffix)
			if os.path.isfile(filename):
				with open(filename, "r") as qrfile:
					qrdata = json.loads(qrfile.readline())
				self.setResult(qrdata, STATUS_OK)
				if not filename.endswith(suffix):
					newFilename = filename + "." + suffix
					os.rename(filename, newFilename)
				readFile = filter(lambda f: f.endswith("read"), os.listdir("tmp/data/"))
				# 移除历史数据
				if readFile is not None and len(readFile) >= 10000:
					os.system("rm tmp/data/*.read")
			else:
				raise ErrorStatusException("qrid[%s] does not exit" % self.qrid, STATUS_PARAM_ERROR)

	@staticmethod
	def checkParams(self):
		opType = self.getIntArg("opType")
		self.writeJson = self.getStrArg("writeJson", None)
		if opType < 0:
			opType = 0 # 返回图片
		if opType != 0 and opType !=1:
			raise ErrorStatusException("opType must be a positive number(0: 返回图片, 1: 根据图片id返回二维码)", STATUS_PARAM_ERROR)
		self.opType = opType
		if opType == 0:
			self.isMobile = self.getIntArg("isMobile")
			self.imgFeature = self.getStrArg("imgFeature", "resources/qr.jpg")
			self.imgFeature = cv2.imread(self.imgFeature)
			if self.imgFeature is None:
				raise ErrorStatusException("imgFeature must be a valid resource name", STATUS_PARAM_ERROR)
			
			if not self.fileExist("paper"):
				self.paperUrl = self.getStrArg("paper")
			else:
				self.paperUrl = None

		qrid = self.getStrArg("qrid")
		if qrid is None or qrid == "":
			raise ErrorStatusException("qrid must not be null, its a unique identification.", STATUS_PARAM_ERROR)
		self.qrid = qrid

		paperType = self.getStrArg("paperType")
		if paperType is None or paperType == "":
			raise ErrorStatusException("paperType must not be null.", STATUS_PARAM_ERROR)
		self.paperType = paperType


