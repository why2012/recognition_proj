# coding: utf-8
import uuid
import os
import numpy as np
import PIL.Image as Image
import cv2
from SiftMatch import *
import platform
import zbar

def detectAndGetImage(img, imgFeature, baseDir):
	#print "-------"
	orientation, qrcode = detectOrientation(img, imgFeature)
	if orientation == -1:
		return None, -1
	#print "-------1"
	imgPath = writeImage(img, baseDir)
	#print "-------2"
	rotateImage(imgPath, orientation)
	img =  cv2.imread(imgPath)
	deleteImage(imgPath)
	#print "-------3"
	return img, qrcode, imgPath

def readQR(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
	return resultCode

def detectOrientation(img, imgFeature):
	H, W, _ = img.shape
	leftTopPoint = [[H / 2, W / 2], [0, W / 2], [0, 0], [H / 2, 0]]
	stepSize = [[H, W], [H / 2, W], [H / 2, W / 2], [H, W / 2]]
	orientation = -1 # 1, 2, 3, 4 bottomRight, topRight, topLeft, bottomRight
	qrcode = -1
	for i in range(4):
		# cropImg = img[leftTopPoint[i][0]: stepSize[i][0], leftTopPoint[i][1]: stepSize[i][1]]
		# localH, localW, _ = cropImg.shape
		# # cropImg = cv2.resize(cropImg, (localW * 3, localH * 3))
		# boundingBox = siftMatchVertical(imgFeature, cropImg, windowHeightRate = 1.0, method = "SURF", octaveLayers = 1, resizeScale = 2, pyrDown = True, enableSubWindow = True, pyrDownRate = 2, useFlann = True, showImg = False)
		# #cv2.imshow("img %d" % (i), cropImg)
		# #cv2.imwrite("tmp/img" + str(i) + ".png", cropImg)
		# print boundingBox
		# if boundingBox is not None and len(boundingBox) != 0:
		# 	orientation = i + 1
		# 	# qrcode = readQR(cropImg)
		# 	qrcode = readQR(img)
		# 	print qrcode
		# 	break

		cropImg = img[leftTopPoint[i][0]: stepSize[i][0], leftTopPoint[i][1]: stepSize[i][1]]
		qrcode = readQR(cropImg)
		# cv2.imshow("img %d" % (i), cropImg)
		# cv2.waitKey(10)
		if qrcode != -1:
			orientation = i + 1
			break
	
	return orientation, qrcode

def writeImage(img, pathDir):
	imgName = str(uuid.uuid1())
	extName = ".png"
	if not pathDir.endswith("/"):
		pathDir += "/"
	imgPath = pathDir + imgName + extName
	cv2.imwrite(imgPath, img)
	return imgPath

def deleteImage(imgPath):
	os.system("rm %s" % (imgPath))

# 顺时针
def rotateImage(imgPath, orientation):
	if orientation not in [1, 2, 3, 4]:
		return
	img = cv2.imread(imgPath)
	degree = 0
	if orientation == 1:
		degree = 180
	elif orientation == 2:
		degree = -90
	elif orientation == 3:
		degree = 0
	elif orientation == 4:
		degree = 90
	os.system("convert -rotate %d %s %s" % (degree, imgPath, imgPath))