# coding: utf-8
import uuid
import os
import numpy as np
import PIL.Image as Image
import cv2
from SiftMatch import *
import platform
import zbar
import math
from lib.PreProcessing import *

def detectAndGetImage(img, imgFeature, baseDir):
	#print "-------"
	orientation, qrcode = detectOrientation(img, imgFeature)
	if orientation == -1:
		return None, -1
	# #print "-------1"
	# imgPath = writeImage(img, baseDir)
	# #print "-------2"
	# rotateImage(imgPath, orientation)
	# img =  cv2.imread(imgPath)
	# deleteImage(imgPath)
	# #print "-------3"
	# return img, qrcode

	img = rotateImageWithCV(img, orientation)
	return img, qrcode

def readQR(img):
	# img = filterBlack(img, [0, 0, 0], [180, 255, 110]) # 110
	originImg = img
	for range02 in [[180, 255, 105], [180, 255, 110]]:
		img = filterBlackOriginImg(originImg, [0, 0, 0], range02)
		# H, W, _ = img.shape
		# img = cv2.resize(img, (W * 2, H * 2))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# cv2.imshow("img" + str(np.random.randint(100)), img)
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
		if resultCode != -1:
			return resultCode
	return resultCode

def detectOrientation(img, imgFeature):
	H, W, _ = img.shape
	leftTopPoint = [[H / 2, W / 2], [H / 2, 0], [0, 0], [0, W / 2]]
	stepSize = [[H, W], [H, W / 2], [H / 2, W / 2], [H / 2, W]]
	orientation = -1 # 1, 2, 3, 4 bottomRight, bottomLeft, topLeft, topRight
	qrcode = -1
	# cv2.imshow("img " , img)
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
		# print [leftTopPoint[i][0], stepSize[i][0], leftTopPoint[i][1], stepSize[i][1]]
		# print qrcode, i
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

def rotate_about_center(src, angle, scale = 1.):
	w = src.shape[1]
	h = src.shape[0]
	rangle = np.deg2rad(angle)  # angle in radians
	# now calculate new image width and height
	nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
	nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
	# ask OpenCV for the rotation matrix
	rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
	# calculate the move from the old center to the new center combined
	# with the rotation
	rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5,0]))
	# the move only affects the translation, so update the translation
	# part of the transform
	rot_mat[0, 2] += rot_move[0]
	rot_mat[1, 2] += rot_move[1]
	return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags = cv2.INTER_LANCZOS4)

# 顺时针
def rotateImageWithCV(img, orientation):
	if orientation not in [1, 2, 3, 4]:
		return
	degree = 0
	if orientation == 1:
		degree = 180
	elif orientation == 2:
		degree = 90
	elif orientation == 3:
		degree = 0
	elif orientation == 4:
		degree = -90
	if degree != 0:
		return rotate_about_center(img, -degree)
	else:
		return img

# 顺时针
def rotateImageWithConvert(imgPath, orientation):
	if orientation not in [1, 2, 3, 4]:
		return
	img = cv2.imread(imgPath)
	degree = 0
	if orientation == 1:
		degree = 180
	elif orientation == 2:
		degree = 90
	elif orientation == 3:
		degree = 0
	elif orientation == 4:
		degree = -90
	if degree != 0:
		os.system("convert -rotate %d %s %s" % (degree, imgPath, imgPath))