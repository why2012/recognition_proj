#coding:utf-8
import cv2
import numpy as np
import json
import lib.ScantronAnalyzeCV as sc
import pickle
from sklearn.svm import SVC
import os

# 图片宽度对应数组列数，高度对应数组行数
def getImgSize(img):
	return (img.shape[1], img.shape[0])

def grayImg(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def getKernel(size):
	kernel = np.uint8(np.zeros(size))
	kw, kh = kernel.shape
	for x in range(kh):  
		kernel[x, kw / 2] = 1 
	for x in range(kw):  
		kernel[kh / 2, x] = 1 
	return kernel

def erosion(grayImg, kernel = getKernel((4, 4)), iterations = 1):
	return cv2.erode(grayImg, kernel, iterations = iterations)

def dilation(grayImg, kernel = getKernel((4, 4)), iterations = 1):
	return cv2.dilate(grayImg, kernel, iterations = iterations)  

def showImgs(*imgs):
	index = 0
	for img in imgs:
		cv2.imshow("img" + str(index), img)
		index += 1
	cv2.waitKey(10)
	cv2.destroyAllWindows()

# 畸变矫正幅度
def getSkewScale(topLeft, topRight, bottomRight, bottomLeft):
	return []

def predictUsingSVM(c1, c2, c3, c4, whRatio, thresh = 0):
	topLeft = np.array([c1[0], c1[1]])
	topRight = np.array([c2[0], c2[1]])
	bottomRight = np.array([c3[0], c3[1]])
	bottomLeft = np.array([c4[0], c4[1]])
	data = []
	data.extend(topLeft)
	data.extend(topRight)
	data.extend(bottomRight)
	data.extend(bottomLeft)
	with open("resources/dbdata/svmdata.svm") as file:
 		svmo = pickle.load(file)
	result = svmo.predict([data])
	# print "---ML---", result, data
	if result[0] == 1:
		difference = 1 - svmo.predict_proba([data])[0][0]
		return True, (topLeft, topRight, bottomRight, bottomLeft), getSkewScale(topLeft, topRight, bottomRight, bottomLeft), difference
	else:
		return False, (), [], float('inf')

def determineBoxRatio(c1, c2, c3, c4, whRatio, thresh = 0.2):
	topLeft = np.array([c1[0], c1[1]])
	topRight = np.array([c2[0], c2[1]])
	bottomRight = np.array([c3[0], c3[1]])
	bottomLeft = np.array([c4[0], c4[1]])
	# 上下，左右，宽高
	w1 = np.sum((topRight - topLeft) * (topRight - topLeft))
	w2 = np.sum((bottomRight - bottomLeft) * (bottomRight - bottomLeft))
	h1 = np.sum((bottomLeft - topLeft) * (bottomLeft - topLeft))
	h2 = np.sum((bottomRight - topRight) * (bottomRight - topRight))
	# 宽高比
	staRatio = whRatio * whRatio
	ratio1 = w1 / h1
	ratio2 = w2 / h2
	ratio3 = w1 / h2
	ratio4 = w2 / h1
	diagLength1 = np.sum((topLeft - bottomRight) * (topLeft - bottomRight))
	diagLength2 = np.sum((topRight - bottomLeft) * (topRight - bottomLeft))
	# 宽高比判定
	whRatioBool = (ratio1 >= staRatio - thresh and ratio1 <= staRatio + thresh) and (ratio2 >= staRatio - thresh and ratio2 <= staRatio + thresh) \
				and (ratio3 >= staRatio - thresh and ratio3 <= staRatio + thresh) and (ratio4 >= staRatio - thresh and ratio4 <= staRatio + thresh)
	# 对角线相对长度
	diagLengthRatio = diagLength1 / diagLength2
	diagLengthRatioBool = (diagLengthRatio >= 1 - thresh) and (diagLengthRatio <= 1 + thresh)
	# 半径方差
	radiusArr = np.array((c1[2], c2[2], c3[2], c4[2]))
	radiusVar = np.sum(np.power(radiusArr - np.array([np.average(radiusArr)] * 4), 2))

	# 宽高比与对角线长度
	# if diagLengthRatioBool:
	# 	print whRatioBool, diagLengthRatioBool, staRatio
	# 	print topLeft.tolist()
	# 	print topRight.tolist()
	# 	print bottomRight.tolist()
	# 	print bottomLeft.tolist()
	# 	print ratio1, ratio2, ratio3, ratio4
	# 	print "--------------"
	if whRatioBool and diagLengthRatioBool:
		# 差异度
		# difference = np.abs(diagLengthRatio - thresh) + radiusVar
		difference = topLeft[0] + topLeft[1] + radiusVar
		return True, (topLeft, topRight, bottomRight, bottomLeft), getSkewScale(topLeft, topRight, bottomRight, bottomLeft), difference
	else:
		return False, (), [], float('inf')

def determineBoxRatioMobile(c1, c2, c3, c4, whRatio, thresh = 0.4):
	topLeft = np.array([c1[0], c1[1]])
	topRight = np.array([c2[0], c2[1]])
	bottomRight = np.array([c3[0], c3[1]])
	bottomLeft = np.array([c4[0], c4[1]])
	# 上下，左右，宽高
	w1 = np.sum((topRight - topLeft) * (topRight - topLeft))
	w2 = np.sum((bottomRight - bottomLeft) * (bottomRight - bottomLeft))
	h1 = np.sum((bottomLeft - topLeft) * (bottomLeft - topLeft))
	h2 = np.sum((bottomRight - topRight) * (bottomRight - topRight))
	# 宽高比
	staRatio = whRatio * whRatio
	ratio1 = w1 / h1
	ratio2 = w2 / h2
	ratio3 = w1 / h2
	ratio4 = w2 / h1
	diagLength1 = np.sum((topLeft - bottomRight) * (topLeft - bottomRight))
	diagLength2 = np.sum((topRight - bottomLeft) * (topRight - bottomLeft))
	# 宽高比判定
	whRatioBool = (ratio1 >= staRatio - thresh and ratio1 <= staRatio + thresh) and (ratio2 >= staRatio - thresh and ratio2 <= staRatio + thresh) \
				and (ratio3 >= staRatio - thresh and ratio3 <= staRatio + thresh) and (ratio4 >= staRatio - thresh and ratio4 <= staRatio + thresh)
	whRatioDiff = np.abs(ratio1 - staRatio) + np.abs(ratio2 - staRatio) + np.abs(ratio3 - staRatio) + np.abs(ratio4 - staRatio)
	# 对角线相对长度
	diagLengthRatio = diagLength1 / diagLength2
	diagLengthRatioBool = (diagLengthRatio >= 1 - thresh) and (diagLengthRatio <= 1 + thresh)
	# 半径方差
	radiusArr = np.array((c1[2], c2[2], c3[2], c4[2]))
	radiusVar = np.sum(np.power(radiusArr - np.array([np.average(radiusArr)] * 4), 2))
	# 宽高比与对角线长度
	# if diagLengthRatioBool:
	# 	print whRatioBool, diagLengthRatioBool, staRatio
	# 	print topLeft.tolist()
	# 	print topRight.tolist()
	# 	print bottomRight.tolist()
	# 	print bottomLeft.tolist()
	# 	print ratio1, ratio2, ratio3, ratio4
	# 	print "--------------"
	if whRatioBool and diagLengthRatioBool:
		# 差异度
		difference = topLeft[0] + topLeft[1] + 0.7 * radiusVar  + 0.7 * (topRight[0] + topRight[1]) + 5 * np.abs((topRight[1] - topLeft[1])) + np.abs((topRight[0] - bottomRight[0])) + 15 * whRatioDiff + 5 * np.abs((bottomRight[1] - bottomLeft[1]))
		# # 宽高比与对角线长度
		# print whRatioBool, diagLengthRatioBool, staRatio
		# print difference, topLeft[0] + topLeft[1], 0.7 * radiusVar, 0.7 * (topRight[0] + topRight[1]), 5 * np.abs((topRight[1] - topLeft[1])), np.abs((topRight[0] - bottomRight[0])), 15 * whRatioDiff, 5 * np.abs((bottomRight[1] - bottomLeft[1]))
		# print topLeft.tolist()
		# print topRight.tolist()
		# print bottomRight.tolist()
		# print bottomLeft.tolist()
		# print ratio1, ratio2, ratio3, ratio4
		# print "--------------"
		return True, (topLeft, topRight, bottomRight, bottomLeft), getSkewScale(topLeft, topRight, bottomRight, bottomLeft), difference
	else:
		return False, (), [], float('inf')

def determingCorrectCircles(circles, whRatio, isMobile = False):
	if len(circles) < 4:
		return [], (), []
	# 分别获取四个象限的点
	topLeftCircles = []
	topRightCircles = []
	bottomLeftCircles = []
	bottomRightCircles = []
	minX = np.min(circles[:, 0])
	maxX = np.max(circles[:, 0])
	minY = np.min(circles[:, 1])
	maxY = np.max(circles[:, 1])
	centerX = (minX + maxX) / 2.0
	centerY = (minY + maxY) / 2.0
	for curCircle in circles:
		x, y, r = curCircle
		if x <= centerX and y <= centerY:
			topLeftCircles.append(curCircle)
		elif x >= centerX and y <= centerY:
			topRightCircles.append(curCircle)
		elif x >= centerX and y >= centerY:
			bottomRightCircles.append(curCircle)
		else:
			bottomLeftCircles.append(curCircle)
	correctResult = []
	for circleTopLeft in topLeftCircles:
		for circleTopRight in topRightCircles:
			for circleBottomRight in bottomRightCircles:
				for circleBottomLeft in bottomLeftCircles:
					# 机器学习模式
					if os.path.exists("resources/dbdata/useml.lock") and os.path.exists("resources/dbdata/svmdata.svm"):
						result, corners, skewScale, difference = predictUsingSVM(circleTopLeft, circleTopRight, circleBottomRight, circleBottomLeft, whRatio)
					# 传统模式
					else:
						if not isMobile:
							result, corners, skewScale, difference = determineBoxRatio(circleTopLeft, circleTopRight, circleBottomRight, circleBottomLeft, whRatio)
						else:
							result, corners, skewScale, difference = determineBoxRatioMobile(circleTopLeft, circleTopRight, circleBottomRight, circleBottomLeft, whRatio)
					if result:
						correctResult.append({"diff": difference, "corners": corners, "skewScale": skewScale, \
							"circleTopLeft": circleTopLeft, "circleTopRight": circleTopRight, "circleBottomRight": circleBottomRight, \
							"circleBottomLeft": circleBottomLeft})
	if correctResult:
		# 选取差异度最小的candidate
		correctResult = sorted(correctResult, key = lambda a: a["diff"])
		# correctResult = sorted(correctResult, key = lambda a: a["circleTopLeft"][0])
		minDiffResult = correctResult[0]
		return minDiffResult["corners"], [minDiffResult["circleTopLeft"], minDiffResult["circleTopRight"], minDiffResult["circleBottomRight"], minDiffResult["circleBottomLeft"]], minDiffResult["skewScale"]
	return [], (), []


# main function, paperW,paperH: 目标区域宽高(相对比例)
# return split image
def circleSplit(originalImg, paperW, paperH, scaleThresh = 1.0, showImg = False):
	imgSize = getImgSize(originalImg)
	w, h = imgSize
	# 按比例缩放
	w, h = (np.int(w * scaleThresh), np.int(h * scaleThresh))
	# 目标区域宽高
	dw, dh = (int(paperW * scaleThresh), int(paperH * scaleThresh))
	minWH = np.min((w, h))
	if scaleThresh != 1.0:
		originalImg = cv2.resize(originalImg, (w, h))
	if showImg:
		# 调试， 画圆
		imgColor = originalImg.copy()
		# 调试， 画圆
		imgColor02 = originalImg.copy()
	img = grayImg(originalImg)
	# img = dilation(img, kernel = getKernel((10, 10)))
	# img = erosion(img, iterations = 4)
	# cv2.imwrite("resources/test.jpg", img)
	# 切割结果
	splitArea = np.array([])
	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minWH * 0.08, param1 = 16, param2 = 8, minRadius = int(np.ceil(minWH * 0.001)), maxRadius = int(np.ceil(minWH * 0.020)))
	if circles is None:
		return ([], [])
	# 只取半径大于平均值的圆
	avgRadius = np.average(circles[0, :, 2]) * 0.8
	# avgRadius = 0
	circles = np.array([circles[0, circles[0, :, 2] >= avgRadius]])
	# 确定四个边角圆
	corners, correctCircles, _ = determingCorrectCircles(circles[0], float(paperW) / paperH)
	corners = np.array(corners, dtype = np.float32)
	# 画出过滤前的圆
	if showImg and circles.any():
		# 调试：画圆
		circles = np.uint16(np.around(circles))
		for i in circles[0]:
			cv2.circle(imgColor02,(i[0],i[1]),i[2],(0,255,0),2)
			cv2.circle(imgColor02,(i[0],i[1]),2,(0,0,255),3)
	blockListImg = []
	if correctCircles:
		if showImg:
			# 调试：画圆
			correctCirclesUint = np.uint16(np.around(correctCircles))
			for i in correctCirclesUint:
				cv2.circle(imgColor,(i[0],i[1]),i[2],(0,255,0),2)
				cv2.circle(imgColor,(i[0],i[1]),2,(0,0,255),3)
		# 映射，切割
		transPs = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype = np.float32)
		transform = cv2.getPerspectiveTransform(corners, transPs)
		splitArea = cv2.warpPerspective(src = originalImg, M = transform, dsize =  (dw, dh))
		blockListImg.append(splitArea)
	if showImg:
		# showImgs(img, imgColor02, imgColor, *blockListImg)
		showImgs(img, imgColor02, imgColor)
	return (correctCircles, blockListImg)

def removeLargeBlackArea(img, thresh = 0.0007, showImg = False, dilationKS = 7):
	originalImg = img
	_, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)
	H, W = img.shape
	S = H * W
	contours, hierarchy = sc.findContours(img, cv2.RETR_LIST)
	filterContours = []
	for contour in contours:
		# print cv2.contourArea(contour), S * thresh
		if cv2.contourArea(contour) >= S * thresh:
			filterContours.append(contour)
	fgImg = sc.createWhiteImg((W, H))
	sc.drawContours(fgImg, filterContours, (0, 0, 0), -1)
	_, fgImg = cv2.threshold(fgImg, 10, 255, cv2.THRESH_BINARY_INV)
	img = np.uint8(fgImg + originalImg)
	img = dilation(img, kernel = getKernel((dilationKS, dilationKS)))
	_, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)
	contours_circle, hierarchy_circle = sc.findContours(img, cv2.RETR_LIST)
	if showImg:
		whiteImg = sc.createWhiteImg((W, H))
		sc.drawContours(whiteImg, contours_circle, (0, 0, 0), -1)
		cv2.imshow("contour_fg", img)
		cv2.imshow("contour", whiteImg)
		cv2.waitKey(10)
	return contours_circle

def getCircles(contours, thresh01 = 0.8):
	topCircles = []
	contourS = []
	for i, contour in enumerate(contours):
		contourS.append((cv2.contourArea(contour), i))
	contourS.sort(key = lambda x: x[0], reverse = True)
	if len(contourS) < 4:
		return []
	for cSI in contourS:
		cS = contours[cSI[1]]
		x, y, w, h = cv2.boundingRect(cS)
		whRatio = float(w) / h
		# print whRatio, 1 - thresh01, 1 + thresh01
		if whRatio >= 1 - thresh01 and whRatio <= 1 + thresh01:
			centralX, centralY, R = np.ceil((x * 2 + w) / 2.0), np.ceil((y * 2 + h) / 2.0), (w + h) / 4
			topCircles.append([centralX, centralY, R])
	return np.array(topCircles)

def circleSplitPlus(originalImg, paperW, paperH, colorImg, resizeScale, scaleThresh = 1.0, showImg = False):
	imgSize = getImgSize(originalImg)
	w, h = imgSize
	# 目标区域宽高
	dw, dh = paperW, paperH
	if showImg:
		imgColor = originalImg.copy()
		imgColor02 = originalImg.copy()
	img = grayImg(originalImg)
	contours_circle = removeLargeBlackArea(img, showImg = showImg, dilationKS = 13)
	circles = getCircles(contours_circle)
	if circles is None or len(circles) == 0:
		return ([], [])
	if len(circles) >= 10:
		# 只取半径大于平均值的圆
		avgRadius = np.average(circles[:, 2])
		# avgRadius = 0
		circles = np.array([circles[circles[:, 2] >= avgRadius]])[0]
	# print circles
	# 切割结果
	splitArea = np.array([])
	# 确定四个边角圆
	corners, correctCircles, _ = determingCorrectCircles(circles, float(paperW) / paperH, True)
	corners = np.array(corners, dtype = np.float32)
	# 画出过滤前的圆
	if showImg and circles.any():
		# 调试：画圆
		circles = np.uint16(np.around(circles))
		for i in circles:
			cv2.circle(imgColor02,(i[0],i[1]),i[2],(0,255,0),2)
			cv2.circle(imgColor02,(i[0],i[1]),2,(0,0,255),3)
	blockListImg = []
	if correctCircles:
		if showImg:
			# 调试：画圆
			correctCirclesUint = np.uint16(np.around(correctCircles))
			for i in correctCirclesUint:
				cv2.circle(imgColor,(i[0],i[1]),i[2],(0,255,0),2)
				cv2.circle(imgColor,(i[0],i[1]),2,(0,0,255),3)
		cH, cW, _ = colorImg.shape
		transPs = np.array([[0, 0], [cW, 0], [cW, cH], [0, cH]], dtype = np.float32)
		transform = cv2.getPerspectiveTransform(corners / resizeScale, transPs)
		splitArea = cv2.warpPerspective(src = colorImg, M = transform, dsize =  (cW, cH))

		blockListImg.append(splitArea)
	if showImg:
		showImgs(colorImg, img, imgColor02, imgColor)
	return (correctCircles, blockListImg)

def circleSplitMobilePlus(originalImg, paperW, paperH, colorImg, resizeScale, scaleThresh = 1.0, showImg = False, records = False):
	imgSize = getImgSize(originalImg)
	w, h = imgSize
	# 目标区域宽高
	dw, dh = paperW, paperH
	if showImg:
		imgColor = originalImg.copy()
		imgColor02 = originalImg.copy()
	img = grayImg(originalImg)
	contours_circle = removeLargeBlackArea(img, showImg = showImg)
	circles = getCircles(contours_circle)
	if circles is None or len(circles) == 0:
		return ([], [])
	if len(circles) >= 10:
		# 只取半径大于平均值的圆
		avgRadius = np.average(circles[:, 2]) * 0.4
		# avgRadius = 0
		circles = np.array([circles[circles[:, 2] >= avgRadius]])[0]
	# print circles
	# 切割结果
	splitArea = np.array([])
	# 确定四个边角圆
	corners, correctCircles, _ = determingCorrectCircles(circles, float(paperW) / paperH, True)
	corners = np.array(corners, dtype = np.float32)
	# 画出过滤前的圆
	if showImg and circles.any():
		# 调试：画圆
		circles = np.uint16(np.around(circles))
		for i in circles:
			cv2.circle(imgColor02,(i[0],i[1]),i[2],(0,255,0),2)
			cv2.circle(imgColor02,(i[0],i[1]),2,(0,0,255),3)
	blockListImg = []
	if correctCircles:
		if showImg:
			# 调试：画圆
			correctCirclesUint = np.uint16(np.around(correctCircles))
			for i in correctCirclesUint:
				cv2.circle(imgColor,(i[0],i[1]),i[2],(0,255,0),2)
				cv2.circle(imgColor,(i[0],i[1]),2,(0,0,255),3)
		cH, cW, _ = colorImg.shape
		transPs = np.array([[0, 0], [cW, 0], [cW, cH], [0, cH]], dtype = np.float32)
		transform = cv2.getPerspectiveTransform(corners / resizeScale, transPs)
		splitArea = cv2.warpPerspective(src = colorImg, M = transform, dsize =  (cW, cH))

		blockListImg.append(splitArea)
	if showImg:
		showImgs(colorImg, img, imgColor02, imgColor)
	return (correctCircles, blockListImg)

# 收集训练数据用
def circleSplitMobilePlusCollectData(originalImg, colorImg, resizeScale, scaleThresh = 1.0, showImg = False):
	imgSize = getImgSize(originalImg)
	w, h = imgSize
	imgColor02 = originalImg.copy()
	img = grayImg(originalImg)
	contours_circle = removeLargeBlackArea(img, showImg = showImg)
	circles = getCircles(contours_circle)
	if circles is None or len(circles) == 0:
		return imgColor02
	if len(circles) >= 10:
		# 只取半径大于平均值的圆
		avgRadius = np.average(circles[:, 2]) * 0.4
		# avgRadius = 0
		circles = np.array([circles[circles[:, 2] >= avgRadius]])[0]
	# print circles
	# 画出过滤前的圆
	if circles.any():
		circleIndex = 0
		circles = np.uint16(np.around(circles))
		for i in circles:
			cv2.circle(imgColor02,(i[0],i[1]),i[2],(0,255,0),2)
			cv2.circle(imgColor02,(i[0],i[1]),2,(0,0,255),3)
			cv2.putText(imgColor02, str(circleIndex), (i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
			circleIndex += 1
	if showImg:
		showImgs(colorImg, img, imgColor02)
	return circles, imgColor02

def circleSplitMobile(originalImg, paperW, paperH, colorImg, resizeScale, scaleThresh = 1.0, showImg = False, records = False):
	imgSize = getImgSize(originalImg)
	w, h = imgSize
	# 按比例缩放
	w, h = (np.int(w * scaleThresh), np.int(h * scaleThresh))
	# 目标区域宽高
	dw, dh = (int(paperW * scaleThresh), int(paperH * scaleThresh))
	minWH = np.min((w, h))
	if scaleThresh != 1.0:
		originalImg = cv2.resize(originalImg, (w, h))
		colorImg = cv2.resize(colorImg, (w, h))
	if showImg:
		# 调试， 画圆
		imgColor = originalImg.copy()
		# 调试， 画圆
		imgColor02 = originalImg.copy()
	img = grayImg(originalImg)
	# img = erosion(img, kernel = getKernel((7, 7)))
	# img = np.uint8(img - img * 0.6)

	# # ver-2-test
	# #tmpImg = img.copy()
	# imgBg = dilation(img, kernel = getKernel((40, 40)), iterations = 1)
	# _, imgBg = cv2.threshold(imgBg, 10, 255, cv2.THRESH_BINARY_INV)
	# imgBg = dilation(imgBg, kernel = getKernel((50, 50)), iterations = 1)
	# img = np.uint8(img + imgBg)
	# img = dilation(img, kernel = getKernel((15, 15)))
	# img = erosion(img, iterations = 8)
	# #showImgs(tmpImg, imgBg, img)
	# #return ([], [])

	# ver-1-valid
	# tmpImg = img.copy()
	imgBg = dilation(img, kernel = getKernel((30, 30)), iterations = 1)
	_, imgBg = cv2.threshold(imgBg, 10, 255, cv2.THRESH_BINARY_INV)
	img = np.uint8(img + imgBg)
	img = dilation(img, kernel = getKernel((10, 10)))
	img = erosion(img, iterations = 4)
	# showImgs(tmpImg, imgBg, img)
	# return ([], [])

	# cv2.imwrite("resources/test.jpg", img)
	# 切割结果
	splitArea = np.array([])
	# circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, minWH * 0.08, param1 = 30, param2 = 10, minRadius = int(np.ceil(minWH * 0.005)), maxRadius = int(np.ceil(minWH * 0.020)))
	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, minWH * 0.08, param1 = 30, param2 = 10, minRadius = int(np.ceil(minWH * 0.001)), maxRadius = int(np.ceil(minWH * 0.020)))
	if circles is None:
		return ([], [])
	originCircles = circles.copy()
	# 只取半径大于平均值的圆
	avgRadius = np.average(circles[0, :, 2]) * 0.8
	# avgRadius = 0
	circles = np.array([circles[0, circles[0, :, 2] >= avgRadius]])
	# 确定四个边角圆
	corners, correctCircles, _ = determingCorrectCircles(circles[0], float(paperW) / paperH, True)
	corners = np.array(corners, dtype = np.float32)
	# 画出过滤前的圆
	if showImg and circles.any():
		# 调试：画圆
		circles = np.uint16(np.around(circles))
		for i in circles[0]:
			cv2.circle(imgColor02,(i[0],i[1]),i[2],(0,255,0),2)
			cv2.circle(imgColor02,(i[0],i[1]),2,(0,0,255),3)
	blockListImg = []
	if correctCircles:
		if showImg:
			# 调试：画圆
			correctCirclesUint = np.uint16(np.around(correctCircles))
			for i in correctCirclesUint:
				cv2.circle(imgColor,(i[0],i[1]),i[2],(0,255,0),2)
				cv2.circle(imgColor,(i[0],i[1]),2,(0,0,255),3)
		# 映射，切割
		# transPs = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype = np.float32)
		# transform = cv2.getPerspectiveTransform(corners, transPs)
		# splitArea = cv2.warpPerspective(src = colorImg, M = transform, dsize =  (dw, dh))
		cH, cW, _ = colorImg.shape
		transPs = np.array([[0, 0], [cW, 0], [cW, cH], [0, cH]], dtype = np.float32)
		transform = cv2.getPerspectiveTransform(corners / resizeScale, transPs)
		splitArea = cv2.warpPerspective(src = colorImg, M = transform, dsize =  (cW, cH))

		blockListImg.append(splitArea)
	if showImg:
		# showImgs(img, imgColor02, imgColor, *blockListImg)
		showImgs(colorImg, img, imgColor02, imgColor)
	# resource competition
	if records and correctCircles is not None and correctCircles != []:
		with open("tmp/data/saved.circledata", "a") as cfile:
			cfile.write(json.dumps([[c.tolist() for c in correctCircles], originCircles.tolist()], ensure_ascii=False) + "\n");
	return (correctCircles, blockListImg)




