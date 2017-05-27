#coding:utf-8
import cv2
import numpy as np

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

def determineBoxRatioMobile(c1, c2, c3, c4, whRatio, thresh = 0.11):
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
	if whRatioBool and diagLengthRatioBool:
		# 差异度
		difference = topLeft[0] + topLeft[1] + radiusVar  + 0.7 * (topRight[0] + topRight[1]) + 5 * np.abs((topRight[1] - topLeft[1])) + np.abs((topRight[0] - bottomLeft[0]))
		# 宽高比与对角线长度
		# print whRatioBool, diagLengthRatioBool, staRatio
		# print difference, topLeft[0] + topLeft[1], radiusVar, 0.7 * (topRight[0] + topRight[1]), 5 * np.abs((topRight[1] - topLeft[1])), np.abs((topRight[0] - bottomLeft[0]))
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
	img = dilation(img, kernel = getKernel((10, 10)))
	img = erosion(img, iterations = 4)
	# cv2.imwrite("resources/test.jpg", img)
	# 切割结果
	splitArea = np.array([])
	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minWH * 0.08, param1 = 16, param2 = 8, minRadius = int(np.ceil(minWH * 0.005)), maxRadius = int(np.ceil(minWH * 0.020)))
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

def circleSplitMobile(originalImg, paperW, paperH, colorImg, scaleThresh = 1.0, showImg = False):
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
		transPs = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype = np.float32)
		transform = cv2.getPerspectiveTransform(corners, transPs)
		splitArea = cv2.warpPerspective(src = colorImg, M = transform, dsize =  (dw, dh))
		blockListImg.append(splitArea)
	if showImg:
		# showImgs(img, imgColor02, imgColor, *blockListImg)
		showImgs(colorImg, img, imgColor02, imgColor)
	return (correctCircles, blockListImg)

