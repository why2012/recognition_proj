#coding:utf-8
import cv2
import numpy as np

# 图片宽度对应数组列数，高度对应数组行数
def getImgSize(img):
	return (img.shape[1], img.shape[0])

def grayImg(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def showImg(*imgs):
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
	if whRatioBool and diagLengthRatioBool:
		# 差异度
		difference = np.abs(diagLengthRatio - thresh) + radiusVar
		return True, (topLeft, topRight, bottomRight, bottomLeft), getSkewScale(topLeft, topRight, bottomRight, bottomLeft), difference
	else:
		return False, (), [], float('inf')

def determingCorrectCircles(circles, whRatio):
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
					result, corners, skewScale, difference = determineBoxRatio(circleTopLeft, circleTopRight, circleBottomRight, circleBottomLeft, whRatio)
					if result:
						correctResult.append({"diff": difference, "corners": corners, "skewScale": skewScale, \
							"circleTopLeft": circleTopLeft, "circleTopRight": circleTopRight, "circleBottomRight": circleBottomRight, \
							"circleBottomLeft": circleBottomLeft})
	if correctResult:
		# 选取差异度最小的candidate
		correctResult = sorted(correctResult, key = lambda a: a["diff"])
		minDiffResult = correctResult[0]
		return minDiffResult["corners"], (minDiffResult["circleTopLeft"], minDiffResult["circleTopRight"], minDiffResult["circleBottomRight"], minDiffResult["circleBottomLeft"]), minDiffResult["skewScale"]
	return [], (), []


# main function, paperW,paperH: 目标区域宽高(相对比例), blockList = [(0.3,0.3,0.5,0.5), ]左上右下
def circleSplit(originalImg, paperW, paperH, blockList = None, scaleThresh = 0.3):
	imgSize = getImgSize(originalImg)
	w, h = imgSize
	# 按比例缩放
	w, h = (np.int(w * scaleThresh), np.int(h * scaleThresh))
	# 目标区域宽高
	dw, dh = (int(paperW * scaleThresh * 2), int(paperH * scaleThresh * 2))
	minWH = np.min((w, h))
	originalImg = cv2.resize(originalImg, (w, h))
	img = grayImg(originalImg)
	# 切割结果
	splitArea = np.array([])
	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minWH * 0.08, param1 = 60, param2 = 20, minRadius = int(np.ceil(3 * scaleThresh)), maxRadius = int(50 * scaleThresh))
	# 只取半径大于平均值的圆
	avgRadius = np.average(circles[0, :, 2]) * 0.9
	circles = np.array([circles[0, circles[0, :, 2] >= avgRadius]])
	# 确定四个边角圆
	corners, correctCircles, _ = determingCorrectCircles(circles[0], float(paperW) / paperH)
	corners = np.array(corners, dtype = np.float32)
	blockListImg = []
	if correctCircles:
		# 映射，切割
		transPs = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype = np.float32)
		transform = cv2.getPerspectiveTransform(corners, transPs)
		splitArea = cv2.warpPerspective(src = originalImg, M = transform, dsize =  (dw, dh))
		blockListImg.append(splitArea)
	return blockListImg



