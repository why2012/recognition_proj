#coding:utf-8
#个性标记
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# 图片宽度对应数组列数，高度对应数组行数
def getImgSize(img):
	return (img.shape[1], img.shape[0])

# 根据宽高创建纯白图，宽高对应数组的列数和行数
def createWhiteImg(size):
	return np.uint8(np.ones((size[1], size[0])) * 255)

# lines[ line[ subline[]], ...]
def drawLines(img, lines, color = (0, 0, 0), thickness = 1):
	for line in lines:
		for subline in line:
			cv2.line(img, (subline[0], subline[1]), (subline[2], subline[3]), color, thickness)

def degreeToPolar(degree):
	return degree * (np.pi / 180.0)

# 两条直线交点
# segment==True, 求线段交点, 允许thresh大小的误差
def computeIntersect(a, b, segment = False, thresh = 100):
	x1 = a[0]; y1 = a[1]; x2 = a[2]; y2 = a[3]; x3 = b[0]; y3 = b[1]; x4 = b[2]; y4 = b[3]  
	d = float((x1 - x2) * (y3 - y4)) - (y1 - y2) * (x3 - x4)
	if d != 0: 
		pt = [0, 0]
		pt[0] = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
		pt[1] = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d 
		if segment:
			x = [x1, x2, x3, x4]
			y = [y1, y2, y3, y4]
			x.remove(min(x))
			x.remove(max(x))
			y.remove(min(y))
			y.remove(max(y))
			x.sort()
			y.sort()
			x[0] -= thresh
			x[1] += thresh
			y[0] -= thresh
			y[1] += thresh
			if (pt[0] < x[0] or pt[0] > x[1]) or (pt[1] < y[0] or pt[1] > y[1]):
				pt = [-1, -1]
		return pt
	else:
		return [-1, -1]

# 从极坐标获取线段端点
def getLinesFromPolarCoord(polarLines, thresh = 4000):
	lines = []
	for rhoThetaPack in polarLines:
		subline = []
		for rho, theta in rhoThetaPack:
		    a = np.cos(theta)
		    b = np.sin(theta)
		    x0 = a * rho
		    y0 = b * rho
		    x1 = int(x0 + thresh * (-b))
		    y1 = int(y0 + thresh * (a))
		    x2 = int(x0 - thresh * (-b))
		    y2 = int(y0 - thresh * (a))
		    subline.append([x1, y1, x2, y2])
		lines.append(subline)
	return lines

# 划线中心点识别
# r1 = int(boudingBox[1][0][1])
# r2 = int(boudingBox[1][3][1])
# c1 = int(boudingBox[1][0][0])
# c2 = int(boudingBox[1][1][0])
# cm.lineMarking(imgDest02[r1:r2, c1:c2], True)
def lineMarking(img, drawImg = False):
	imgOrigin = img
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img, (5, 5), 0)
	img = cv2.Canny(img, 10, 50, apertureSize = 3)
	lines = cv2.HoughLines(img, 1, np.pi / 360, 40)
	lines = np.array(lines)
	# 与y轴的张角
	# angle <= 60' & angle > 5'
	# lines: [r, theta]
	lines = lines[(lines[:, :, 1] <= degreeToPolar(70)) & (lines[:, :, 1] >= degreeToPolar(5))]
	# 预处理
	linesTrain = lines - (min(lines[:, 0]), min(lines[:, 1]))
	linesTrain[:, 1] = linesTrain[:, 1] * 100
	# 根据倾角聚类
	db = DBSCAN(eps = 10).fit(linesTrain)
	labelSet = set(db.labels_)
	newLines = []
	# 得到n类数据，每一类取平均值
	for l in labelSet:
		newLines.append([np.average(lines[db.labels_ == l][:, 0]), np.average(lines[db.labels_ == l][:, 1])])
	lines = np.array(newLines)
	# print newLines
	# 极坐标表示直线转成直角坐标表示
	lines = lines.reshape(-1, 1, 2)
	# 需要保证直线端点坐标不超出图片范围
	minLen = min(img.shape)
	maxLen = max(img.shape)
	lines = getLinesFromPolarCoord(lines, maxLen)
	# print lines
	# 计算中心点
	centroid = []
	for lineItem in lines:
		for x1, y1, x2, y2 in lineItem:
			# 求边缘交点，需要保证直线端点坐标不超出图片范围
			x1s, y1s = computeIntersect([x1, y1, x2, y2], [0, 0, maxLen, 0])
			x2s, y2s = computeIntersect([x1, y1, x2, y2], [0, minLen, maxLen, minLen])
			# print x1s, y1s, x2s, y2s
			centroid.append([ (x1s + x2s) / 2.0, (y1s + y2s) / 2.0])
	if drawImg:
		# 调试: 画出延长线
		wimg = createWhiteImg(getImgSize(img))
		drawLines(wimg, lines)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		for x, y in centroid:
			cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), 3)
		cv2.imshow("img01", img)
		cv2.imshow("img02", wimg)
		cv2.waitKey(10)
	return centroid






