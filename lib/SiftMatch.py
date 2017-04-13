#coding:utf-8
import cv2
import numpy as np

def filter_matches(kp1, kp2, matches, ratio = 0.75):  
	mkp1, mkp2 = [], []  
	for m in matches:  
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:  
			m = m[0]  
			mkp1.append(kp1[m.queryIdx])  
			mkp2.append(kp2[m.trainIdx])  
	p1 = np.float32([kp.pt for kp in mkp1])  
	p2 = np.float32([kp.pt for kp in mkp2])  
	kp_pairs = zip(mkp1, mkp2)  
	return p1, p2, kp_pairs  

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):  
	h1, w1 = img1.shape[:2]  
	h2, w2 = img2.shape[:2]  
	vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)  
	vis[:h1, :w1] = img1  
	vis[:h2, w1:w1 + w2] = img2  
	vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)  
	  
	if H is not None:  
		corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])  
		corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))  
		cv2.polylines(vis, [corners], True, (255, 255, 255))  
  	
	if status is None:  
		status = np.ones(len(kp_pairs), np.bool)  
	p1 = np.int32([kpp[0].pt for kpp in kp_pairs])  
	p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)
    
	green = (0, 255, 0)  
	red = (0, 0, 255)  
	white = (255, 255, 255)  
	kp_color = (51, 103, 236)  
	for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):  
		if inlier:  
			col = green  
			cv2.circle(vis, (x1, y1), 3, col, -1)  
			cv2.circle(vis, (x2, y2), 3, col, -1)  
		else:  
			col = red  
			r = 2  
			thickness = 3  
			cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)  
			cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)  
			cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)  
			cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)  
	vis0 = vis.copy()  
	for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):  
		if inlier:  
			cv2.line(vis, (x1, y1), (x2, y2), red)  
  
	cv2.imshow(win, vis)

# 包围框极大值抑制, 如果重叠就进行抑制
def NMS(boundingBox, boundingBoxAttr):
	newBoundingBox = []
	boundingBoxGroup = []
	boundingBoxIter = []
	for boundingBoxIndex, boundingBoxItem in enumerate(boundingBox):
		# 添加进组里的不在重复处理
		if boundingBoxIndex in boundingBoxIter:
			continue
		topLeft, topRight, bottomRight, bottomLeft = boundingBoxItem
		attr = boundingBoxAttr[boundingBoxIndex]
		boundingBoxGroupItem = []
		boundingBoxGroupItem.append([boundingBoxItem, attr])
		boundingBoxIter.append(boundingBoxIndex)
		for boundingBoxInnerIndex in range(boundingBoxIndex + 1, len(boundingBox)): 
			boundingBoxInnerItem = boundingBox[boundingBoxInnerIndex]
			if boundingBoxInnerIndex in boundingBoxIter:
				continue
			itopLeft, itopRight, ibottomRight, ibottomLeft = boundingBoxInnerItem
			iattr = boundingBoxAttr[boundingBoxInnerIndex]
			# if overlap, 添加进同一个组
			if not (topRight[0] < itopLeft[0] or bottomLeft[1] < itopLeft[1] or topLeft[0] > itopRight[0] or topRight[1] > ibottomRight[1]):
				boundingBoxGroupItem.append([boundingBoxInnerItem, iattr])
				boundingBoxIter.append(boundingBoxInnerIndex)
		boundingBoxGroup.append(boundingBoxGroupItem)
	for boundingBoxGroupItem in boundingBoxGroup:
		# 选择匹配上的特征点最多的
		newBoundingBox.append(max(boundingBoxGroupItem, key = lambda i: i[1])[0])
	return newBoundingBox

# imgFeature = cv2.imread("pics/IMG_1203.JPG")
# imgDest = cv2.imread("pics/IMG_1205_SRC.JPG")
# imgDest2 = cv2.imread("pics/IMG_1205_SRC2.JPG")
# imgDest3 = cv2.imread("pics/IMG_1205_SRC3.JPG") 
# matchResult=[];reload(siftM);siftM.siftTest(imgFeature, imgDest, matchResult)
def siftTest(imgFeature, imgDest, matchResult = [], drawBoundingBox = True):
	imgFeatureGray = cv2.cvtColor(imgFeature, cv2.COLOR_BGR2GRAY)
	imgDestGray = cv2.cvtColor(imgDest, cv2.COLOR_BGR2GRAY)
	if not matchResult:
		print "Detect..."
		sift = cv2.xfeatures2d.SIFT_create()
		(kpsFeatureImg, descsFeatureImg) = sift.detectAndCompute(imgFeature, None)
		(kpsDestImg, descsDestImg) = sift.detectAndCompute(imgDest, None)
		bf = cv2.BFMatcher(cv2.NORM_L2)
		matches = bf.knnMatch(descsFeatureImg, descsDestImg, k = 2)
		matchResult.extend([kpsFeatureImg, kpsDestImg, matches])
	else:
		kpsFeatureImg = matchResult[0]
		kpsDestImg = matchResult[1]
		matches = matchResult[2]
	print "Match..."
	p1, p2, kpPairs = filter_matches(kpsFeatureImg, kpsDestImg, matches, ratio = 0.5) # ratio = 0.5
	print "MatchResult: ", len(kpPairs)
	if kpPairs:
		if drawBoundingBox:
			M, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
			imgFeatureH, imgFeatureW = imgFeatureGray.shape
			pts = np.float32([[0, 0], [imgFeatureW - 1, 0], [imgFeatureW - 1, imgFeatureH - 1], [0, imgFeatureH - 1]])
			pts = pts.reshape(-1, 1, 2);
			dst = cv2.perspectiveTransform(pts, M)
			print dst
			imgBoundingBox = cv2.polylines(imgDest.copy(), [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
			cv2.imshow("bounding", imgBoundingBox)
		explore_match('matches', imgFeatureGray, imgDestGray, kpPairs) 
		cv2.waitKey(10)

# service
# 滑动窗口匹配，从上往下, 窗口高度默认10%, 每个窗口配对一个子窗口
# 默认进行下采样
# boudingBox = siftM.siftMatchVertical(imgFeature, imgDest, 0.05)
def siftMatchVertical(imgFeature, imgDest, windowHeightRate = 0.1, showImg = False, pyrDown = True):
	# imgFeature = cv2.pyrDown(imgFeature)
	# 下采样
	imgDestHOrigin, imgDestWOrigin, _ = imgDest.shape
	if pyrDown:
		imgDest = cv2.pyrDown(imgDest, dstsize = (imgDestWOrigin / 2, imgDestHOrigin / 2))
	# imgDestHOrigin, imgDestWOrigin, _ = imgDest.shape
	# imgDest = cv2.pyrDown(imgDest, dstsize = (imgDestWOrigin / 2, imgDestHOrigin / 2))

	imgFeatureGray = cv2.cvtColor(imgFeature, cv2.COLOR_BGR2GRAY)
	imgDestGray = cv2.cvtColor(imgDest, cv2.COLOR_BGR2GRAY)
	imgFeatureH, imgFeatureW = imgFeatureGray.shape
	imgDestH, imgDestW = imgDestGray.shape
	featurePts = np.float32([[0, 0], [imgFeatureW - 1, 0], [imgFeatureW - 1, imgFeatureH - 1], [0, imgFeatureH - 1]])
	featurePts = featurePts.reshape(-1, 1, 2);
	# 窗口高度
	windowHeight = int(imgDestH * windowHeightRate)
	windowRange = range(0, imgDestH, int(windowHeight))
	windowRangeExtend = []
	# 生成子窗口
	for windowYPos in windowRange:
		extendH = windowYPos + windowHeight / 2.0
		if extendH <= imgDestH - windowHeight:
			windowRangeExtend.append([windowYPos, extendH])
	windowRange = windowRangeExtend
	windowRange = np.int32(windowRange)
	sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers = 5)
	# surf = cv2.xfeatures2d.SURF_create();
	method = sift
	bf = cv2.BFMatcher(cv2.NORM_L2)
	# 提取模板特征
	(kpsFeatureImg, descsFeatureImg) = method.detectAndCompute(imgFeature, None)
	# print kpsFeatureImg
	boundingBox = []
	boundingBoxAttr = []
	# 滑动窗口
	for windowYPosList in windowRange:
		# 子窗口中最好的匹配结果
		p1 = None
		p2 = None
		kpPairs = []
		windowYPos = -1
		windowImg = None
		# 滑动子窗口
		# 选择最好的子窗口匹配结果
		for windowYPosItem in windowYPosList:
			windowImgItem = imgDest[windowYPosItem: windowYPosItem + windowHeight, 0: imgDestW];
			# cv2.imshow("img" + str(windowYPosItem), windowImgItem)
			# 提取目标特征
			(kpsDestImg, descsDestImg) = method.detectAndCompute(windowImgItem, None)
			# 选取最好的匹配结果
			if kpsDestImg != [] and descsDestImg is not None:
				# 特征匹配
				matches = bf.knnMatch(descsFeatureImg, descsDestImg, k = 2)
				# 过滤匹配结果
				p1Item, p2Item, kpPairsItem = filter_matches(kpsFeatureImg, kpsDestImg, matches, ratio = 0.75)
				# 至少匹配到10个特征点
				if kpPairsItem and len(kpPairsItem) >= 10:
					# 选择最好的子窗口匹配结果
					if len(kpPairsItem) > len(kpPairs):
						p1 = p1Item
						p2 = p2Item
						kpPairs = kpPairsItem
						windowYPos = windowYPosItem
						windowImg = windowImgItem
		if p1 is not None and p2 is not None and kpPairs != [] and windowYPos != -1:
			# 求出映射矩阵
			M, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
			if M is not None:
				# 执行映射
				dst = cv2.perspectiveTransform(featurePts, M)
				dst = dst[:, 0]
				dst[:,1] += windowYPos
				boundingBox.append(dst)
				boundingBoxAttr.append(len(kpPairs))
			# if showImg:
			# 	explore_match('matches' + str(windowYPos), imgFeatureGray, cv2.cvtColor(windowImg, cv2.COLOR_BGR2GRAY), kpPairs) 
	# 极大值抑制，消除重叠包围框
	boundingBox = NMS(boundingBox, boundingBoxAttr)
	if showImg:
		imgBoundingBox = cv2.polylines(imgDest.copy(), np.int32(boundingBox), True, (0, 255, 0), 3, cv2.LINE_AA)
		cv2.imshow("img", imgBoundingBox)
		cv2.waitKey(10)
	if pyrDown:
		boundingBox = np.array(boundingBox) * 2
	return boundingBox





