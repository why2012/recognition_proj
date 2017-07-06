# coding: utf-8
from BaseController import *
import os
import cv2
from lib.HoughCircleSplit import *
from lib.PreProcessing import *
from lib.imagemagick import detectAndGetImage
import numpy as np
import conf.Config as conf
import pickle
from sklearn.svm import SVC
import sys
import json
import sqlite3

def getConnection(db = "resources/dbdata/data.db"):
	return sqlite3.connect(db)

class FileAccessController(BaseController):
	def execute(self, filepath):
		if not (filepath.startswith("resources")):
			raise ErrorStatusException("Illegal path.", 404)
		if not os.path.exists(filepath):
			raise ErrorStatusException("path does not exist.", 404)
		with open(filepath) as file:
			rawData = file.read()
			#self.set_header('Content-Type', 'image/jpeg')
			self.write(rawData)
			self.flush()

class NextImage(BaseController):
	def createTable(self):
		self.conn = getConnection()
		# status comment "1 没标注, 2 已标注"
		self.conn.execute(""" 
			create table if not exists circles(
				imgname varchar(255) not null,
				bucket varchar(255) not null,
				circle_pair text default -1,
				correct_pair varchar(255) default -1,
				status tinyint default 1,

				primary key(imgname, bucket)
			)
			""")
		self.conn.commit()

	def createBucket(self, basedir, bucket):
		data = self.conn.execute("select * from circles where bucket=? limit 1", (bucket,))
		if data.fetchone() is None:
			images = filter(lambda path: path.lower().endswith(".jpg") or path.lower().endswith(".png"), os.listdir(basedir))
			if images is None or len(images) == 0:
				raise ErrorStatusException("empty directory: " + basedir, 1)
			sqldata = []
			for img in images:
				sqldata.append((img, bucket))
			self.conn.executemany("insert into circles(imgname, bucket) values(?, ?)", sqldata)
			self.conn.commit()

	def execute(self, basedir, bucket):
		if not basedir.endswith("/"):
			basedir = basedir + "/"
		basedir = "resources/tmp/" + basedir
		if not os.path.isdir(basedir):
			raise ErrorStatusException("illegal path: " + basedir, 1)
		self.createTable()
		self.createBucket(basedir, bucket)
		# 选出一张status=1的图片
		singleimg = self.conn.execute("select imgname, bucket from circles where status=1 and bucket=? limit 1", (bucket,))
		singleimg = singleimg.fetchone()
		if self.conn is not None:
			self.conn.close()
		if singleimg is not None:
			imgname = singleimg[0]
			bucketname = singleimg[1]
			imgid = imgname + "+" + bucketname
			imgpath = basedir + imgname
			self.setResult({"imgid": imgid, "imgpath": imgpath.replace("/", "+")}, status = 0)
		else:
			self.setResult({}, STATUS_ENCODE_ERROR)

class ShowImage(BaseController):
	def execute(self, imgpath, imgid):
		imgname, bucketname = imgid.split("+")
		imgpath = imgpath.replace("+", "/")
		img = cv2.imread(imgpath)
		img, _ = detectAndGetImage(img, None, "tmp/image/")
		imgH, imgW, _ = img.shape
		originImg = img
		img = filterBlack(img, [0, 0, 0], [180, 255, 100])
		if imgW >= 2000:
			resizeScale = 0.5		
		else:
			resizeScale = 0.8
		resizeW, resizeH = (int(imgW * resizeScale), int(imgH * resizeScale))
		img = cv2.resize(img, (resizeW, resizeH))
		circles, img = circleSplitMobilePlusCollectData(img, scaleThresh = 1.0, colorImg = originImg, resizeScale = resizeScale, showImg = False)
		if circles is not None:
			circles_json = json.dumps(circles.tolist(), ensure_ascii=False)
			conn = getConnection()
			conn.execute("update circles set circle_pair=? where imgname=? and bucket=?", (circles_json, imgname, bucketname))
			conn.commit()
			conn.close()
		retval, buf = cv2.imencode(".jpg", img)
		if retval:
			if int(self.version[2]) >= 10:
				rawData = buf.tobytes()
			else:
				rawData = buf.tostring()
			self.set_header('Content-Type', 'image/jpeg')
			self.write(rawData)
			self.flush()
		else:
			self.setResult([], STATUS_ENCODE_ERROR)


class SubmitSample(BaseController):
	def execute(self, imgid, cindex):
		imgname, bucketname = imgid.split("+")
		cindex = cindex.replace("，", ",")
		topleft, topright, bottomright, bottomleft = [int(i) for i in cindex.split(",")]
		conn = getConnection()
		conn.execute("update circles set correct_pair=?, status=2 where imgname=? and bucket=?", (cindex, imgname, bucketname))
		conn.commit()
		conn.close()
		self.setResult([], status = 0)

class TrainModel(BaseController):
	@classmethod
	def getcirclepair(self, circles):
		pairs = []
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
						pairs.append([circleTopLeft.tolist(), circleTopRight.tolist(), circleBottomRight.tolist(), circleBottomLeft.tolist()])
		return pairs

	@classmethod
	def compactpair(self, data):
		compactData = []
		for item in data:
			# 忽略半径
			item = (item[0], item[1])
			compactData.extend(item)
		return compactData

	def execute(self, bucket):
		conn = getConnection()
		traindata = conn.execute("select * from circles where bucket=? and status=2", (bucket,))
		traindata = traindata.fetchall()
		conn.close()
		datalist = []
		labellist = []
		weightlist = []
		# 每张图取10个反例，一个正例
		for item in traindata:
			imgname = item[0]
			bucketname = item[1]
			circle_pair = item[2]
			correct_pair = item[3]
			circle_pair = np.array(json.loads(circle_pair))
			correct_index = np.array([int(i) for i in correct_pair.split(",")])
			if correct_pair[0] == -1 or correct_pair[1] == -1 or correct_pair[2] == -1 or correct_pair[3] == -1:
				continue
			# print circle_pair
			# print circle_pair[correct_index]
			# 加入正例
			correctpair = circle_pair[correct_index]
			datalist.append(TrainModel.compactpair(correctpair));labellist.append(1);weightlist.append(5)
			# 四个为一组
			circlepair = TrainModel.getcirclepair(circle_pair)
			# 加入10个反例
			upperBound = 0
			for pair in circlepair:
				if upperBound > 10:
					break
				upperBound += 1
				# print pair
				# print correctpair.tolist()
				if pair == correctpair.tolist():
					continue
				datalist.append(TrainModel.compactpair(pair));labellist.append(0);weightlist.append(1)
		# print datalist
		# print "------------"
		# print labellist
		# print "------------"
		# print weightlist
		# print "------------"
		if len(datalist) > 0 and len(labellist) > 0:
			svmobj = SVC()
			svmobj.probability = True
			svmobj.fit(datalist, labellist, weightlist)
			# 后备数据，待启用
			with open("resources/dbdata/svmdata.svm.sec", "w") as file:
				file.write(pickle.dumps(svmobj))
		self.setResult("Training process is finished.", status = 0)

class TestModel(BaseController):
	def execute(self, bucket):
		conn = getConnection()
		traindata = conn.execute("select * from circles where bucket=? and status=2", (bucket,))
		traindata = traindata.fetchall()
		conn.close()
		datalist = []
		labellist = []
		weightlist = []
		# 每张图取10个反例，一个正例
		for item in traindata:
			imgname = item[0]
			bucketname = item[1]
			circle_pair = item[2]
			correct_pair = item[3]
			circle_pair = np.array(json.loads(circle_pair))
			correct_index = np.array([int(i) for i in correct_pair.split(",")])
			if correct_pair[0] == -1 or correct_pair[1] == -1 or correct_pair[2] == -1 or correct_pair[3] == -1:
				continue
			# print circle_pair
			# print circle_pair[correct_index]
			# 加入正例
			correctpair = circle_pair[correct_index]
			datalist.append(TrainModel.compactpair(correctpair));labellist.append(1);weightlist.append(5)
			# 四个为一组
			circlepair = TrainModel.getcirclepair(circle_pair)
			# 加入10个反例
			upperBound = 0
			for pair in circlepair:
				if upperBound > 10:
					break
				upperBound += 1
				# print pair
				# print correctpair.tolist()
				if pair == correctpair.tolist():
					continue
				datalist.append(TrainModel.compactpair(pair));labellist.append(0);weightlist.append(1)
		if len(datalist) == 0:
			self.setResult("无数据", status = 1)
			return
		# 测试后备数据
		svmobj = None
		with open("resources/dbdata/svmdata.svm.sec") as file:
			svmobj = pickle.loads(file.read())
		testresult = svmobj.predict(datalist)
		#print testresult
		testresult = np.array(testresult)
		realresult = np.array(labellist)
		correctRatio = float(np.sum(testresult == realresult)) / len(realresult)
		self.setResult("Test process is finished. CorrectRatio=%f" % correctRatio, status = 0)


class BucketList(BaseController):
	def execute(self):
		conn = getConnection()
		data = conn.execute("select distinct bucket from circles")
		data = data.fetchall()
		conn.close()
		bucketlist = []
		for item in data:
			bucketlist.append(item[0])
		self.setResult(bucketlist, status = 0)

# 启用训练数据
class ApplyModel(BaseController):
	def execute(self):
		os.system("cp resources/dbdata/svmdata.svm.sec resources/dbdata/svmdata.svm")
		self.setResult("OK", status = 0)

# 切换传统方式和机器学习方式
class SwitchTraditional(BaseController):
	def execute(self, optionint):
		optionint = int(optionint)
		# 机器学习方式
		if optionint == 1:
			os.system("touch resources/dbdata/useml.lock")
		# 传统方式
		else:
			os.system("rm resources/dbdata/useml.lock")
		if os.path.exists("resources/dbdata/svmdata.svm"):
			self.setResult("OK", status = 0)
		else:
			self.setResult("切换成功，但是无训练数据，切换无效", status = 0)







