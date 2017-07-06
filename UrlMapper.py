# coding: utf-8
from controller.TestController import TestController
from controller.ScantronRecogController import ScantronRecogController
from controller.BarCodeRecogController import BarCodeRecogController
from controller.ScoreMarkingController import ScoreMarkingController
from controller.StudentIdRecogController import StudentIdRecogController
from controller.HoughCircleSplitController import HoughCircleSplitController
from controller.SiftMatchController import SiftMatchController
from controller.CustomMarkingController import CustomMarkingController
from controller.QuestionAreaMatchController import QuestionAreaMatchController
from controller.MultiTypeScoreMarkController import MultiTypeScoreMarkController
from controller.AreaExceptionDetectController import AreaExceptionDetectController
from controller.HoughCircleSplitWithRotateController import HoughCircleSplitWithRotateController
from controller.ControllPanelController import *

class UrlMapper(object):

	def __init__(self):
		self.mapper = [
			(r"/test", TestController),
			(r"/recog", ScantronRecogController),
			(r"/barcode", BarCodeRecogController),
			(r"/markingScore", ScoreMarkingController),
			(r"/studentIdRecog", StudentIdRecogController),
			(r"/paperSplit", HoughCircleSplitController),
			(r"/paperSplitWithRotate", HoughCircleSplitWithRotateController),
			(r"/paperMatch", SiftMatchController),
			(r"/lineMarking", CustomMarkingController),
			(r"/questionAreaMatch", QuestionAreaMatchController),
			(r"/multiTypeMark", MultiTypeScoreMarkController),
			(r"/areaExceptionDetect", AreaExceptionDetectController),

			(r"/panelsrc/(.*)", FileAccessController),# 获取资源
			(r"/panel/nextimg/(.*)/(.*)", NextImage),# basedir, bucket 下一张图片信息
			(r"/panel/showimg/(.*)/(.*)", ShowImage),# imgpath, imgid 下一张图片数据
			(r"/panel/submitsample/(.*)/(.*)", SubmitSample),# imgid, cindex 提交图片角点编号
			(r"/panel/trainmodel/(.*)", TrainModel),# bucket
			(r"/panel/testmodel/(.*)", TestModel),# bucket
			(r"/panel/bucketlist", BucketList),
			(r"/panel/applymodel", ApplyModel),# 应用训练好的数据
			(r"/panel/switchtraditional/(.*)", SwitchTraditional),# 切换传统方式和机器学习方式
		]	

	def getMapper(self):
		return self.mapper