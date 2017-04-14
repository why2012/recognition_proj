# coding: utf-8
from controller.TestController import TestController
from controller.ScantronRecogController import ScantronRecogController
from controller.BarCodeRecogController import BarCodeRecogController
from controller.ScoreMarkingController import ScoreMarkingController
from controller.StudentIdRecogController import StudentIdRecogController
from controller.HoughCircleSplitController import HoughCircleSplitController
from controller.SiftMatchController import SiftMatchController
from controller.CustomMarkingController import CustomMarkingController

class UrlMapper(object):

	def __init__(self):
		self.mapper = [
			(r"/test", TestController),
			(r"/recog", ScantronRecogController),
			(r"/barcode", BarCodeRecogController),
			(r"/markingScore", ScoreMarkingController),
			(r"/studentIdRecog", StudentIdRecogController),
			(r"/paperSplit", HoughCircleSplitController),
			(r"/paperMatch", SiftMatchController),
			(r"/lineMarking", CustomMarkingController)
		]	

	def getMapper(self):
		return self.mapper