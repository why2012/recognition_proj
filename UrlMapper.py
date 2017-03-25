# coding: utf-8
from controller.TestController import TestController
from controller.ScantronRecogController import ScantronRecogController
from controller.BarCodeRecogController import BarCodeRecogController
from controller.ScoreMarkingController import ScoreMarkingController
from controller.StudentIdRecogController import StudentIdRecogController

class UrlMapper(object):

	def __init__(self):
		self.mapper = [
			(r"/test", TestController),
			(r"/recog", ScantronRecogController),
			(r"/barcode", BarCodeRecogController),
			(r"/markingScore", ScoreMarkingController),
			(r"/studentIdRecog", StudentIdRecogController)
		]	

	def getMapper(self):
		return self.mapper