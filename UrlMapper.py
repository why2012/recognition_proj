# coding: utf-8
from controller.TestController import TestController
from controller.ScantronRecogController import ScantronRecogController
from controller.BarCodeRecogController import BarCodeRecogController

class UrlMapper(object):

	def __init__(self):
		self.mapper = [
			(r"/test", TestController),
			(r"/recog", ScantronRecogController),
			(r"/barcode", BarCodeRecogController)
		]	

	def getMapper(self):
		return self.mapper