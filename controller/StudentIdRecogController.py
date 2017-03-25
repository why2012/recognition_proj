# coding: utf-8
from BarCodeRecogController import *
from ScantronRecogController import *

class StudentIdRecogController(BarCodeRecogController, ScantronRecogController):
	def execute(self):
		self.checkParam02()
		if self.idType == 1:
			self.setArg("barcode", self.getArg("code"))
			BarCodeRecogController.execute(self)
		if self.idType == 2:
			self.setArg("carde", self.getArg("code"))
			ScantronRecogController.execute(self)

	# type: 1条码|2填涂
	def checkParam02(self):
		idType = self.getArg("type", None)
		if idType is None:
			raise ErrorStatusException("type must be number (1条码|2填涂)", STATUS_PARAM_ERROR)
		if idType != 1 and idType != 2:
			raise ErrorStatusException("type must be number (1条码|2填涂)", STATUS_PARAM_ERROR)
		self.idType = idType

