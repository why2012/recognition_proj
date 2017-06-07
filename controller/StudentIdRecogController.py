# coding: utf-8
from BarCodeRecogController import *
from ScantronRecogController import *
import numpy as np

class StudentIdRecogController(BarCodeRecogController, ScantronRecogController):
	def execute(self):
		StudentIdRecogController.checkParam02(self)
		if self.idType == 1:
			self.changeArgName("barcode", "code")
			BarCodeRecogController.execute(self)
		if self.idType == 2:
			self.changeArgName("card", "code")
			if self.isMobile == 1:
				baseYBias = True
			else:
				baseYBias = False
			ScantronRecogController.execute(self, baseYBias = baseYBias)
			self.determingId(self.getResult()["ans"])

	def determingId(self, idMat):
		idMat = np.array(idMat)
		stuId = [0] * idMat.shape[1]
		for index, _ in enumerate(stuId):
			colData = idMat[:, index]
			colData = np.nonzero(colData == 1)[0]
			if len(colData) == 0:
				idNum = 0
			else:
				idNum = colData[0]
			stuId[index] = idNum
		self.setResult(''.join([str(i) for i in stuId]), STATUS_OK)

	# type: 1条码|2填涂
	@staticmethod
	def checkParam02(self):
		idType = self.getIntArg("type", None)
		self.isMobile = self.getIntArg("isMobile", -1)
		if idType is None:
			raise ErrorStatusException("type must be number (1条码|2填涂)", STATUS_PARAM_ERROR)
		if idType != 1 and idType != 2:
			raise ErrorStatusException("type must be number (1条码|2填涂)", STATUS_PARAM_ERROR)
		self.idType = idType

