# coding: utf-8
from ScantronRecogController import *

class ScoreMarkingController(ScantronRecogController):
	def execute(self):
		self.checkParams02()
		ScantronRecogController.execute(self)

	# standAns:A,B|C|D|A, direction:horizontal: 1|vertical: 2, type: choice: 2|multiChoice: 3, 
	def marking(self, mat, standAns, quesType, direction, totalScore):
		return [0, 1, 2, 3], "A,B|C|D|-"

	def checkParams02():
		standAns = self.getArg("standAns", None)
		quesType = self.getArg("quesType", None)
		direction = self.getArg("direction", None)
		totalScore = self.getArg("totalScore", None)
		if standAns is None:
			raise ErrorStatusException("standAns must be string indicate standard answer", STATUS_PARAM_ERROR)
		if quesType is None:
			raise ErrorStatusException("quesType must be number indicate question type(choice|multichoice)", STATUS_PARAM_ERROR)
		if direction is None:
			raise ErrorStatusException("direction must be number indicate question direction", STATUS_PARAM_ERROR)
		if totalScore is None:
			raise ErrorStatusException("totalScore must be number", STATUS_PARAM_ERROR)
		self.standAns = standAns
		self.quesType = quesType
		self.direction = direction
		self.totalScore = totalScore

