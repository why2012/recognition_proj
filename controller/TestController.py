# coding: utf-8
from BaseController import *

class TestController(BaseController):
	def get(self):
		self.jsonWrite({"data": [[1,2,3], [4,5,6]]})
