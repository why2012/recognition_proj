# coding: utf-8
import tornado.web as web
import json
from util.ErrorCode import *

class BaseController(web.RequestHandler):
	def post(self):
		self.invokeExecute()

	def get(self):
		self.invokeExecute()

	def invokeExecute(self):
		try:
			self.execute()
		except ErrorStatusException, e:
			self.setResult(status = e.getCode(), msg = e.getMsg())
		except Exception, e:
			self.setResult(status = STATUS_SCAN_ERROR, msg = "Internal Error: " + repr(type(e)) + ", " + str(e))
		finally:
			self.jsonWrite(self.result)

	def execute(self):
		pass

	def jsonWrite(self, data):
		self.write(json.dumps(data))

	def jsonDump(self, data):
		return json.dumps(data)

	def jsonLoad(self, data):
		return json.loads(data)

	def setResult(self, ans = [], status = "", msg = ""):
		self.result = {"status": status, "ans": ans, "msg": msg}

	def fileExist(self, name = "file"):
		if name in self.request.files:
			return True
		else: 
			return False

	def processUpFile(self, name = "file", raiseException = True):
		if name in self.request.files:
			fileMetas = self.request.files[name]
			if fileMetas:
				for meta in fileMetas:
					return meta['body']
		elif raiseException:
			raise ErrorStatusException(name + " must not be None", STATUS_PARAM_ERROR)

	def getIntArg(self, key, default = -1):
		return int(self.get_argument(key, default))

	def getStrArg(self, key, default = ""):
		return self.get_argument(key, default)

class ErrorStatusException(Exception):
	def __init__(self, errMsg, errCode):
		super(ErrorStatusException, self).__init__(errMsg, errCode)
		self.errMsg = errMsg
		self.errCode = errCode

	def getMsg(self):
		return self.errMsg

	def getCode(self):
		return self.errCode

