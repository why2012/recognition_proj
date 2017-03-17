# coding: utf-8
import tornado.ioloop as ioloop
import tornado.web as web 
import logging.config
from UrlMapper import *
import Setting

class MakeApp(object):

	def __init__(self):
		self.urlMapper = UrlMapper()

	def make(self):
		logging.config.fileConfig("conf/Logging.conf")
		return web.Application(self.urlMapper.getMapper(), **Setting.Conf)

if __name__ == "__main__":
	app = MakeApp().make()
	app.listen(20001)
	ioloop.IOLoop.current().start()