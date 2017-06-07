# coding: utf-8
import tornado.httpserver as httpserver
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
	if Setting.Conf["debug"]:
		app.listen(20001)
		ioloop.IOLoop.current().start()
	else:
		httpServer = httpserver.HTTPServer(app)
		httpServer.bind(20001)
		httpServer.start(num_processes = 0) 
		ioloop.IOLoop.current().start()