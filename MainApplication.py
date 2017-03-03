# coding: utf-8
import tornado.ioloop as ioloop
import tornado.web as web 
from UrlMapper import *
import Setting

class MakeApp(object):

	def __init__(self):
		self.urlMapper = UrlMapper()

	def make(self):
		return web.Application(self.urlMapper.getMapper(), **Setting.Conf)

if __name__ == "__main__":
	app = MakeApp().make()
	app.listen(20001)
	ioloop.IOLoop.current().start()