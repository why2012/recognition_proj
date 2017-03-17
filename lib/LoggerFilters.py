# coding: utf-8
import logging 

class WarngingFilter(logging.Filter):
	def filter(self, record):
		return record.levelno == logging.WARNING

class ErrorFilter(logging.Filter):
	def filter(self, record):
		return record.levelno == logging.ERROR

class InfoDebugFilter(logging.Filter):
	def filter(self, record):
		return (record.levelno == logging.INFO or record.levelno == logging.DEBUG)