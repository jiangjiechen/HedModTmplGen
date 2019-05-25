# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging, os, cjjpy as cjj
from coloredlogs import ColoredFormatter

logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET, from_scratch=False):
	fmt = "[%(asctime)s %(levelname)s] %(message)s"
	log_format = ColoredFormatter(fmt=fmt)
	# log_format = logging.Formatter()
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)

	console_handler = logging.StreamHandler()
	console_handler.setFormatter(log_format)
	logger.handlers = [console_handler]

	if log_file and log_file != '':
		if from_scratch and os.path.exists(log_file):
			logger.warning('Removing previous log file: %s' % log_file)
			os.remove(log_file)
		path = os.path.dirname(log_file)
		cjj.MakeDir(path)
		file_handler = logging.FileHandler(log_file)
		file_handler.setLevel(log_file_level)
		file_handler.setFormatter(log_format)
		logger.addHandler(file_handler)

	return logger
