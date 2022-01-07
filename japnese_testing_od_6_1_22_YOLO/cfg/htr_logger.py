# -*- coding: utf-8 -*-
"""
@author: RawatTech
"""

import logging
# from cloghandler import ConcurrentRotatingFileHandler    
from logging.handlers import RotatingFileHandler

import os


class Htr_logger(object):
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"
    critical = "critical" 
    logger = '' 

    @staticmethod
    def gethtrLogger():
        if Htr_logger.logger == '':
            # Gets or creates a logger
            Htr_logger.logger = logging.getLogger("Htr_logger")  
            # set log level
            Htr_logger.logger.setLevel(logging.DEBUG)
            file = os.path.abspath(os.path.dirname(__file__))+'/../logs//htrservices.log'
            #file_handler = logging.FileHandler(file)
            file_handler = RotatingFileHandler(file, "a", 30*1024*1024, 10)
            formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
            file_handler.setFormatter(formatter)
            # add file handler to logger
            Htr_logger.logger.addHandler(file_handler)
        
        return Htr_logger.logger



    @staticmethod
    def log(log_level, log_msg):
        logger = Htr_logger.gethtrLogger() if Htr_logger.logger == '' else Htr_logger.logger 
        if log_level=="debug":
                logger.debug(log_msg)
        elif log_level=="info":
                logger.info(log_msg)
        elif log_level=="warning":
                logger.warning(log_msg)
        elif log_level=="error":
                logger.error(log_msg)
        elif log_level=="critical":
                logger.critical(log_msg)












     
