import os
import sys
from core_logic.htr_singlechar.pre_processing.preprocessing_util import  preprocess
from core_logic.htr_singlechar.recognition.languages.math.htr_num import math_recog
from core_logic.htr_singlechar.recognition.languages.eng.htr_english import english_recog 
from core_logic.htr_singlechar.recognition.languages.jpn.htr_jap import jap_recog
from cfg.htr_logger import Htr_logger
from core_logic.htr_singlechar.pre_processing.blankdetection import blank_detection
import numpy as np
import base64
from core_logic.htr_lib.perfmeter import Timer
import cv2
timer= Timer()
#gateway for HTR process
def recognition_process(operation,test_img,image_name):
	try:
		operation = operation.upper()
		# convert base64 image string to numpy array
		Htr_logger.log(Htr_logger.info, "Recognition : process start")
		byte2image = base64.b64decode(test_img)
		test_img = np.fromstring(byte2image, dtype=np.uint8)
		test_img = cv2.imdecode(test_img,cv2.IMREAD_COLOR)
		
		#Object detection module
		timer.startOp()
		text = ""
		sp_image = []
		segments = []
		img= preprocess(test_img)

		Htr_logger.log(Htr_logger.debug, "Recognition :over all preprocess_time : {}".format(timer.endOp()))
		timer.startOp()
		#blank detection module 

		img = blank_detection(img,operation)
		cv2.imwrite("od_output/{}".format(image_name),img)
		Od_output="od_output/{}".format(image_name)

		Htr_logger.log(Htr_logger.debug, "Recognition :blank_detection_time : {}".format(timer.endOp()))
		

		Htr_logger.log(Htr_logger.info, "Recognition : preprocessing done")
		
		#condition for english
		
		if operation=="E":
			timer.startOp()
			result, confidence_score = english_recog(img)
			Htr_logger.log(Htr_logger.debug, "Recognition :English : recognition_time: {}".format(timer.endOp()))
			Htr_logger.log(Htr_logger.info, "Recognition : English : Result : {} : Confidence score : {}".format(result,confidence_score))
		
		#condition for math
		elif operation=="M":
			timer.startOp()
			result, confidence_score = math_recog(img)
			Htr_logger.log(Htr_logger.debug, "Recognition :Math : recognition_time: {}".format(timer.endOp()))
			Htr_logger.log(Htr_logger.info, "Recognition : Math : Result : {} : Confidence score : {}".format(result,confidence_score))		#condtion for japanese
		
		#condition for japanese
		elif operation=="J":
			timer.startOp()
			result, confidence_score = jap_recog(img)
			Htr_logger.log(Htr_logger.debug, "Recognition :Japanese : recognition_time: {}".format(timer.endOp()))
			Htr_logger.log(Htr_logger.info, "Recognition : Japanese : Result : {} : Confidence score : {}".format(result,confidence_score))		
			
		return result, confidence_score,text,sp_image, segments,Od_output

	except Exception as e:
		timer.endOp()
		Htr_logger.log(Htr_logger.error, "Recognition : failed : {}".format(str(e)))
		raise e

