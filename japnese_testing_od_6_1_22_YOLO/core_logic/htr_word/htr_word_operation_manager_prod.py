import os
import sys
# from core_logic.htr_word.preprocess.language.eng.english_od import english_preprocess
from core_logic.htr_word.preprocess.language.jpn.japanese_od import japanese_single_char_preprocess, japanese_multi_char_preprocess
from core_logic.htr_word.preprocess.language.math.math_od import math_single_char_preprocess, math_multi_char_preprocess
from core_logic.htr_word.common.white_space_removal import white_space_remove
from core_logic.htr_word.preprocess.language.eng.eng_white_space_removal import eng_white_space_remove
from core_logic.htr_word.preprocess.language.math.segmentation_math import Segmentation_Math
from core_logic.htr_word.preprocess.language.jpn.segmentation_japnese import Segmentation_Japanese
# from core_logic.htr_word.preprocess.language.eng.segmentation_english import Segmentation_english 
from core_logic.htr_word.recognition.misc.unit.htr_unit import unit_recog
from core_logic.htr_word.operation.eng.operation_eng import process_english
from core_logic.htr_word.operation.jpn.operation_jpn import process_japnese,special_char_jpn
from core_logic.htr_word.operation.num.operation_num import process_maths,special_char_num
from core_logic.htr_word.preprocess.language.jpn.japnese_segmentation_by_od import segmentation_by_Od
from core_logic.htr_word.preprocess.language.jpn.jap_white_space_removal import jpn_white_space_remove
from core_logic.htr_word.recognition.misc.jpn.utils.similar import filter_block
from core_logic.htr_word.preprocess.language.eng.english_od import english_segmentation
from cfg.htr_logger import Htr_logger
from core_logic.htr_word.common.blankdetection import blank_detection
import numpy as np
import base64
from core_logic.htr_lib.perfmeter import Timer
import cv2
import silence_tensorflow.auto
import warnings
warnings.filterwarnings('ignore')
# #import matplotlib.pyplot as plt

timer= Timer()
# HORIZONTAL = "H"
# VERTICAL = "V"

#this is our main function

def recognition_process(operation,test_img,Is_multi_char_type = False, reading_direction = "HORIZONTAL",expected_len=0,uuid=None):
	try:
		operation = operation.upper()

		Htr_logger.log(Htr_logger.info, "[{}] : recognition_process : process start 1".format(uuid))
		byte2image = base64.b64decode(test_img)
		test_img = np.fromstring(byte2image, dtype=np.uint8)
		test_img = cv2.imdecode(test_img,cv2.IMREAD_COLOR)
		Htr_logger.log(Htr_logger.info, "[{}] : recognition_process : process start 2".format(uuid))
				
		#condition for english1
		
		if operation=="E":

			timer.startOp()
			
			#for pre processign (Object detection)
			text_image_list = english_segmentation(test_img,expected_len)
			#for blank detection
			# img = blank_detection(img,"E")
			#for segmentation
			# img_list = Segmentation_english(img)
			#for removing white spaces
			text_image_list = eng_white_space_remove(text_image_list)

			Recognized_text_string, confidence_score_list = process_english(text_image_list)
			
			Htr_logger.log(Htr_logger.debug, "Recognition :English : recognition_time: {}".format(timer.endOp()))
			confidence_score_list = np.mean(confidence_score_list)
			Htr_logger.log(Htr_logger.info, "Recognition : English : Result : {} : Confidence score : {}".format(Recognized_text_string, confidence_score_list))

			# return Recognized_text_string, confidence_score_list

		#condition for math
		elif operation=="M":

			timer.startOp()

			segments = []
			if Is_multi_char_type:
				text_image_list, start_x_list_text, sp_list, start_x_list_sp, Sp_char_position_list ,unit= math_multi_char_preprocess(test_img)
			else:
				img, sp_list, start_x_list_sp, Sp_char_position_list ,unit = math_single_char_preprocess(test_img)
				img = blank_detection(img,"M")
				text_image_list, start_x_list_text= Segmentation_Math(img)
				text_image_list,start_x_list_text = white_space_remove(text_image_list,start_x_list_text)

			sp_dict,sp_confidence_dict = special_char_num(sp_list,Sp_char_position_list,start_x_list_sp)

			Recognized_text_string, confidence_score_list = process_maths(text_image_list,start_x_list_text, sp_dict, sp_confidence_dict,Sp_char_position_list)

			if type(unit).__name__ != "str" and type(unit) != "NoneType":

				unit,confidence_score = unit_recog(unit)
				Recognized_text_string = Recognized_text_string+" "+unit
				confidence_score_list.append(confidence_score)

			confidence_score_list = np.mean(confidence_score_list)

			# return Recognized_text_string, confidence_score_list
			Htr_logger.log(Htr_logger.debug, "Recognition :Math : recognition_time: {}".format(timer.endOp()))
			Htr_logger.log(Htr_logger.info, "Recognition : Math : Result : {} : Confidence score : {}".format(Recognized_text_string, confidence_score_list))		#condtion for japanese
		
		#condition for japanese
		elif operation=="J":
			timer.startOp()

			# sp_list = []
			# Sp_char_position_list = []
			# start_x_list_sp = []
			if Is_multi_char_type:
				text_image_list, start_x_list_text, sp_list, start_x_list_sp, Sp_char_position_list= japanese_multi_char_preprocess(test_img)
			else:
				img, sp_list, start_x_list_sp, Sp_char_position_list = japanese_single_char_preprocess(test_img)


			# img = blank_detection(img,"J")

			# text_image_list, start_x_list_text= Segmentation_Japanese(img,reading_direction)
			# print("----------------------",expecetd_len)
			text_image_list, start_x_list_text = segmentation_by_Od(img,expected_len,reading_direction)
			# print("2")
			text_image_list, start_x_list_text,height,area = jpn_white_space_remove(text_image_list, start_x_list_text)
			# print("3")
			
			sp_dict,sp_confidence_dict = special_char_jpn(sp_list,Sp_char_position_list,start_x_list_sp)
			# print("4")
			Recognized_text_string, confidence_score_list = process_japnese(text_image_list,start_x_list_text, sp_dict, sp_confidence_dict, Sp_char_position_list,reading_direction,height,area)
			# Recognized_text_string=check_small(Recognized_text_string,height,area,flag)
			Recognized_text_string=filter_block(Recognized_text_string)
			confidence_score_list = np.mean(confidence_score_list)

			Htr_logger.log(Htr_logger.debug, "Recognition :Japanese : recognition_time: {}".format(timer.endOp()))
			Htr_logger.log(Htr_logger.info, "Recognition : Japanese : Result : {} : Confidence score : {}".format(Recognized_text_string, confidence_score_list))


		return Recognized_text_string, confidence_score_list,text_image_list
			

	except Exception as e:
		Htr_logger.log(Htr_logger.error, "Recognition : failed : {}".format(str(e)))
		timer.endOp()
		raise e

