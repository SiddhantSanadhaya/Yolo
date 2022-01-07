import os
import sys
# from core_logic.htr_word.preprocess.language.eng.english_od import english_preprocess
from core_logic.htr_word.preprocess.language.jpn.japanese_od import japanese_single_char_preprocess, japanese_multi_char_preprocess
from core_logic.htr_word.preprocess.language.jpn.japnese_segmentation_by_od import segmentation_by_object_detection
from core_logic.htr_word.recognition.misc.jpn.utils.similar import filter_block,check_small
# from core_logic.htr_word.preprocess.language.math.math_od import math_single_char_preprocess, math_multi_char_preprocess
# from core_logic.htr_word.common.white_space_removal import white_space_remove
# from core_logic.htr_word.preprocess.language.math.segmentation_math import Segmentation_Math
# from core_logic.htr_word.preprocess.language.jpn.segmentation_japnese import Segmentation_Japanese
# from core_logic.htr_word.preprocess.language.eng.segmentation_english import Segmentation_english 
# from core_logic.htr_word.recognition.misc.unit.htr_unit import unit_recog
# from core_logic.htr_word.operation.eng.operation_eng import process_english
from core_logic.htr_word.operation.jpn.operation_jpn import process_japnese
# from core_logic.htr_word.operation.num.operation_num import process_maths,special_char_num
from core_logic.htr_word.preprocess.language.jpn.jap_white_space_removal import jpn_white_space_remove
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
HORIZONTAL = "H"
VERTICAL = "V"

def adaptive_resizing(img,aspect_ratio=3.15):
    height=img.shape[0]
    width=img.shape[1]

    local_asp=width/height
    if height<69 and width>218:
        height=(width/aspect_ratio)
        width=height*local_asp
    elif width<218 and height>69:
        width=height*aspect_ratio
        height=width/local_asp
    elif width<218 and height<69:
        width=218
        height=69
    #print(width,height)
    return cv2.resize(img,(round(width),round(height)))


# results = []
# confidence_scores = []
#,math_multi_char_preprocess,japanese_multi_char_preprocess, math_single_char_preprocess, japanese_single_char_preprocess


# def process_english(img_list):
# 	try:
# 		results = []
# 		confidence_scores = []
# 		for img in img_list:

# 			result,confidence_score = english_recog(img)
# 			results.append(result)
# 			confidence_scores.append(confidence_score)

# 		result = ''.join(results)

# 		return result, confidence_scores
# 	except Exception as e:
# 		raise e

# def process_maths(img_list,x_start_list_text, sp_dict, sp_confidence_dict,Sp_char_position_list):

# 	try:
# 		for img in img_list:
# 			img= cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# 			result, confidence_score = math_recog(img)
# 			results.append(result)
# 			confidence_scores.append(confidence_score)

# 		math_dict = dict(zip(x_start_list_text,results))
# 		confidence_dict = dict(zip(x_start_list_text,confidence_scores))
		
# 		sp_dict.update(math_dict)

# 		sp_confidence_dict.update(confidence_dict)

# 		sp_dict = OrderedDict(sorted(sp_dict.items()))
# 		sp_confidence_dict = OrderedDict(sorted(sp_confidence_dict.items()))

# 		Recognized_text_string = ''.join(str(val) for key, val in sp_dict.items())

# 		confidence_score_list = sp_confidence_dict.values()
# 		confidence_score_list = list(confidence_score_list)

# 		return Recognized_text_string, confidence_score_list
# 	except Exception as e:
# 		raise e

# def process_japnese(img_list,start_x_list_text, sp_dict, sp_confidence_dict, Sp_char_position_list,image_type):
	
# 	try:

# 		for img in img_list:
		
# 			result, confidence_score = jap_recog(img)
# 			results.append(result)
# 			confidence_scores.append(confidence_score)

# 		jap_dict = dict(zip(start_x_list_text,results))
# 		confidence_dict = dict(zip(start_x_list_text,confidence_scores))
# 		#print(confidence_dict,"confidence_dict")

# 		sp_dict.update(jap_dict)
# 		sp_confidence_dict.update(confidence_dict)

# 		if  Sp_char_position["Mid"] in Sp_char_position_list  or Sp_char_position["End"] in Sp_char_position_list:
# 			sp_dict = OrderedDict(sorted(sp_dict.items()))
# 			sp_confidence_dict = OrderedDict(sorted(sp_confidence_dict.items()))	

# 		Recognized_text_string = ''.join(str(val) for key, val in sp_dict.items())

# 		confidence_score_list = sp_confidence_dict.values()
# 		confidence_score_list = list(confidence_score_list)

# 		return Recognized_text_string, confidence_score_list
# 	except Exception as e:
# 		raise e


# def special_char(sp_list,Sp_char_position_list,start_x_list_sp):
	
# 	spl_chr, confidence_sp = specialChar_recog(sp_list)
		
# 	for i in range(len(Sp_char_position_list)):
# 		if Sp_char_position_list[i] == Sp_char_position["Start"]:
# 			start_x_list_sp[i] = i-3

# 	sp_dict = dict(zip(start_x_list_sp,spl_chr))
# 	sp_confidence_dict = dict(zip(start_x_list_sp,confidence_sp))
# 	return sp_dict,sp_confidence_dict

#this is our main function

def recognition_process(operation,test_img,image_name,len_expected,direction, Is_multi_char_type = False, reading_direction = HORIZONTAL):
	try:
		operation = operation.upper()

		Htr_logger.log(Htr_logger.info, "recognition_process : process start")
		byte2image = base64.b64decode(test_img)
		test_img = np.fromstring(byte2image, dtype=np.uint8)
		test_img = cv2.imdecode(test_img,cv2.IMREAD_COLOR)
		Htr_logger.log(Htr_logger.info, "recognition_process : process start 2")
				
		#condition for english1
		
		if operation=="E":

			timer.startOp()

			#for pre processign (Object detection)
			img = english_preprocess(test_img)
			#for blank detection
			img = blank_detection(img,"E")
			#for segmentation
			img_list = Segmentation_english(img)
			#for removing white spaces
			img_list = white_space_remove(img_list)

			Recognized_text_string, confidence_score_list = process_english(img_list)
			
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

				text_image_list = white_space_remove(text_image_list)

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
			# sp_list=[]
			sp_image=[]
			text=""
			segments=[]
			# Sp_char_position_list=[]
			timer.startOp()
			if Is_multi_char_type:
				text_image_list, start_x_list_text, sp_list, start_x_list_sp, Sp_char_position_list= japanese_multi_char_preprocess(test_img)
			else:
			
				#img, sp_list, start_x_list_sp, Sp_char_position_list = japanese_single_char_preprocess(test_img,.75)
                                test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
                                ret3, test_img = cv2.threshold(test_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                test_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
                                #test_img= adaptive_resizing(test_img,3.15)
                                img, sp_list, start_x_list_sp, Sp_char_position_list = japanese_single_char_preprocess(test_img,.75)
                                cv2.imwrite("od1_output/{}".format(image_name),img)


			

			# text_image_list, start_x_list_text= Segmentation_Japanese(img,reading_direction)
			text_image_list, start_x_list_text,img = segmentation_by_object_detection(img,len_expected,direction)
			# print("1")
			
			
			
			# text_image_list = blank_detection(text_image_list,"J")
			# print("2")
			text_image_list,start_x_list_text,height,area = jpn_white_space_remove(text_image_list,start_x_list_text)
			print(len_expected,len(text_image_list),"condition match")
			if len_expected == len(text_image_list):
				cv2.imwrite("od_output/{}".format(image_name),img)
				Od_output="od_output/{}".format(image_name)
				for i in range(len(text_image_list)):
						# print(image_name,"image name")
						# plt.imshow(text_image_list[i])
						# plt.show()
						cv2.imwrite("segments_jpn/{}_{}".format(i,image_name), text_image_list[i])
						segments.append("segments_jpn/{}_{}".format(i,image_name))
						
				#sp_dict,sp_confidence_dict = special_char_jpn(sp_list,Sp_char_position_list,start_x_list_sp)
				print("4")
				Recognized_text_string, confidence_score_list,flag,output2,prob2 = process_japnese(text_image_list,height,area)
                                
				Recognized_text_string=check_small(Recognized_text_string,height,area,flag,direction)
				Recognized_text_string=filter_block(Recognized_text_string)
				#confidence_score_list = np.mean(confidence_score_list)
			else:
				img, sp_list, start_x_list_sp, Sp_char_position_list = japanese_single_char_preprocess(test_img,.90)
				#cv2.imwrite("od1_output/1_{}".format(image_name),img)
				text_image_list, start_x_list_text,img = segmentation_by_object_detection(img,len_expected,direction)
				cv2.imwrite("od_output/{}".format(image_name),img)
				Od_output="od_output/{}".format(image_name)
				
				# text_image_list = blank_detection(text_image_list,"J")
				# print("2")
				text_image_list,start_x_list_text,height,area = jpn_white_space_remove(text_image_list,start_x_list_text)
				# print("3")
				for i in range(len(text_image_list)):
						# print(image_name,"image name")
						# plt.imshow(text_image_list[i])
						# plt.show()
						cv2.imwrite("segments_jpn/{}_{}".format(i,image_name), text_image_list[i])
						segments.append("segments_jpn/{}_{}".format(i,image_name))
						
				#sp_dict,sp_confidence_dict = special_char_jpn(sp_list,Sp_char_position_list,start_x_list_sp)
				print("4")
				Recognized_text_string, confidence_score_list,flag,output2,prob2 = process_japnese(text_image_list,height,area)
				Recognized_text_string=check_small(Recognized_text_string,height,area,flag,direction)
				Recognized_text_string=filter_block(Recognized_text_string)
				#confidence_score_list = confidence_score_list

			Htr_logger.log(Htr_logger.debug, "Recognition :Japanese : recognition_time: {}".format(timer.endOp()))
			Htr_logger.log(Htr_logger.info, "Recognition : Japanese : Result : {} : Confidence score : {}".format(Recognized_text_string, confidence_score_list))


		return Recognized_text_string, confidence_score_list,text,sp_image, segments,Od_output,output2,prob2
			

	except Exception as e:
		Htr_logger.log(Htr_logger.error, "Recognition : failed : {}".format(str(e)))
		timer.endOp()
		raise e

