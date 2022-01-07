from core_logic.htr_word.recognition.misc.math.htr_num import math_recog
import cv2
from collections import OrderedDict
from core_logic.htr_word.recognition.misc.specialChar.htr_specialChar import specialChar_recog
from core_logic.htr_word.preprocess.language.math.math_od import Sp_char_position




def process_maths(img_list,x_start_list_text, sp_dict, sp_confidence_dict,Sp_char_position_list):
	results = []
	confidence_scores = []
	try:
		for img in img_list:
			img= cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
			result, confidence_score = math_recog(img)
			results.append(result)
			confidence_scores.append(confidence_score)

		math_dict = dict(zip(x_start_list_text,results))
		confidence_dict = dict(zip(x_start_list_text,confidence_scores))
		
		sp_dict.update(math_dict)

		sp_confidence_dict.update(confidence_dict)

		sp_dict = OrderedDict(sorted(sp_dict.items()))
		sp_confidence_dict = OrderedDict(sorted(sp_confidence_dict.items()))

		Recognized_text_string = ''.join(str(val) for key, val in sp_dict.items())

		confidence_score_list = sp_confidence_dict.values()
		confidence_score_list = list(confidence_score_list)

		return Recognized_text_string, confidence_score_list
	except Exception as e:
		raise e


def special_char_num(sp_list,Sp_char_position_list,start_x_list_sp):
	
	spl_chr, confidence_sp = specialChar_recog(sp_list)
		
	for i in range(len(Sp_char_position_list)):
		if Sp_char_position_list[i] == Sp_char_position["Start"]:
			start_x_list_sp[i] = i-3

	sp_dict = dict(zip(start_x_list_sp,spl_chr))
	sp_confidence_dict = dict(zip(start_x_list_sp,confidence_sp))
	return sp_dict,sp_confidence_dict