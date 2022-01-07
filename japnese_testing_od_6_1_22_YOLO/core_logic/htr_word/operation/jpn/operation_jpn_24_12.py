import cv2
from core_logic.htr_word.recognition.misc.jpn.htr_jap import jap_recog
from collections import OrderedDict
#from core_logic.htr_word.preprocess.language.jpn.japanese_od import Sp_char_position
#from core_logic.htr_word.recognition.misc.specialChar.htr_specialChar import specialChar_recog
import matplotlib.pyplot as plt

def check_ichi(img_list,height,area):
	count1=0
	flag="False"
	for count in range(len(img_list)):
		if height[count] < 10:
			# if area[count] <= sorted(area)[1]:
			flag="True"+str(count1)
			count1+=1
			# else:
				# flag="False"
	return flag


def process_japnese(img_list,height,area):
	results = []
	confidence_scores = []
	try:
		flag=check_ichi(img_list,height,area)
		count=0
		for img in img_list:
			#result, confidence_score = jap_recog(img)
                        #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                        result, confidence_score = jap_recog(img)
                        count=count+1
                        results.append(result)
                        confidence_scores.append(confidence_score)

		#jap_dict = dict(zip(start_x_list_text,results))
		#confidence_dict = dict(zip(start_x_list_text,confidence_scores))
		#print(confidence_dict,"confidence_dict")

		#sp_dict.update(jap_dict)
		#sp_confidence_dict.update(confidence_dict)

		#if  Sp_char_position["Mid"] in Sp_char_position_list  or Sp_char_position["End"] in Sp_char_position_list:
			#sp_dict = OrderedDict(sorted(sp_dict.items()))
			#sp_confidence_dict = OrderedDict(sorted(sp_confidence_dict.items()))	

		#Recognized_text_string = ''.join(str(val) for key, val in sp_dict.items())

		#confidence_score_list = sp_confidence_dict.values()
		#confidence_score_list = list(confidence_score_list)

		return results, confidence_scores,flag
	except Exception as e:
		raise e
