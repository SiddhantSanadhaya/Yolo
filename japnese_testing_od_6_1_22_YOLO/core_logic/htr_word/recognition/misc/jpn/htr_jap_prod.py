import os
import sys
from core_logic.htr_word.recognition.misc.jpn.utils.jap_read_img import read_img
from core_logic.htr_word.recognition.misc.jpn.utils.jap_recognition_util import jap_recognition,jap_recognition1
from cfg.htr_logger import Htr_logger
#jpanese recognition calling
def jap_recog(test_img,height,area,flag,count):
	try:
		# print(test_img.size,height,area)
		if flag=="True0":
			if area[count] <= sorted(area)[1] and height[count] <= max(height)*0.75:
				# print("small.................")
				Htr_logger.log(Htr_logger.info, "htr_jap : jap_recog : start")
				#reading test img
				img= read_img(test_img)
				#passing test image to recognition model
				result, confidence_score = jap_recognition1(img)
				Htr_logger.log(Htr_logger.info, "htr_jap : jap_recog : complete")
			else:
				Htr_logger.log(Htr_logger.info, "htr_jap : jap_recog : start")
				#readning test img
				img= read_img(test_img)
				#passing test image to recognition model
				result, confidence_score = jap_recognition(img)
				Htr_logger.log(Htr_logger.info, "htr_jap : jap_recog : complete")
		elif flag =="True1":
			if area[count] <= sorted(area)[2] and height[count] <= max(height)*0.75:
				# print("small.................")
				Htr_logger.log(Htr_logger.info, "htr_jap : jap_recog : start")
				#reading test img
				img= read_img(test_img)
				#passing test image to recognition model
				result, confidence_score = jap_recognition1(img)
				Htr_logger.log(Htr_logger.info, "htr_jap : jap_recog : complete")
			else:
				Htr_logger.log(Htr_logger.info, "htr_jap : jap_recog : start")
				#readning test img
				img= read_img(test_img)
				#passing test image to recognition model
				result, confidence_score = jap_recognition(img)
				Htr_logger.log(Htr_logger.info, "htr_jap : jap_recog : complete")
		else:
			if area[count] <= sorted(area)[0] and height[count] <= max(height)*0.65:
				# print("small.................")
				Htr_logger.log(Htr_logger.info, "htr_jap : jap_recog : start")
				#reading test img
				img= read_img(test_img)
				#passing test image to recognition model
				result, confidence_score = jap_recognition1(img)
				Htr_logger.log(Htr_logger.info, "htr_jap : jap_recog : complete")

			else:
				Htr_logger.log(Htr_logger.info, "htr_jap : jap_recog : start")
				#readning test img
				img= read_img(test_img)
				#passing test image to recognition model
				result, confidence_score = jap_recognition(img)
				Htr_logger.log(Htr_logger.info, "htr_jap : jap_recog : complete")
		return result,confidence_score
	except Exception as e:
            Htr_logger.log(Htr_logger.error, "htr_jap : jap_recog : failure : {}".format(e))
            raise e

