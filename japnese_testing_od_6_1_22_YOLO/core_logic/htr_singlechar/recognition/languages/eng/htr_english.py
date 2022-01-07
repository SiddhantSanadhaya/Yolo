import os
import sys
from core_logic.htr_singlechar.recognition.languages.eng.utils.english_read_img import read_img 
from core_logic.htr_singlechar.recognition.languages.eng.utils.english_recognition_util import english_recognition
from cfg.htr_logger import Htr_logger

#english recog code
def english_recog(img):
	try:
		#read image
		img= read_img(img)
		#calling recognition model and getting output
		result, confidence_score = english_recognition(img)
		Htr_logger.log(Htr_logger.info, "htr_english : english_recog : complete")
		return result,confidence_score
	except Exception as e:
		Htr_logger.log(Htr_logger.error, "htr_english : english_recog : failure : {}".format(e))
		raise e