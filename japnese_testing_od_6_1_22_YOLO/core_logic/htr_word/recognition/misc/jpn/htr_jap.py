from core_logic.htr_word.recognition.misc.jpn.utils.jap_read_img import read_img
from core_logic.htr_word.recognition.misc.jpn.utils.jap_recognition_util import jap_recognition
#from cfg.htr_logger import Htr_logger


#jpanese recognition calling
def jap_recog(test_img):
	try:
		#Htr_logger.log(Htr_logger.info, "htr_jap : jap_recog : start")
		#reading test img 
		img= read_img(test_img)
		#passing test image to recognition model
		result, confidence_score,output2,prob2 = jap_recognition(img)
		#Htr_logger.log(Htr_logger.info, "htr_jap : jap_recog : complete")
		return result,confidence_score,output2,prob2
	except Exception as e:
		#Htr_logger.log(Htr_logger.error, "htr_jap : jap_recog : failure : {}".format(e))
		raise e




