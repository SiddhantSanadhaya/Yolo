import os
import sys
from cfg.htr_logger import Htr_logger
from core_logic.htr_singlechar.recognition.languages.math.utils.math_read_img import read_img
from core_logic.htr_singlechar.recognition.languages.math.utils.math_recognition_util import math_recognition



def math_recog(test_img):
    try:
        img= read_img(test_img)
        result, confidence_score = math_recognition(img)
        Htr_logger.log(Htr_logger.info, "htr_num : math_recog : complete")
        return result,confidence_score
    except Exception as e:
        Htr_logger.log(Htr_logger.error, "htr_num : math_recog : failure : {}".format(e))
        raise e
