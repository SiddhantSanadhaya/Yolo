from keras.models import model_from_json
from keras.models import load_model
import cv2
import os
import numpy as np
from cfg.htr_logger import Htr_logger
from core_logic.htr_singlechar import htr_singlechar_modelmanager as modelmanager
# from utilities.bd_logger import BD_logger

class_idx = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
             'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
               ]
				
def english_recognition(image):
    
    try:
       
        model = modelmanager.get_english_model()
    except OSError as e:
        Htr_logger.log(Htr_logger.error, "english_recognition_util : english_recognition : failure : {}".format(e))
        raise e
        
        
    # recogniton of charater
    a_pred = model.predict_classes(image)
    #probality function
    prob = sorted(model.predict_proba(image)[0])[-1]
    #taking correspoing character
    value=class_idx[a_pred[0]]
    
    Htr_logger.log(Htr_logger.info, "english_recognition_util : english_recognition : complete")
    return value,round(prob*100,2)