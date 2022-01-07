import numpy as np
import cv2
import os
import sys
import core_logic.htr_singlechar.recognition.languages.jpn.utils.hira_kana_label_1 as labels
from keras.models import load_model,model_from_json
from cfg.htr_logger import Htr_logger
import warnings
np.warnings.filterwarnings("ignore")
import tensorflow.compat.v1 as tf
from core_logic.htr_singlechar import htr_singlechar_modelmanager as  modelmanager



def load_model():
    try:
        model = modelmanager.get_japanese_model()
        return model
    except Exception as e:
        Htr_logger(Htr_logger.error, "jap_recognition_util : load_model : failure : {}".format(e))

def jap_recognition(img):
    try:
        model = load_model()
    except FileNotFoundError as e:
        Htr_logger.log(Htr_logger.error, "jap_recognition_util : jap_recognition : failure : {}".format(e))
        raise e
        
    out = labels.labels[model.predict_classes(img).item()]
    #find probability of each character image with sorted
    prob = sorted(model.predict_proba(img)[0])[-1]
    Htr_logger.log(Htr_logger.info, "jap_recognition_util : jap_recognition : complete")
    return out,round(prob*100,2)

 
