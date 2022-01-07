import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from cfg.htr_logger import Htr_logger
import os
import silence_tensorflow.auto
from keras.models import load_model


def load_preprocess_model():
    try:
        #path for saves model
        PATH_TO_SAVED_MODEL = os.path.join(os.getcwd(),"core_logic/htr_singlechar/pre_processing/preprocessing_model/effcient_d1/saved_model")
        
        #model loading
        model_loaded = tf.saved_model.load(PATH_TO_SAVED_MODEL)
        return model_loaded
    except Exception as e:
        Htr_logger.log(Htr_logger.error,"modelManger : load_model : failure : {}".format(e))
        raise e

def load_english_recognition_model():
    model = load_model('core_logic/htr_singlechar/recognition/languages/eng/model/english.h5')
    return model

def load_japanese_recognition_model():

    model = tf.keras.models.load_model('core_logic/htr_singlechar/recognition/languages/jpn/model/p1_2.2_model_1.h5')
    return model

def load_numeric_recognition_model():
    model = load_model('core_logic/htr_singlechar/recognition/languages/math/model/mnistv3.h5')
    return model

def  get_detectfn():
    return detect_fn_p1

def  get_english_model():
    return english_model_p1

def  get_japanese_model():
    return japanese_model_p1

def  get_numeric_model():
    return numeric_model_p1

def load_all_models():

    global detect_fn_p1,english_model_p1,japanese_model_p1,numeric_model_p1
    detect_fn_p1=load_preprocess_model()
    #english_model_p1=load_english_recognition_model()
    japanese_model_p1=load_japanese_recognition_model()
    #numeric_model_p1=load_numeric_recognition_model()

load_all_model=load_all_models()
