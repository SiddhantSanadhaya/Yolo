
import tensorflow as tf
from cfg.htr_logger import Htr_logger
import os
import silence_tensorflow.auto
import warnings
from core_logic.htr_word.preprocess.language.jpn.yolov5.models.common import DetectMultiBackend
warnings.filterwarnings('ignore')
from keras.models import load_model

def english_preprocess_model():
    try:
        #path for saves model
        PATH_TO_SAVED_MODEL = os.path.join(os.getcwd(),"core_logic/htr_word/preprocess/language/eng/model/saved_model")
        # print(PATH_TO_SAVED_MODEL)
        #model loading
        model_loaded = tf.saved_model.load(PATH_TO_SAVED_MODEL)
        return model_loaded
    except Exception as e:
        Htr_logger.log(Htr_logger.error,"modelManger : load_model : failure : {}".format(e))
        raise e

def math_multi_char_preprocess_model():
    try:
        #path for saves model
        PATH_TO_SAVED_MODEL = os.path.join(os.getcwd(),"core_logic/htr_word/preprocess/language/math/model/multi_char/saved_model")
        # print(PATH_TO_SAVED_MODEL)
        #model loading
        model_loaded = tf.saved_model.load(PATH_TO_SAVED_MODEL)
        return model_loaded
    except Exception as e:
        Htr_logger.log(Htr_logger.error,"modelManger : load_model : failure : {}".format(e))
        raise e
def japanese_multi_char_preprocess_model():
    try:
        #path for saves model
        PATH_TO_SAVED_MODEL = os.path.join(os.getcwd(),"core_logic/htr_word/preprocess/language/jpn/model/multi_char/saved_model")
        # print(PATH_TO_SAVED_MODEL)
        #model loading
        model_loaded = tf.saved_model.load(PATH_TO_SAVED_MODEL)
        return model_loaded
    except Exception as e:
        Htr_logger.log(Htr_logger.error,"modelManger : load_model : failure : {}".format(e))
        raise 

def math_single_char_preprocess_model():
    try:
        #path for saves model
        PATH_TO_SAVED_MODEL = os.path.join(os.getcwd(),"core_logic/htr_word/preprocess/language/math/model/single_char/saved_model")
        # print(PATH_TO_SAVED_MODEL)
        #model loading
        model_loaded = tf.saved_model.load(PATH_TO_SAVED_MODEL)
        return model_loaded
    except Exception as e:
        Htr_logger.log(Htr_logger.error,"modelManger : load_model : failure : {}".format(e))
        raise e

def japanese_single_char_preprocess_model():
    try:
        #path for saves model
        PATH_TO_SAVED_MODEL = os.path.join(os.getcwd(),"core_logic/htr_word/preprocess/language/jpn/model/single_char/saved_model_printed/")
        # print(PATH_TO_SAVED_MODEL)
        #model loading
        model_loaded = tf.saved_model.load(PATH_TO_SAVED_MODEL)
        return model_loaded
    except Exception as e:
        Htr_logger.log(Htr_logger.error,"modelManger : load_model : failure : {}".format(e))
        raise e

def japanese_segmentation_preprocess_model():
    try:
        weights = '../../best1.pt'
        model_loaded = DetectMultiBackend(weights)
        return model_loaded
    except Exception as e:
        Htr_logger.log(Htr_logger.error,"modelManger : japanese_segmentation_preprocess_model : failure : {}".format(e))
        exit()

def english_preprocess_od_model_for_unknown_expected():
    try:
        #path for saves moel
        PATH_TO_SAVED_MODEL = os.path.join(os.getcwd(),"core_logic/htr_word/preprocess/language/eng/model/remove_border_ob_model/saved_model")
        # print(PATH_TO_SAVED_MODEL)
        #model loading
        model_loaded = tf.saved_model.load(PATH_TO_SAVED_MODEL)
        return model_loaded
    except Exception as e:
        Htr_logger.log(Htr_logger.error,"modelManger : load_model : failure : {}".format(e))
        raise e


def load_english_recognition_model():
    model = load_model('core_logic/htr_word/recognition/misc/eng/model/english.h5')
    return model

def load_japanese_recognition_model():

    model = tf.keras.models.load_model('core_logic/htr_word/recognition/misc/jpn/model/1416_classes_300_vertical_wrong_73_73_40_1.h5')
    return model

def load_numeric_recognition_model():
    model = load_model('core_logic/htr_word/recognition/misc/math/model/math.h5')
    return model

def load_specialChar_recognition_model():
    model = load_model('core_logic/htr_word/recognition/misc/specialChar/model/specialchar.h5')
    return model

def load_unit_recognition_model():
    model = load_model('core_logic/htr_word/recognition/misc/unit/model/unit_model.h5')
    return model

def  get_english_detectfn():
    return english_detect_fn

def  get_math_multi_char_detectfn():
    return math_multi_char_detect_fn

def  get_japanese_multi_char_detectfn():
    return japanese_multi_char_detect_fn

def  get_math_single_char_detectfn():
    return math_single_char_detect_fn

def  get_japanese_single_char_detectfn():
    return japanese_single_char_detect_fn

def get_japanese_segmentation_preprocess_detect_fn():
    return japanese_segmentation_detect_fn

def  get_english_model():
    return english_model

def  get_japanese_model():
    return japanese_model

def  get_numeric_model():
    return numeric_model

def  get_specialChar_model():
    return specialchar_model

def  get_unit_model():
    return unit_model

def get_english_wordfn():
    return english_text_fn


def load_all_models():
    
    global english_text_fn,english_detect_fn,english_model,japanese_model,numeric_model,specialchar_model,unit_model,math_multi_char_detect_fn,math_single_char_detect_fn,japanese_single_char_detect_fn,japanese_multi_char_detect_fn,japanese_segmentation_detect_fn
    
    # english_text_fn = english_preprocess_od_model_for_unknown_expected() #OD1

    # english_detect_fn = english_preprocess_model() #OD2

    # math_single_char_detect_fn = math_single_char_preprocess_model()

    japanese_single_char_detect_fn = japanese_single_char_preprocess_model()

    # math_multi_char_detect_fn = math_multi_char_preprocess_model()

    japanese_segmentation_detect_fn = japanese_segmentation_preprocess_model()

    # japanese_multi_char_detect_fn = japanese_multi_char_preprocess_model()

    # english_model = load_english_recognition_model()

    japanese_model = load_japanese_recognition_model()

    # numeric_model = load_numeric_recognition_model()

    # specialchar_model = load_specialChar_recognition_model()

    # unit_model = load_unit_recognition_model()

    
load_all_models()
