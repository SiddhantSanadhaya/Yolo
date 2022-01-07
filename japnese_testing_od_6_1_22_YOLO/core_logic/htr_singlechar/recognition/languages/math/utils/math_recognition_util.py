from keras.models import load_model
import os
import numpy as np
from cfg.htr_logger import Htr_logger
from core_logic.htr_singlechar import htr_singlechar_modelmanager as modelmanager


def math_recognition(image):
	try:
		model = modelmanager.get_numeric_model()
		
	except FileNotFoundError as e:
		Htr_logger.log(Htr_logger.error,"math_recognition_util : failure : Model file does not found")
		raise e

	a_pred = np.argmax(model.predict(image), axis=-1)
	prob = sorted(model.predict_proba(image)[0])[-1]
	Htr_logger.log(Htr_logger.info,"math_recognition_util : complete")
	return a_pred[0],round(prob*100,2)

