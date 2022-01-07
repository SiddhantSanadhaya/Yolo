import cv2
import numpy as np
from core_logic.htr_singlechar.recognition.languages.math.constant import constant_values
from cfg.htr_logger import Htr_logger
def read_img(image):
	try:
		
		# thresholding
		dst = cv2.detailEnhance(image, sigma_s = 80.0, sigma_r = 0.90)
		gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
		thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,9,9)
		img = np.pad(thresh,(constant_values.pad_x,constant_values.pad_y))
		#resize
		img = cv2.resize(img,(constant_values.resize_x,constant_values.resize_y))
		im2arr = np.array(img)

		#reshape for model
		im2arr = im2arr.reshape(constant_values.n_image,constant_values.resize_x,constant_values.resize_y,constant_values.channel)
		Htr_logger.log(Htr_logger.info, "math_read_image : read_image : complete")
		return im2arr
		
	except Exception as e:
		Htr_logger.log(Htr_logger.error, "math_read_image : read_image : failure : {}".format(e))
		raise e