import cv2
import numpy as np
from core_logic.htr_singlechar.recognition.languages.jpn.constant import constant_values
from cfg.htr_logger import Htr_logger

def read_img(image):
	try:
		
		#thresholding
		dst = cv2.detailEnhance(image, sigma_s = 80.0, sigma_r = 0.90)
		gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
		ret2,th4 = cv2.threshold(gray,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
		#adding padding to image
		img = np.pad(th4,(constant_values.pad_x,constant_values.pad_y))
		#float type conversion
		img = img.astype('float32')
		
		#image resize
		img = cv2.resize(img,(constant_values.resize_x,constant_values.resize_y))
		img = cv2.threshold(img,0,255, cv2.THRESH_BINARY)[1]
		
		#reshaping image for model
		im2arr = img.reshape(constant_values.n_image,constant_values.resize_x,constant_values.resize_y,constant_values.channel)
		Htr_logger.log(Htr_logger.info, "jap_read_ig : read_img : complete")
		return im2arr
	except Exception as e:
		Htr_logger.log(Htr_logger.error, "jap_read_ig : read_img : failure : {}".format(e))
		raise e
