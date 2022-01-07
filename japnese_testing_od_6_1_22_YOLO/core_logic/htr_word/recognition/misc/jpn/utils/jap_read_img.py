import cv2
import numpy as np
from core_logic.htr_word.recognition.misc.jpn.constant import constant_values
from cfg.htr_logger import Htr_logger

def read_img(image):
	try:
		#thresholding
		#image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		image=cv2.bitwise_not(image)

		#adding padding to image
		img = np.pad(image,(constant_values.pad_x,constant_values.pad_y))
		
		#thresholding image 
		ret2,th4 = cv2.threshold(img,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
		
		#image resize
		img = cv2.resize(th4,(constant_values.resize_x,constant_values.resize_y))
		
		#reshaping image for model       
		im2arr = img.reshape(constant_values.n_image,constant_values.resize_x,constant_values.resize_y,constant_values.channel)
		Htr_logger.log(Htr_logger.info, "jap_read_ig : read_img : complete")
		return im2arr
	except Exception as e:
		Htr_logger.log(Htr_logger.error, "jap_read_ig : read_img : failure : {}".format(e))
		raise e
