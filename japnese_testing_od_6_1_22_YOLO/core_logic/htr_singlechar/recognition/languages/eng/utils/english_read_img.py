import cv2
import numpy as np
from core_logic.htr_singlechar.recognition.languages.eng.constant import constant_values
from cfg.htr_logger import Htr_logger
def read_img(image):
	try:
		#conversion to gray
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				
		image=cv2.bitwise_not(gray)
		kernel = np.ones(constant_values.gb_kernel_size,np.uint8)
		img=cv2.dilate(image,kernel,iterations = 1)
		#add pading to image
		img= np.pad(img,constant_values.pad_1)
		#image resizing
		img=cv2.resize(img, (constant_values.resize_x,constant_values.resize_y))
		
		#image reshapping according to input
		im2arr = img.reshape(constant_values.n_image,constant_values.resize_x,constant_values.resize_y,constant_values.channel)
		Htr_logger.log(Htr_logger.info, "english_read_img : read_img : complete")
		return im2arr
	except Exception as e:
			Htr_logger.log(Htr_logger.error, "english_read_img : read_img : failure : {}".format(e))
			raise e