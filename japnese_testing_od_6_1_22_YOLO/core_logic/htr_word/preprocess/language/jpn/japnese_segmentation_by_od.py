import tensorflow as tf
import numpy as np
import cv2
# import utilities.perfmeter as perfmeter
import silence_tensorflow.auto
# from utilities.bd_logger import BD_logger
from core_logic.htr_word import htr_word_modelmanager
import torch
import torch.backends.cudnn as cudnn
# from  cfg.hmr_logger import Hmr_logger
# from core_logic.model_manager import maths_detect_fn as model
from core_logic.htr_word.preprocess.language.jpn.yolov5.utils.datasets import LoadImages
from core_logic.htr_word.preprocess.language.jpn.yolov5.utils.general import (non_max_suppression, scale_coords)
from core_logic.htr_word.preprocess.language.jpn.yolov5.utils.plots import save_one_box
from matplotlib import pyplot as plt
import cv2


@torch.no_grad()

def segmentation_by_object_detection_with_expected(np_data,len_expected,reading_direction):
	image_list = []
	start_x_list_text = []
	image_list_new1 = []
	# timer.startOp()
	japanese_segmentation_detect_fn = htr_word_modelmanager.get_japanese_segmentation_preprocess_detect_fn()
	# print(timer.endOp())
	if reading_direction.lower()=="vertical":
		np_data = cv2.rotate(np_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
	try:
		japanese_segmentation_detect_fn.model.float()
		dataset = LoadImages(np_data)
		im, im0s = dataset.__next__()
		im = torch.from_numpy(im)
		im = im.float()  # uint8 to fp16/32
		im /= 255  # 0 - 255 to 0.0 - 1.0
		if len(im.shape) == 3:
			im = im[None]  # expand for batch dim

        # Inference
		pred = japanese_segmentation_detect_fn(im)
		# NMS
		pred = non_max_suppression(pred)[0]
		imc = im0s.copy()# for save_crop
		if len(pred):
			# Rescale boxes from img_size to im0 size
			pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], im0s.shape).round()
			imc_copy = imc.copy()
			for *xyxy, conf, cls in reversed(pred):
				cropped_img,start_x_text,imc_copy= save_one_box(xyxy, imc,imc_copy, BGR=True)
				cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
				if start_x_text not in start_x_list_text:
					image_list.append(cropped_img)
					start_x_list_text.append(start_x_text)
				# plt.imshow(cropped_img)
				# # plt.show()
		start_x_text_new = [x for x,y in sorted(zip(start_x_list_text,image_list))]
		image_list_new = [y for x,y in sorted(zip(start_x_list_text,image_list))]

		for img in image_list_new:
			if reading_direction.lower() == "vertical":
				img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
				image_list_new1.append(img)
			else:
				image_list_new1.append(img)

		return image_list_new1, start_x_text_new,imc_copy


	except Exception as e:
		raise e



def segmentation_by_object_detection_without_expected(np_data,reading_direction):
	image_list = []
	start_x_list_text = []
	image_list_new1=[]
	# timer.startOp()
	japanese_segmentation_detect_fn = htr_word_modelmanager.get_japanese_segmentation_preprocess_detect_fn()
	if reading_direction.lower()=="vertical":
		np_data = cv2.rotate(np_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
	
	try:
		img = np_data.copy()
		Od_image=np_data.copy()
		# count = 0
		
		for i in range(13):

			input_tensor = tf.convert_to_tensor(np_data)
			input_tensor = input_tensor[tf.newaxis, ...]
			detections = japanese_segmentation_detect_fn(input_tensor)
			index_text= np.where(detections["detection_classes"]==1)[1]
			# print(len(index_text))
			# print(detections["detection_scores"][0][index_text[0]])
			if detections["detection_scores"][0][index_text[0]] >= .15:
				# count += 1
				start_x_text=int(detections["detection_boxes"][0][index_text[0]][1]*np_data.shape[1])
				start_y_text=int(detections["detection_boxes"][0][index_text[0]][0]*np_data.shape[0])
				end_x_text=int(detections["detection_boxes"][0][index_text[0]][3]*np_data.shape[1])
				end_y_text=int(detections["detection_boxes"][0][index_text[0]][2]*np_data.shape[0])
				image = img[start_y_text:end_y_text,start_x_text:end_x_text]
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				if start_x_text not in start_x_list_text:
					image_list.append(image)
					start_x_list_text.append(start_x_text)
				cv2.rectangle(np_data,(start_x_text,start_y_text),(end_x_text,end_y_text),(255,255,255),-1)

		start_x_text_new = [x for x,y in sorted(zip(start_x_list_text,image_list))]
		image_list_new = [y for x,y in sorted(zip(start_x_list_text,image_list))]

		for img in image_list_new:
			if reading_direction.lower() == "vertical":
				img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
				image_list_new1.append(img)
			else:
				image_list_new1.append(img)
		

	except Exception as e:
		raise e

	return image_list_new1, start_x_text_new

def segmentation_by_object_detection(test_img,expected,reading_direction):
	try:
		if expected == 0:
			text_image_list, start_x_list_text = segmentation_by_object_detection_without_expected(test_img,reading_direction)
		else:
			text_image_list, start_x_list_text,img = segmentation_by_object_detection_with_expected(test_img,expected,reading_direction)
	except Exception as e:
		raise e
	return text_image_list, start_x_list_text,img
