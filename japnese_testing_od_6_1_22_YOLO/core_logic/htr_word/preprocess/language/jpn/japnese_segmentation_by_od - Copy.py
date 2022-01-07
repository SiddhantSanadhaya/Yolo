import tensorflow as tf
import numpy as np
import cv2
# import utilities.perfmeter as perfmeter
import silence_tensorflow.auto
# from utilities.bd_logger import BD_logger
from core_logic.htr_word import htr_word_modelmanager

# final_list = []
def segmentation_by_object_detection_with_expected(np_data,len_expected,reading_direction):
	image_list = []
	start_x_list_text = []
	image_list_nms = []
	start_x_list_text_nms = []
	image_list_new1=[]

	# timer.startOp()
	japanese_segmentation_detect_fn = htr_word_modelmanager.get_japanese_segmentation_preprocess_detect_fn()
	# print(timer.endOp())
	if reading_direction.lower()=="vertical":
		np_data = cv2.rotate(np_data, cv2.ROTATE_90_COUNTERCLOCKWISE)

	try:
		img = np_data.copy()
		Od_image=np_data.copy()
		input_tensor = tf.convert_to_tensor(np_data)
		input_tensor = input_tensor[tf.newaxis, ...]
		detections = japanese_segmentation_detect_fn(input_tensor)
		index_text= np.where(detections["detection_classes"]==1)[1]
		boxes = []
		scores = []

		for i in range(len_expected):

			start_x_text=int(detections["detection_boxes"][0][index_text[i]][1]*np_data.shape[1])
			start_y_text=int(detections["detection_boxes"][0][index_text[i]][0]*np_data.shape[0])
			end_x_text=int(detections["detection_boxes"][0][index_text[i]][3]*np_data.shape[1])
			end_y_text=int(detections["detection_boxes"][0][index_text[i]][2]*np_data.shape[0])
			image = img[start_y_text:end_y_text,start_x_text:end_x_text]
			boxes.append((start_x_text,start_y_text,end_x_text,end_y_text))
			scores.append(detections["detection_scores"][0][index_text[i]])
			# image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
			if start_x_text not in start_x_list_text:
				image_list.append(image)
				start_x_list_text.append(start_x_text)
				cv2.rectangle(Od_image,(start_x_text,start_y_text),(end_x_text,end_y_text),(0,255,0),2)

		boxes=np.array(boxes)
		scores = np.array(scores)
		box1 = non_max_suppression_fast(boxes, 0.2)

		for (start_x_text,start_y_text,end_x_text,end_y_text) in box1:
			cv2.rectangle(np_data,(start_x_text,start_y_text),(end_x_text,end_y_text),(0,255,0),2)
			image_nms = img[start_y_text:end_y_text,start_x_text:end_x_text]
			image_nms= cv2.cvtColor(image_nms,cv2.COLOR_BGR2GRAY)
			image_list_nms.append(image_nms)
			start_x_list_text_nms.append(start_x_text)

		start_x_text_new = [x for x,y in sorted(zip(start_x_list_text_nms,image_list_nms))]
		image_list_new = [y for x,y in sorted(zip(start_x_list_text_nms,image_list_nms))]

		for img in image_list_new:
			if reading_direction.lower() == "vertical":
				img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
				image_list_new1.append(img)
			else:
				image_list_new1.append(img)
		

	except Exception as e:
		raise e

	
	return image_list_new1, start_x_text_new,Od_image


def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


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
