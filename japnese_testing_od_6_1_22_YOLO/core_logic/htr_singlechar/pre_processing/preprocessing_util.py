import os
import time
from core_logic.htr_lib.perfmeter import Timer
import sys
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import cv2
import silence_tensorflow.auto
from cfg.htr_logger import Htr_logger
from core_logic.htr_singlechar import  htr_singlechar_modelmanager as modelmanager
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
timer=Timer()

#Object detection module

# def read_image(image_np):
#     #read image
#     #line 19 can generates an error 
#     try:
#         #condition to check  whether the input is correct or not
#         if image_np.shape:
#             Htr_logger.log(Htr_logger.info,"preprocessing_util : read_image : complete")
#             return color, thresh

#     except Exception as e: 
#         Htr_logger.log(Htr_logger.error,"preprocessing_util : read_image : failure : {}".format(e))
#         raise e

def preprocess(np_data):
    try:
        
        
        imOP = cv2.cvtColor(np_data, cv2.COLOR_BGR2GRAY)
        h,w=imOP.shape
        
        x_end=int(w*.85)
        y_end=int(h*.80)
        y=h-y_end
        x=w-x_end
        timer.startOp()
        
        Htr_logger.log(Htr_logger.debug,"preprocessing_util : preprocess : read_image_time : {}".format(timer.endOp()))
        #model loading
        detect_fn = modelmanager.get_detectfn()
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(np_data)

        input_tensor = input_tensor[tf.newaxis, ...]
        timer.startOp()
        detections = detect_fn(input_tensor)
        
        Htr_logger.log(Htr_logger.debug,"preprocessing_util : preprocess : obj_detection_time : {}".format(timer.endOp()))
        #finidig text classes index in ouput tensor from image
        index= np.where(detections["detection_classes"]==4)[1]
        text_ind=np.where(detections["detection_classes"]==3)[1]

        #cropping the text class from image
        timer.startOp()
        if(detections["detection_scores"][0][index[0]] >=.80):
            start_x=int(detections["detection_boxes"][0][index[0]][1]*np_data.shape[1])
            start_y=int(detections["detection_boxes"][0][index[0]][0]*np_data.shape[0])
            end_x=int(detections["detection_boxes"][0][index[0]][3]*np_data.shape[1])
            end_y=int(detections["detection_boxes"][0][index[0]][2]*np_data.shape[0])
            cv2.rectangle(np_data,(start_x,start_y),(end_x,end_y),(255,255,255),-1)
        

            
        start_x=int(detections["detection_boxes"][0][text_ind[0]][1]*np_data.shape[1])
        start_y=int(detections["detection_boxes"][0][text_ind[0]][0]*np_data.shape[0])
        end_x=int(detections["detection_boxes"][0][text_ind[0]][3]*np_data.shape[1])
        end_y=int(detections["detection_boxes"][0][text_ind[0]][2]*np_data.shape[0])

        image = np_data[start_y:end_y,start_x:end_x]

        
        img=np.array(image)
        if ((end_x>x and start_y <y_end) and (start_x<x_end and start_y<y_end)) and ((end_x>x and start_y<y_end) and (end_x>x and end_y>y)) and ((start_x<x_end and end_y>y) and (start_x<x_end and start_y<y_end)) and ((end_x>x and end_y>y) and (start_x<x_end and end_y>y)):
            Htr_logger.log(Htr_logger.info,"preprocessing_util : preprocess : complete")
            timer.endOp()
            return img
        else:
            #raise ValueError("Image is blank")
            print("blank")

    except Exception as e:
        Htr_logger.log(Htr_logger.error,"preprocessing_util : preprocess : failure : {}".format(e))
        timer.endOp()
        raise e




