import silence_tensorflow.auto
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import cv2
from core_logic.htr_word import htr_word_modelmanager
# from core_logic.htr_lib.perfmeter import Timer
from cfg.htr_logger import Htr_logger
# timer =Timer()

Sp_char_position = {"No_Special_char":0,"Start":1, "Mid":2,"End":3 }

def japanese_multi_char_preprocess(np_data):

    try:
        sp_list = []
        start_x_list_sp = []
        japanese_detect_fn = htr_word_modelmanager.get_japanese_multi_char_detectfn()

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(np_data)

        input_tensor = input_tensor[tf.newaxis, ...]
        # timer.startOp()
        detections = japanese_detect_fn(input_tensor)

        # Htr_logger.log(Htr_logger.debug,"japanese_od : japanese_multi_char_preprocess : obj_detection_time : {}".format(timer.endOp()))
        
        #finidig text classes index in ouput tensor from image
        index_text= np.where(detections["detection_classes"]==5)[1]
        index_sp=np.where(detections["detection_classes"]==4)[1]


           
        start_x_text=int(detections["detection_boxes"][0][index_text[0]][1]*np_data.shape[1])
        start_y_text=int(detections["detection_boxes"][0][index_text[0]][0]*np_data.shape[0])
        end_x_text=int(detections["detection_boxes"][0][index_text[0]][3]*np_data.shape[1])
        end_y_text=int(detections["detection_boxes"][0][index_text[0]][2]*np_data.shape[0])
        
        image = np_data[start_y_text:end_y_text,start_x_text:end_x_text]
        Sp_char_position_list = []
        for i in range(2):
            if (detections["detection_scores"][0][i] >=.70):

                # print(detections["detection_scores"][0][i])
                start_x_sp = int(detections["detection_boxes"][0][index_sp[i]][1] * np_data.shape[1])
                start_y_sp = int(detections["detection_boxes"][0][index_sp[i]][0] * np_data.shape[0])
                end_x_sp = int(detections["detection_boxes"][0][index_sp[i]][3] * np_data.shape[1])
                end_y_sp = int(detections["detection_boxes"][0][index_sp[i]][2] * np_data.shape[0])
                image_sp = np_data[start_y_sp:end_y_sp, start_x_sp:end_x_sp]

                if start_x_sp < start_x_text:
                    # print("True")
                    Special_char_position = Sp_char_position["Start"]
                elif start_x_sp > start_x_text and end_x_sp <end_x_text:
                    cv2.rectangle(np_data,(start_x_sp,start_y_sp),(end_x_sp,end_y_sp),(255,255,255),-1)
                    Special_char_position = Sp_char_position["Mid"]
                    start_x_sp = start_x_sp-start_x_text
                elif end_x_sp>end_x_text:
                    Special_char_position = Sp_char_position["End"]
                else:
                    Special_char_position = Sp_char_position["No_Special_char"]
                Sp_char_position_list.append(Special_char_position)
                # print(Sp_char_position_list,"Sp_char_position_list in preprocess")

                image_sp=np.array(image_sp)

                start_x_list_sp.append(start_x_sp)
                sp_list.append(image_sp)

        img=np.array(image)
        
        Htr_logger.log(Htr_logger.info,"japanese_od : japanese_multi_char_preprocess : complete")
        # print(Sp_char_position_list,"Sp_char_position_list in preprocess+out side loop")
        return img,sp_list,start_x_list_sp, Sp_char_position_list

    except Exception as e:
        Htr_logger.log(Htr_logger.error,"japanese_od : japanese_multi_char_preprocess : failure : {}".format(e))
        # timer.endOp()
        raise e

def japanese_single_char_preprocess(np_data):

    try:
        # print(type(np_data))
        image_copy = np_data.copy()
        sp_list = []
        start_x_list_sp = []
        japanese_detect_fn = htr_word_modelmanager.get_japanese_single_char_detectfn()

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(np_data)

        input_tensor = input_tensor[tf.newaxis, ...]
        # timer.startOp()
        detections = japanese_detect_fn(input_tensor)

        # Htr_logger.log(Htr_logger.debug,"japanese_od : japanese_single_char_preprocess : obj_detection_time : {}".format(timer.endOp()))
        
        #finidig text classes index in ouput tensor from image
        index_printed= np.where(detections["detection_classes"]==2)[1]
        index_text= np.where(detections["detection_classes"]==4)[1]
        index_sp=np.where(detections["detection_classes"]==3)[1]
        # printed_name= (detections["detection_classes"]==2)[0]
        # print(printed_name)


        start_x_text=int(detections["detection_boxes"][0][index_text[0]][1]*np_data.shape[1])
        start_y_text=int(detections["detection_boxes"][0][index_text[0]][0]*np_data.shape[0])
        end_x_text=int(detections["detection_boxes"][0][index_text[0]][3]*np_data.shape[1])
        end_y_text=int(detections["detection_boxes"][0][index_text[0]][2]*np_data.shape[0])
        cv2.rectangle(np_data,(start_x_text,start_y_text),(end_x_text,end_y_text),(255,255,255),-1)
        np_data_1 = np_data
        # plt.imshow(np_data_1)
        # plt.show()
        
        Sp_char_position_list = []
        for i in range(2):
            if detections["detection_scores"][0][index_sp[i]] >= .46:

                # print(detections["detection_scores"][0][i])
                start_x_sp = int(detections["detection_boxes"][0][index_sp[i]][1] * np_data.shape[1])
                start_y_sp = int(detections["detection_boxes"][0][index_sp[i]][0] * np_data.shape[0])
                end_x_sp = int(detections["detection_boxes"][0][index_sp[i]][3] * np_data.shape[1])
                end_y_sp = int(detections["detection_boxes"][0][index_sp[i]][2] * np_data.shape[0])
                image_sp = np_data[start_y_sp:end_y_sp, start_x_sp:end_x_sp]

                if start_x_sp < start_x_text:
                    Special_char_position = Sp_char_position["Start"]
                elif start_x_sp > start_x_text and end_x_sp <end_x_text:
                    cv2.rectangle(np_data,(start_x_sp,start_y_sp),(end_x_sp,end_y_sp),(255,255,255),-1)
                    Special_char_position = Sp_char_position["Mid"]
                    start_x_sp = start_x_sp-start_x_text
                elif end_x_sp>end_x_text:
                    Special_char_position = Sp_char_position["End"]
                else:
                    Special_char_position = Sp_char_position["No_Special_char"]
                Sp_char_position_list.append(Special_char_position)

                image_sp=np.array(image_sp)

                start_x_list_sp.append(start_x_sp)
                sp_list.append(image_sp)
        if detections["detection_scores"][0][index_printed[0]] >= .75:
            start_x_printed=int(detections["detection_boxes"][0][index_printed[0]][1]*np_data_1.shape[1])
            start_y_printed=int(detections["detection_boxes"][0][index_printed[0]][0]*np_data_1.shape[0])
            end_x_printed=int(detections["detection_boxes"][0][index_printed[0]][3]*np_data_1.shape[1])
            end_y_printed=int(detections["detection_boxes"][0][index_printed[0]][2]*np_data_1.shape[0])

            cv2.rectangle(image_copy,(start_x_printed,start_y_printed),(end_x_printed,end_y_printed),(255,255,255),-1)


        Htr_logger.log(Htr_logger.info,"japanese_od : japanese_single_char_preprocess : complete")
        
        return image_copy,sp_list,start_x_list_sp, Sp_char_position_list

    except Exception as e:
        Htr_logger.log(Htr_logger.error,"japanese_od : japanese_single_char_preprocess : failure : {}".format(e))
        # timer.endOp()
        raise e
