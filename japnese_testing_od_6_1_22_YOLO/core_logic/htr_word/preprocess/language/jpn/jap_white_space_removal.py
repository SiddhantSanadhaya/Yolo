import cv2
import os
import numpy as np
# import matplotlib.pyplot as plt
# count=1

def jpn_white_space_remove(text_image_list,start_x_list_text):
    final_list = []
    start_x_list = []
    height_lst=[]
    area_lst=[]
    # width_lst=[]
    for i in range(len(text_image_list)):
    
        # print(i[2])
        try:     
            # print(i[2][x])
            img= text_image_list[i]
            black = np.zeros((text_image_list[i].shape[1],1),np.uint8)
            black = cv2.bitwise_not(black)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #blur = cv2.bilateralFilter(gray,9,75,75)
            #blur = cv2.medianBlur(img, 3)
            ret,binarized_img_h= cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            img_erosion = cv2.erode(binarized_img_h, (3,3), iterations=1)
            res = cv2.matchTemplate(black,img_erosion,cv2.TM_SQDIFF_NORMED)
            for j in range(res.shape[1]):
                res[0][j] = round(res[0][j],2)
            lst=[]
            for j in range(len(res[0])):
                if res[0][j]!=0:
                    lst.append(j)
            if len(lst) != 0:
                binarized_img_h = binarized_img_h[:,min(lst):max(lst)]
            
            

            img = cv2.rotate(binarized_img_h, cv2.ROTATE_90_CLOCKWISE)
            black = np.zeros((img.shape[0],1),np.uint8)
            black = cv2.bitwise_not(black)
            #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #blur = cv2.bilateralFilter(gray,9,75,75)
            #blur = cv2.medianBlur(img, 3)
            ret,binarized_img_v = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            img_erosion = cv2.erode(binarized_img_v, (3,3), iterations=1)
            res = cv2.matchTemplate(black,img_erosion,cv2.TM_SQDIFF_NORMED)
            for j in range(res.shape[1]):
                res[0][j] = round(res[0][j],2)
            lst=[]
            for j in range(len(res[0])):
                if res[0][j]!=0:
                    lst.append(j)
            if len(lst) != 0:
                img = binarized_img_v[:,min(lst):max(lst)]
                area_lst.append(text_image_list[i].size)
                height_lst.append(img.shape[0])
                # width_lst.append(img.shape[1])
                final_list.append(text_image_list[i])
                start_x_list.append(start_x_list_text[i])
            else:
                pass

        except ValueError as e:
            pass    
        except Exception as e:
            raise e
    # print(area_lst,height_lst)
    # width=width_lst
    area=area_lst
    height=height_lst
    # print(area,height)
    return final_list,start_x_list,height,area
