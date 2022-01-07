import cv2
import os
import numpy as np
# count=1

def white_space_remove(img_list,start_x_list_text):
    final_list = []
    start_x_list = []
    for i in  range(len(img_list)):
    
        # print(i[2])
        try:     
            # print(i[2][x])
            # img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            black = np.zeros((img_list[i].shape[1],1),np.uint8)
            black = cv2.bitwise_not(black)
            img_rotated = cv2.rotate(img_list[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
            #blur = cv2.bilateralFilter(gray,9,75,75)
            blur = cv2.medianBlur(img_rotated, 3)
            ret,th4 = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
            #th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            img_erosion = cv2.erode(th4, (3,3), iterations=1)
            res = cv2.matchTemplate(black,img_erosion,cv2.TM_SQDIFF_NORMED)
            for j in range(res.shape[1]):
                res[0][j] = round(res[0][j],2)
            lst=[]
            for j in range(len(res[0])):
                if res[0][j]!=0:
                    lst.append(j)
            if len(lst) != 0:
                img_rotated_cropped = img_rotated[:,min(lst):max(lst)]
            else:
                img_rotated_cropped = img_rotated

            img = cv2.rotate(img_rotated_cropped, cv2.ROTATE_90_CLOCKWISE)
            black = np.zeros((img.shape[0],1),np.uint8)
            black = cv2.bitwise_not(black)
            #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #blur = cv2.bilateralFilter(gray,9,75,75)
            blur = cv2.medianBlur(img, 3)
            ret,th4 = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
            #th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            img_erosion = cv2.erode(th4, (3,3), iterations=1)
            res = cv2.matchTemplate(black,img_erosion,cv2.TM_SQDIFF_NORMED)
            for j in range(res.shape[1]):
                res[0][j] = round(res[0][j],2)
            lst=[]
            for j in range(len(res[0])):
                if res[0][j]!=0:
                    lst.append(j)
            if len(lst) != 0:        
                img = img[:,min(lst):max(lst)]
            else:
                img = img
            
            
            # h,w = img.shape
            # print(h,w)

            if img.shape[0] != 0 and img.shape[1] !=0:
                final_list.append(img)
                start_x_list.append(start_x_list_text[i])
                # print(1)
            # #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # final_list.append(img)
            # start_x_list.append(start_x_list_text[i])
            # print(img.shape)
            # cv2.imwrite("D:/Rawattech/Math_Recog/segments_1/{}".format(i[2][x]),img)
            # count+=1
        except ValueError as e:
            pass
        except Exception as e:
            raise e
    return final_list,start_x_list