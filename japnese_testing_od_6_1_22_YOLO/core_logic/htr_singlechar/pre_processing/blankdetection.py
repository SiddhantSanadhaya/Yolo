import os
import sys
import numpy as np
import cv2
from cfg.htr_logger import Htr_logger


def remove_lines(image,operation):
    try:
        image = np.array(image)
        image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result=image.copy()

        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30,1))
        remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(result, [c], -1, (255,255,255), 5)
        if operation == "J":
            #Remove vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,30))
            remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(result, [c], -1, (255,255,255),5)

        unused_return,masked_img_inv = cv2.threshold(result,127,255,cv2.THRESH_BINARY)
        masked_img_inv=cv2.cvtColor(masked_img_inv, cv2.COLOR_BGR2GRAY)

        return masked_img_inv
    except Exception as e:
        Htr_logger.log(Htr_logger.error,"blankdetection : remove_lines : failure : {}".format(e))
        raise e

def blank_detection(image,operation):
    try:
        img = np.array(image)
        dst = cv2.detailEnhance(img, sigma_s =60.0, sigma_r = 1.0)
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)[1]
        grey = remove_lines(thresh,operation)
        black = np.zeros((grey.shape[1],1),np.uint8)
        black = cv2.bitwise_not(black)
        kernel = np.ones((3,3), np.uint8)
        img = cv2.rotate(grey, cv2.ROTATE_90_COUNTERCLOCKWISE)
        th4 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,20)
        img_erosion = cv2.erode(th4, kernel, iterations=2)
        res = cv2.matchTemplate(black,img_erosion,cv2.TM_SQDIFF_NORMED)

        for j in range(res.shape[1]):
            res[0][j] = round(res[0][j],2)
        lst=[]
        for j in range(len(res[0])):
            if res[0][j] != 0:
                lst.append(j)
        if len(lst) != 0:

            img = img[:,min(lst):max(lst)]
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            black = np.zeros((img.shape[0],1),np.uint8)
            black = cv2.bitwise_not(black)
            kernel = np.ones((3,3), np.uint8)
            
            th4 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,20)
            img_erosion = cv2.erode(th4, kernel, iterations=2)
            res = cv2.matchTemplate(black,img_erosion,cv2.TM_SQDIFF_NORMED)

            for j in range(res.shape[1]):
                res[0][j] = round(res[0][j],2)
            lst=[]
            for j in range(len(res[0])):
                if res[0][j]!=0:
                    lst.append(j)
            if len(lst)!=0:

                img = img[:,min(lst):max(lst)]
                
                contrast = grey.std()
                if image.shape[0]>image.shape[1]:
                    if contrast>10 and img.shape[0]>image.shape[0]-image.shape[1]:
                        Htr_logger.log(Htr_logger.info,"blankdetection : blank_detection : completed")
                        return image
                    else:
                        #raise ValueError("Image is blank")
                        print("blank")
                else:
                    if contrast>10 and img.shape[1]>image.shape[1]-image.shape[0]:
                        Htr_logger.log(Htr_logger.info,"blankdetection : blank_detection : completed")
                        return image
                    else:
                        #raise ValueError("Image is blank")
                        print("blank")

            else:
                #raise ValueError("Image is blank")
                print("blank")
        else:
            #raise ValueError("Image is blank")
            print("blank")


    except Exception as e:
        Htr_logger.log(Htr_logger.error,"blankdetection : blank_detection : failure : {}".format(e))
        raise e
