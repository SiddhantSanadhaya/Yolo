import numpy as np
#import pandas as pd
import cv2
import os
import core_logic.htr_word.preprocess.language.jpn.utilities.swt as swt
#from PIL import Image
import statistics 



diagnostics = False




class utility:   
    #Read file function
    

    def read_img(self,image,mode):
        try:
            if(mode=="v"):
                image= cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return image
        except:
            raise e
            # print("Image does not exist.")
            # sys.exit()
    #preprocessing function        
    def preprocessing(self,image,mode):
        # Grayscale 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        # blur
        blur = cv2.GaussianBlur(gray,(5,5),0)
        # adaptive threshold
        self.th5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        # erosion
        self.kernel = np.ones((3,3), np.uint8) 
        self.img_erosion = cv2.erode(self.th5, self.kernel, iterations=1)

        return image
    
    #stroke width tranform function
    def stroke_width_tranform(self,image):
        try:

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            gray= np.pad(gray,(10,10),constant_values=255)
            IMAGE= gray.copy()
            blur = cv2.GaussianBlur(gray,(3,3),0)
            # adaptive threshold
            th5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            th6=th5.copy()

            #finding vertical projection of of threshold image

            img1 = swt.skew_correct(th5)
            img2 = swt.vertical_proj(img1)
            sum_x = swt.horizontal_proj(img2)
          

            height, width = th5.shape[:2]

            cordin=[]
            line=list()
            count=0
            j=0

            for i in range(len(sum_x[0])):
                line.append(count)
                count=[]
                if sum_x[0][i]==0:
                    cv2.line(th5,(j,0),(j,th5.shape[0]),(0,0,60))
                    cordin.append(j)

                j+=1
            letter=[]
            letter_start=[]

            kernel = np.ones((3,3), np.uint8) 
            for i in range(0,len(cordin)-1):
                if cordin[i]==cordin[i+1]-1:
                    continue
                else:
                    letter.append(cordin[i])
            letter1=[]
            for i in range(0,len(cordin)-1):
                if cordin[i-1]==cordin[i]-1:
                    continue
                else:
                    letter1.append(cordin[i])
            
                  
            
            del letter1[0]
            letter1.append(img1.shape[1])

            
            img_lst=[]
            ccc=[]
            c=0
            bb=[]
            img_lst=[]
            
            letter1b=letter.copy()
            letter2b=letter1.copy()
            try:
                for i in range(len(letter1)-2):
                    if letter[i+1]-letter1[i]<=1:
                        letter1b.pop(i+1)
                        letter2b.pop(i)
            except:
                pass
            # print(letter1b,letter2b,"Letters")
            


            count=[]
            i_count=False
            big_seg=[]
            for i in range(len(letter1b)):

                img_crop=gray[:,letter1b[i]:letter2b[i]]
                img=img_crop.copy()
                if img.shape[1]>23 and img.shape[1]<48:
                    if i_count==False:
                        img_lst.append(img)
                        i_count=False
                    
                if img.shape[1]<=23:
                    dis1=letter1b[i]-letter2b[i-1]
                    if i+1<len(letter1b):
                        dis2=letter1b[i+1]-letter2b[i]
                        if dis2<dis1:
                            img_con=gray[:,letter1b[i+1]:letter2b[i+1]]
                            img=cv2.hconcat([img,img_con])
                            img_lst.append(img)
                            i_count =True
                        else:

                            img_con=gray[:,letter1b[i-1]:letter2b[i-1]]
                            img=cv2.hconcat([img_con,img])
                            try:
                                img_lst.pop()
                            except:
                                pass
                            img_lst.append(img)
                    else:
                        img_con=gray[:,letter1b[i-1]:letter2b[i-1]]
                        img=cv2.hconcat([img_con,img])
                        try:
                            img_lst.pop()
                        except:
                            pass
                        img_lst.append(img)
                if img_crop.shape[1]>=48:
                    big_seg.append(img)
            final_list= img_lst+big_seg
            x_start_list = []
            x_end_list = []
            x_start_list.append(0)
            x_end_list.append(0)
            counter = 0
            
            return final_list, letter1b
        except:
            pass
            # print(Exception)
        
            
