import numpy as np
import cv2
##import matplotlib.pyplot as plt
import os
import utilities.swt as swt
import utilities.thinig_algo as thin
import utilities.path as distance
import utilities.shortest as short
import utilities.thinning as thinning
##from PIL import Image
import imutils
import math
from .id_colors import build_colormap
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max
import statistics 
from collections import defaultdict
import hashlib
from typing import TypeVar, NamedTuple, List, Optional, Tuple
import math
import scipy.ndimage as ndimage
import sys
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy.sparse, scipy.spatial

from scipy.signal import find_peaks
import time


diagnostics = False


def showimg(img, title=''):
        plt.imshow(img, cmap='gray')
        if title:
            plt.title(title)
        plt.show()
def fitToSize(thresh1):
    
    mask = thresh1 > 0
    coords = np.argwhere(mask)

    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
    cropped = thresh1[x0:x1,y0:y1]
    return cropped
def baselines(letter2, upoints, dpoints,h,w):
##-------------------------Creating upper baseline-------------------------------##
    colu = []
    for i in range(len(upoints)):
        colu.append(upoints[i][1])
    
    maxyu = max(colu)
    minyu = min(colu)
    avgu = (maxyu + minyu) // 2
    meanu = np.around(np.mean(colu)).astype(int)
    print('Upper:: Max, min, avg, mean:: ',maxyu, minyu, avgu, meanu)
    
##-------------------------------------------------------------------------------##
##-------------------------Creating lower baseline process 1--------------------------##
    cold = []
    for i in range(len(dpoints)):
        cold.append(dpoints[i][1])
    
    maxyd = max(cold)
    minyd = min(cold)
    avgd = (maxyd + minyd) // 2
    meand = np.around(np.mean(cold)).astype(int)
    print('Lower:: Max, min, avg, mean:: ',maxyd, minyd, avgd, meand)
    
##-------------------------------------------------------------------------------##
##-------------------------Creating lower baseline process 2---------------------------##
    cn = []
    count = 0

    for i in range(h):
        for j in range(w):
            if(letter2[i,j] == 255):
                count+=1
        if(count != 0):
            cn.append(count)
            count = 0    
    maxindex = cn.index(max(cn))
    print('Max pixels at: ',meanu,meand)
    
##------------------Printing upper and lower baselines-----------------------------##
    
    #cv2.line(letter2,(0,meanu),(w,meanu),(255,0,0),2)
    lb = 0
    if(maxindex > meand):
        lb = maxindex
        #cv2.line(letter2,(0,maxindex),(w,maxindex),(0,0,0),2)
    else:
       lb = meand
       #cv2.line(letter2,(0,meand),(w,meand),(255,0,0),2)
    #cv2.line(letter2,(0,int(meand-maxindex/2)),(w,int(meand-maxindex/2)),(255,0,0),2)
    avg=int((meand+meanu)/2)
    print(meand,meanu,avg)
    showimg(letter2)
    return meand,maxindex,avg
def findCapPoints(img):
    cpoints=[]
    dpoints=[]
    for i in range(img.shape[1]):
        col = img[:,i:i+1]
        k = col.shape[0]
        while k > 0:
            if col[k-1]==255:
                dpoints.append((i,k))
                break
            k-=1
        
        for j in range(col.shape[0]):
            if col[j]==255:
                cpoints.append((i,j))
                break
    return cpoints,dpoints


class utility:   
    #Read file function
    

    def read_img(self,img):
    	try:
    		image = cv2.imread('segmentation/'+img,)
    		black = np.zeros((image.shape[1],1),np.uint8)
    		black = cv2.bitwise_not(black)

    		image= cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    		#blur = cv2.medianBlur(gray, 3)
    		ret,th4 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    		#th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    		img_erosion = cv2.erode(th4, (3,3), iterations=1)
    		res = cv2.matchTemplate(black,img_erosion,cv2.TM_SQDIFF_NORMED)
    		for j in range(res.shape[1]):
    			res[0][j] = round(res[0][j],2)
    		lst=[]
    		for j in range(len(res[0])):
    			if res[0][j]!=0:
    				lst.append(j)
    		img = image[:,min(lst):max(lst)]
    		image= cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    		return image
    	except:
    		print("Image does not exist.")
    		sys.exit()
    #preprocessing function        
    def preprocessing(self,image):
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
        #runnning swt algo and taking output of swt
        output=swt.main(image)
        out,im=0,0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(3,3),0)
        ret,thresh1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                out= out+output[i,j]
                im = im+ thresh1[i,j]
        awg_width= out/im;

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        gray= np.pad(gray,(10,10),constant_values=255)
        IMAGE= gray.copy()
        blur = cv2.GaussianBlur(gray,(3,3),0)
        # adaptive threshold
        th5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        th6=th5.copy()

        #finding vertical projection of of threshold image

        img1 = swt.skew_correct(th5)
        showimg(img1,"skew_correct image")
        img2 = swt.vertical_proj(img1)
        sum_x = swt.horizontal_proj(img2)



      

        height, width = th5.shape[:2]

        cordin=[]
        line=list()
        count=0
        j=0
#thresh = 0.05*max(sum_x[0])
#thresh = (np.mean(minima)-np.std(minima))
        #Segmenting chachters where  projection value is zero

        for i in range(len(sum_x[0])):
            line.append(count)
            count=[]
            if sum_x[0][i]<=awg_width.any():
        #count.append()
                cv2.line(img1,(j,0),(j,th5.shape[0]),(0,0,60))
        #i+=1
                cordin.append(j)

            j+=1
        showimg(img1,"segmented image")
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
        
        for i in range(len(letter)):

            #if (letter1[i]-letter[i]>10):
            img_lst.append(img1[:,letter[i]+1:letter1[i]])
            b=letter1[i]-letter[i]
            ccc.append(b)
            c=c+b;
            bb.append(b)
        print(len(img_lst))
        img_lst1=[]
        counter=0
        for k in range(len(img_lst)):
            print(k)
            myself= img_lst[0]
            showimg(img_lst[k],"myimage")

            thresh1= cv2.bitwise_not(img_lst[k])
            blur = cv2.GaussianBlur(thresh1,(3,3),0)
            ret,thresh1 = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
            
            thinned = cv2.ximgproc.thinning(thresh1)

            count = 0
            arr_img = []
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    thresh1.astype(np.uint8), connectivity=8)
            # stats = sorted(stats, key=lambda x: x[0])
            for i in range(1, ret):
                if stats[i, cv2.CC_STAT_HEIGHT] >= 0.07*thresh1.shape[0]:
                    count+=1
                    new_img = np.zeros_like(thresh1)
                    new_img[labels == i] = 1
                    arr_img.append((new_img, stats[i]))
                

            arr_img = sorted(arr_img, key=lambda x: x[-1][0])


            result=[]
            reagion = []
            tmp = arr_img[0][0]
            bounding_left, bounding_right = arr_img[0][-1][0], arr_img[0][-1][0]+arr_img[0][-1][2]
            for i_img in range(len(arr_img) -1):
                x2 = arr_img[i_img][-1][0]+arr_img[i_img][-1][2]
                #print(x2)
                x1 = arr_img[i_img+1][-1][0]
                #print(x1)
                smaller_width = arr_img[i_img][-1][2] if arr_img[i_img][-1][2] < arr_img[i_img+1][-1][2] else arr_img[i_img+1][-1][2]
                #print(smaller_width)
                condition = (x2-x1) > 0.5*smaller_width
                
                print(condition)
                if condition:
                    bounding_left = min(bounding_left, arr_img[i_img+1][-1][0])
                    bounding_right = max(bounding_right, arr_img[i_img+1][-1][0]+arr_img[i_img+1][-1][2])
                    tmp = tmp | arr_img[i_img+1][0]
                else:
                    reagion.append((bounding_left, bounding_right))
                    result.append(tmp)
                    tmp = arr_img[i_img+1][0]
                    bounding_left, bounding_right = arr_img[i_img+1][-1][0], arr_img[i_img+1][-1][0]+arr_img[i_img+1][-1][2]
                #if i_img == len(arr_img) - 2 and not condition:
                #    result.append(arr_img[-1][0])

             

            reagion.append((bounding_left, bounding_right))
            result.append(arr_img[-1][0])
            pad_img = np.zeros_like(thresh1)[:,:90]
            pad_img[:,50:60] = 1
            final_result = []
            char_list=[]
            print(len(result),"result")
            if len(result)<=1:
                img_lst1.append(img_lst[k])
            if len(result)>1:
                for i in range(len(result)):
                	print(img_lst[k][:,reagion[i][0]:reagion[i][1]].shape)
                	tmp_img = img_lst[k][:,reagion[i][0]:reagion[i][1]]
                	final_result.append(tmp_img)
                	char_list.append(cv2.bitwise_not(tmp_img))
                	final_result.append(pad_img)
            im_h = cv2.hconcat(final_result)
            #showimg(im_h)
            if len(char_list)!=0:
                
                for i in range(len(char_list)):
                    im= cv2.bitwise_not(char_list[i])
                
                    img_lst1.append(im)

               
        print(len(img_lst1),"final")
        width=0
        lst=[]
        for i in img_lst1:
            width=width+i.shape[1]
            print(i.shape)
            lst.append(i.shape[1])
            showimg(i)
        width=width/len(img_lst1)
        #std_of_seg=statistics.stdev(lst)
        #print(std_of_seg)
        print(width)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #gray= cv2.bitwise_not(gray)
        th, letterGray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        letterGray = swt.skew_correct(letterGray)
        letterGray=cv2.bitwise_not(letterGray)
        letterGray = fitToSize(letterGray)
        letter2 = letterGray.copy()
        letterGray = cv2.dilate(letterGray,None,iterations = 4)
        upoints, dpoints=findCapPoints(letterGray)
        maxindex,minindex,avg_index = baselines(letter2, upoints, dpoints,letter2.shape[0],letter2.shape[1])
        img= letter2[0:avg_index,:]
        img1=cv2.flip(letter2,0)
        upoints, dpoints=findCapPoints(img1)
        maxindex,minindex,avg_index = baselines(img1, upoints, dpoints,letter2.shape[0],letter2.shape[1])
        imgx1= img1[0:avg_index,:]
        # showimg(img)
        # showimg(img1)
        #print(result)
        img= cv2.bitwise_not(img)
        # showimg(img)
        img1 = swt.skew_correct(img)
        img2 = swt.vertical_proj(img1)
        sum_x1 = swt.horizontal_proj(img2)
        img= cv2.bitwise_not(imgx1)
        sum_x2 = swt.horizontal_proj(img)

        cordin=[]
        line=list()
        count=0
        j=0

        for i in range(len(sum_x1[0])):
            line.append(count)
            count=[]
            if sum_x1[0][i]==0:
            	if sum_x1[0][i]==0 :
            		if sum_x1[0][i]==0:
            			cv2.line(letter2,(j,0),(j,letter2.shape[0]),(255,255,255))
            			cordin.append(j)

            j+=1
        plt.imshow(letter2,cmap="gray")
        plt.show()
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

       	letter.insert(0,0)
        letter1.append(img1.shape[1])

        
        img_lst=[]
        ccc=[]
        c=0
        bb=[]
        img_lst=[]
        
        for i in range(len(letter)):

            #if (letter1[i]-letter[i]>10):
            img_lst.append(letterGray[:,letter[i]:letter1[i]])
            b=letter1[i]-letter[i]
            ccc.append(b)
            c=c+b;
            #showimg(letter2[:,letter[i]+1:letter1[i]])
            bb.append(b)
        print(ccc,c/len(ccc),np.std(ccc))
        mean=c/len(ccc)
        count0=0
        img_lst2=[]
        min_width= mean-int(np.std(ccc))
        # for i in range(len(img_lst)):
        	
        # 	if ccc[count0]<min_width:
        # 		print(ccc[count0])
        # 		img=cv2.hconcat([img_lst[count0][:,0:img_lst[count0].shape[1]-1],img_lst[count0+1]])
        # 		showimg(img)
        # 		img_lst2.append(img)
        # 		count0+=1
        # 		i+=1
        # 	else:
        # 		print(ccc[count0])
        # 		img_lst2.append(img_lst[count0])
        # 		count0+=1
        # counter=30
        # for i in range(len(sum_x1[0])):
        #     cv2.line(myself,(counter,0),(counter,letter2.shape[0]),(0,0,0))
        #     counter+=30
        # showimg(myself,"cropped")
                        
        


            
            
            
            

        

       

   

   
      
           