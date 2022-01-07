# assential libraries
import os
import cv2
import numpy as np

def Segmentation_Japanese(img,image_type):
    
    segment=[]
    all_segments = []  
    image_type=image_type.upper()
    if image_type=="V":
        img=cv2.transpose(img)
    # rough pre_processing
    grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(grayimage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    (h, w)=thresh1.shape #Return height and width

    ##cv2.imshow('img1',thresh1)
    vertical_hist = img.shape[0] - np.sum(thresh1,axis=0,keepdims=True)/255

    # to make image to np array 
    ary = np.array(255-thresh1)
    # to get width of the char in the image
    width_list = []

    column_count = 0
    for pixel_count in vertical_hist[0]:
      if pixel_count > 0:
        column_count = column_count+1
        
      elif (column_count == 0) and (pixel_count ==0):
        column_count = 0
        
      elif (column_count > 0) and (pixel_count ==0):
        width_list.append(column_count)


    # to get unique values in width list
    unique_width_list =[]
    for x in width_list:
            if x not in unique_width_list:
                unique_width_list.append(x)


    ## to get final width list

    final_width_list = []
    j=0
    for i in unique_width_list:
      final_width_list.append(i-j)
      j=i

    ##  to get image segments without width conditions
    append_count = 0
    segment_array = []

    second_last=0
    for col in ary.T:
      last = (sum(col))/255
      if last >0:
        
        
        segment_array.append(col)
        
      elif (last ==0) and (second_last >0):
          pass

         
        
      else:
        pass
      second_last = last
      


    
    char_list=[]
    final_segment = []
    j = 0
    if all(vertical_hist[0]) == True:
        pass
        
    else:
        average_char_width = sum(final_width_list)/len(final_width_list)

    for char in unique_width_list:  
        if (char-j) < (average_char_width*0.9):
            continue

        else:
            final_segment.extend(np.array(segment_array)[j:char])
            all_segments.append(np.array(segment_array)[j:char].T)
            segment_array = np.array(segment_array)
            # cv2.line(segment_array , (0, char), (h, char), (255, 255, 255),2)
            j = char
            char_list.append(char)
    for i in all_segments:
        if image_type=="V":
            i=cv2.transpose(i)
            im=cv2.bitwise_not(i)
        else:
            im=cv2.bitwise_not(i)
        segment.append(im)
    # print(len(segment))
    return segment,char_list












# def load_images_from_folder(folder):
#     images = []
#     # path = r"C:\Users\yogendra\Desktop\my_segmentations_1"   # path where to save
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder,filename))
#         if img is None:
#             print('blank image')
#         else:
#             image_seg = Segmentation_Japanese(img)
#             for im in image_seg:
#                 cv2.imshow('img_segment_final',im)
#                 cv2.waitKey(0) 
#             # if image_seg is not None:
#                 # cv2.imwrite(os.path.join(path, filename), image_seg)

# folder = r"E:\rawattech\core_logic (1) (1)\output_dir_Jap"      # path where to get images


# load_images_from_folder(folder)
# print('Finish  :')

