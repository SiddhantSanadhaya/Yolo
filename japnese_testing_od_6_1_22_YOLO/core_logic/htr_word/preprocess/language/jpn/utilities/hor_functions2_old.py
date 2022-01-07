import numpy as np
import cv2
##import matplotlib.pyplot as plt
import os
import core_logic.htr_new.recognition_utils.operation.jap.segmentation.utilities.swt as swt
##from PIL import Image
import statistics 



diagnostics = False


def showimg(img, title=''):
        plt.imshow(img, cmap='gray')
        if title:
            plt.title(title)
        plt.show()


class utility:   
    #Read file function
    

    def read_img(self,image,mode):
        try:
            # image = cv2.imread('new_cropped4/'+img,)
            if(mode=="v"):
                image= cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return image
        except:
            print("Image does not exist.")
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
        # showimg(image)
    #runnning swt algo and taking output of swt
        # output=swt.main(image)
        # out,im=0,0
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray,(3,3),0)
        # ret,thresh1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY)
        # for i in range(image.shape[0]):
        #     for j in range(image.shape[1]):
        #         out= out+output[i,j]
        #         im = im+ thresh1[i,j]
        # awg_width= out/im;

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
    #thresh = 0.05*max(sum_x[0])
    #thresh = (np.mean(minima)-np.std(minima))
            #Segmenting chachters where  projection value is zero

            for i in range(len(sum_x[0])):
                line.append(count)
                count=[]
                if sum_x[0][i]==0:
            #count.append()
                    cv2.line(th5,(j,0),(j,th5.shape[0]),(0,0,60))
            #i+=1
                    cordin.append(j)

                j+=1
            # plt.imshow(th5)
            # plt.show()
            # cv2.imwrite("segment/"+x,th5);
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
            # print(letter,letter1)
            try:
                for i in range(len(letter1)-2):
                    # print(letter[i+1]-letter1[i])
                    if letter[i+1]-letter1[i]<=1:
                        letter1b.pop(i+1)
                        letter2b.pop(i)
            except:
                pass
            # print(letter1b,letter2b)
            


            # counter=1
            count=[]
            i_count=False
            big_seg=[]
            for i in range(len(letter1b)):

                #if (letter1[i]-letter[i]>10):
                # print(letter1b[i],letter2b[i])
                img_crop=gray[:,letter1b[i]:letter2b[i]]
                img=img_crop.copy()
                # print("img_shape",img.shape[1])
                if img.shape[1]>23 and img.shape[1]<48:
                    
                    
                    if i_count==False:
                        img_lst.append(img)
                        i_count=False
                    # count.append(counter)
                    
                if img.shape[1]<=23:
                    dis1=letter1b[i]-letter2b[i-1]
                    if i+1<len(letter1b):
                        dis2=letter1b[i+1]-letter2b[i]
                        # print("dis1:",dis1,"dis2",dis2)
                        if dis2<dis1:
                            img_con=gray[:,letter1b[i+1]:letter2b[i+1]]
                            img=cv2.hconcat([img,img_con])
                            img_lst.append(img)
                            i_count =True
                            # cv2.imwrite("char/"+"{}_".format(counter)+x,img);
                            # showimg(img,"concated")
                        else:

                            img_con=gray[:,letter1b[i-1]:letter2b[i-1]]
                            img=cv2.hconcat([img_con,img])
                            # cv2.imwrite("char/"+"{}_".format(counter)+x,img);
                            try:
                                img_lst.pop()
                            except:
                                pass
                            img_lst.append(img)
                            # showimg(img,"concated")
                    else:
                        img_con=gray[:,letter1b[i-1]:letter2b[i-1]]
                        img=cv2.hconcat([img_con,img])
                        try:
                            img_lst.pop()
                        except:
                            pass
                        img_lst.append(img)
                        # cv2.imwrite("char/"+"{}_".format(counter)+x,img);
                if img_crop.shape[1]>=48:
                    big_seg.append(img)
                    # cv2.imwrite("char/"+"{}_".format(counter)+x,img);

                        # showimg(img,"concated")
            # counter=1
            # for img in img_lst:
            #     # showimg(img,"img_list")
            #     cv2.imwrite("char/"+"{}_".format(counter)+x,img);
            #     count.append(counter)
            #     counter+=1
            # for img in big_seg:
            #     showimg(img,"big_seg")
            #     cv2.imwrite("char/"+"{}_".format(counter)+x,img);
            #     count.append(counter)
            #     counter+=1
            final_list= img_lst+big_seg
            for img in final_list:
                x,y,w,h= cv2.boundingRect(img)
                x = x_end_list[counter-1]+x_end_list[counter]
                x_start_list.append(x)
                x_end_list.append(w)
                x_end_list[0] = 32
                print(x_start_list)
                counter+=1
                    
                # b=letter1[i]-letter[i]
                # ccc.append(b)i+1
                # c=c+b;
                # bb.append(b)
            
            # img_lst1=[]
            
            return final_list
        except:
            print(Exception)
        
            
        # for k in range(len(img_lst)):
        #     print(k)

        #     thresh1= cv2.bitwise_not(img_lst[k])
        #     blur = cv2.GaussianBlur(thresh1,(3,3),0)
        #     ret,thresh1 = cv2.threshold(blur,100,255,cv2.THRESH_BINARY)
        #     showimg(thresh1,"ccccccccc")
            
        #     thinned = cv2.ximgproc.thinning(thresh1)

        #     count = 0
        #     arr_img = []
        #     ret, labels, stats, centroids = cv2.connectedComponentsWithStats(
        #             thresh1.astype(np.uint8), connectivity=4)
        #     # stats = sorted(stats, key=lambda x: x[0])
        #     for i in range(1, ret):
        #         if stats[i, cv2.CC_STAT_HEIGHT] >= 0.07*thresh1.shape[0]:
        #             count+=1
        #             new_img = np.zeros_like(thresh1)
        #             new_img[labels == i] = 1
        #             arr_img.append((new_img, stats[i]))
                

        #     arr_img = sorted(arr_img, key=lambda x: x[-1][0])


        #     result=[]
        #     reagion = []
        #     tmp = arr_img[0][0]
        #     bounding_left, bounding_right = arr_img[0][-1][0], arr_img[0][-1][0]+arr_img[0][-1][2]
        #     for i_img in range(len(arr_img) -1):
        #         x2 = arr_img[i_img][-1][0]+arr_img[i_img][-1][2]
        #         #print(x2)
        #         x1 = arr_img[i_img+1][-1][0]
        #         #print(x1)
        #         smaller_width = arr_img[i_img][-1][2] if arr_img[i_img][-1][2] < arr_img[i_img+1][-1][2] else arr_img[i_img+1][-1][2]
        #         #print(smaller_width)
        #         condition = (x2-x1) > 0.5*smaller_width
                
        #         print(condition)
        #         if condition:
        #             bounding_left = min(bounding_left, arr_img[i_img+1][-1][0])
        #             bounding_right = max(bounding_right, arr_img[i_img+1][-1][0]+arr_img[i_img+1][-1][2])
        #             tmp = tmp | arr_img[i_img+1][0]
        #         else:
        #             reagion.append((bounding_left, bounding_right))
        #             result.append(tmp)
        #             tmp = arr_img[i_img+1][0]
        #             bounding_left, bounding_right = arr_img[i_img+1][-1][0], arr_img[i_img+1][-1][0]+arr_img[i_img+1][-1][2]
        #         #if i_img == len(arr_img) - 2 and not condition:
        #         #    result.append(arr_img[-1][0])

             

        #     reagion.append((bounding_left, bounding_right))
        #     result.append(arr_img[-1][0])
        #     pad_img = np.zeros_like(thresh1)[:,:90]
        #     pad_img[:,50:60] = 1
        #     final_result = []
        #     char_list=[]
        #     print(len(result),"result")
        #     if len(result)<=1:
        #         img_lst1.append(img_lst[k])
        #     if len(result)>1:
        #         for i in range(len(result)):
        #         	print(img_lst[k][:,reagion[i][0]:reagion[i][1]].shape)
        #         	tmp_img = img_lst[k][:,reagion[i][0]:reagion[i][1]]
        #         	final_result.append(tmp_img)
        #         	char_list.append(cv2.bitwise_not(tmp_img))
        #         	final_result.append(pad_img)
        #     im_h = cv2.hconcat(final_result)
        #     #showimg(im_h)
        #     if len(char_list)!=0:
                
        #         for i in range(len(char_list)):
        #             im= cv2.bitwise_not(char_list[i])
        #             im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        #             #showimg(im)
        #             img_lst1.append(im)

               
        # print(len(img_lst1),"final")
        # width=0
        # lst=[]
        # if (mode=="v"):
        #     for i in range(len(img_lst1)):
        #         img_lst1[i]= cv2.rotate(img_lst1[i], cv2.ROTATE_90_CLOCKWISE)
        # for i in img_lst1:
        #     width=width+i.shape[1]
        #     print(i.shape)
        #     lst.append(i.shape[1])
        #     showimg(i)
        # width=width/len(img_lst1)
        # std_of_seg=statistics.stdev(lst)
        # print(std_of_seg)
        # print(width)

            

        
                    

            

# # plt.plot(sum_x[0])
# # plt.xlim([0, width])
# # plt.show()

#         #finding starting and ending part of each charchter segementation
        

       
                
       
               
      
#         if len(ccc)>1:
#             print("inside")
#             std_of_seg= np.std(ccc)
#             mean= np.mean(ccc)
#         else:
#             print("INSIDE")
#             std_of_seg=0
#             mean=image.shape[1]
#         MIN= min(ccc)
#         print(MIN)
#         max_width=round(std_of_seg+mean)
#         min_width=round(mean-std_of_seg)
#         print(mean,max_width,min_width)
       
        
        
        # for i in range(len(letter1)):
        #     if(letter[i]-letter1[i]>15):
        #         img_lst.append(img1[:,letter1[i]:letter[i]])
        #         b=letter[i]-letter1[i]
        #         ccc.append(b)
        #         c=c+b;
        #         bb.append(b)
                
        # print(len(img_lst))
               
            
            
            
            

    #     final_image_list=[]
    #     for i in img_lst1:
    #         if i.shape[1]<=width:
    #             final_image_list.append(i)
    #         if i.shape[1]>width:
    #             plt.imshow(i)
    #             plt.show()
    #             print(i.shape)                #labels, components = connected_components(swt) 
    #             thinned = cv2.ximgproc.thinning(cv2.bitwise_not(i))
    #             plt.imshow(thinned)
    #             plt.show()
    #             gray = cv2.cvtColor(thinned, cv2.COLOR_GRAY2BGR)
    #             img12=cv2.bitwise_not(gray)
    #             img13=cv2.bitwise_not(img12)
                
    #             kernel = np.ones((3,3), np.uint8) 
    #             img_erosion = cv2.erode(i, kernel, iterations=1) 

               
    #             #img11=cv2.bitwise_not(img11)
                 
    #     # blur
                
                
                
    #             ximg1 = swt.skew_correct(i)
    #             ximg2 = swt.vertical_proj(ximg1)
    #             xsum_x = swt.horizontal_proj(ximg2)
    #             xsum_x=swt.smooth(xsum_x[0],1)
               
    #             #mean= mean-std_of_seg
    #             peak, _ = find_peaks(xsum_x,distance=(width+std_of_seg)/2)
    #             plt.plot(xsum_x)
    #             plt.plot(peak, xsum_x[peak], "x")
    #             plt.show()
    #             print(peak)
    #             img14=cv2.bitwise_not(img13)
                
    #             for l in peak: 
    #                 print(l)      
    #                 cv2.line(img13,(l,0),(l,i.shape[0]),(255,255,255))
                    
    #             plt.imshow(img13,cmap='gray')
    #             plt.show()
    #             hh=int(img13.shape[0]/2)
    #             print(img13.shape,img12.shape)
    #             path=0
    #             points = []
    #             while(path!=1):
    #                 path=distance.main(img13,img12,peak[0],peak[1],hh,hh)
    #                 print(path)
    #                 if(path==1):
    #                     mid=int(round(img12.shape[1]/2))
    #                     center=[0,mid]
    #                     break
    #                 mid=int(round(len(path)/2))+1
    #                 print(mid)
    #                 print(path[mid])
    #                 center= path[mid]
    #                 print(center)
    #                 points.append(center)
    #                 img13[center[0],center[1]]=[0,0,0]
    #                 img12[center[0],center[1]]=[255,255,255]
    #                 img14[center[0],center[1]]=[255,255,255]

    #                 plt.imshow(img13,cmap='gray')
    #                 plt.show()
                
    #             sum_b=0
                
    #             path_short=short.main(img14,center[1],center[1],0,img12.shape[0])
                
    #             # cv2.line(img12,(center_y,0),(center_y,img12.shape[0]),(0,0,60))
    #             # plt.imshow(img12,cmap="gray")
    #             # plt.show()
    #             # cv2.line(img14,(points[0][1],0),(points[0][1],img14.shape[0]),(0,0,60))
    #             # for i in range(len(points)-1):
    #             #     print(points[i][1],points[i][0],points[i+1][1],points[i+1][0])
    #             #     cv2.line(img14,(points[i][1],points[i][0]),(points[i+1][1],points[i+1][0]),(0,0,60))
    #             # cv2.line(img14,(center[1],center[0]),(center[1],img14.shape[0]),(0,0,60))

    #             line_images = []
    #             img15=img14.copy()
                
    #             plt.imshow(img14)
    #             plt.show()
                
               
                
                              
    #             #edges = cv2.Canny(img12,200,300)
    #             #plt.imshow(edges,cmap="gray")
    #             #plt.show()s
    #             img16 = np.ones((img15.shape[0],img15.shape[1],3), np.uint8)
    #             img17 = np.ones((img15.shape[0],img15.shape[1],3), np.uint8)
    #             red= False
    #             list2=[]
    #             for i in range(img15.shape[0]):
    #                 for j in range(img15.shape[1]):
    #                     if img15[i,j,0]==255 and img15[i,j,1]==0 and img15[i,j,2]==0:
    #                         red=True
    #                         img16[i,j,0]=255
    #                         img16[i,j,1]=255
    #                         img16[i,j,2]=255

    #                     if(red==False):
    #                         print(i,j)
    #                         img16[i,j,0]=img15[i,j,0]
    #                         img16[i,j,1]=img15[i,j,1]
    #                         img16[i,j,2]=img15[i,j,2]
    #                     else:
    #                         img16[i,j,0]=255
    #                         img16[i,j,1]=255
    #                         img16[i,j,2]=255


                        
    #                     if(red==True):
    #                         img17[i,j,0]=img15[i,j,0]
    #                         img17[i,j,1]=img15[i,j,1]
    #                         img17[i,j,2]=img15[i,j,2]
    #                         if img17[i,j,0]==255 and img17[i,j,1]==0 and img17[i,j,2]==0:
                            
    #                             img17[i,j,0]=255
    #                             img17[i,j,1]=255
    #                             img17[i,j,2]=255
    #                     else:
    #                         img17[i,j,0]=255
    #                         img17[i,j,1]=255
    #                         img17[i,j,2]=255

    #                     if(j==img15.shape[1]-1):
    #                         red=False
    #             plt.imshow(img16)
    #             plt.show()
    #             plt.imshow(img17)
    #             plt.show()
    #             list2.append(img16)
    #             list2.append(img17)
    #             for l in list2:

    #                 black = np.zeros((l.shape[0],1),np.uint8)
    #                 black = cv2.bitwise_not(black)
    #                 #l = cv2.rotate(l, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # #blur = cv2.bilateralFilter(gray,9,75,75)
    #                 gray = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
    #                 #blur = cv2.medianBlur(gray, 3)
    #                 ret,th4 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # #th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #                 img_erosion = cv2.erode(th4, kernel, iterations=1) 
    #                 res = cv2.matchTemplate(black,img_erosion,cv2.TM_SQDIFF_NORMED)
    #                 for j in range(res.shape[1]):
    #                     res[0][j] = round(res[0][j],2)
    #                 lst=[]
    #                 for j in range(len(res[0])):
    #                     if res[0][j]!=0:
    #                         lst.append(j)
    #                 img = img_erosion[:,min(lst):max(lst)]
    #                 #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    #                 final_image_list.append(img)
    #     print(len(final_image_list))

       

   

   
      
           