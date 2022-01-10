import os
from core_logic.htr_word.htr_word_operation_manager import recognition_process
from endtoend_testing_report import output_jpn,output_english
import base64
import sys
import csv
import cv2
# directory = "/home/jayesh/Rawattech/HTR_project/endtoend_testing/TEST_DIR/"
# output_path = "/home/jayesh/Rawattech/HTR_project/English segment/English segment_upda/segmentation"
# operation_dict={'E':english(path=sys.argv[2]) , 'J':math(path=sys.argv[2])}

# def math():
#     seg_expected = {
#        "New folder":1,"NON_BLANK_non_unitsDIGITv2":1, "Non_Blank_v2":3, "image20201105":2
#         }
#     obj_expected= {"New folder":1,"NON_BLANK_non_unitsDIGITv2":1, "Non_Blank_v2":3, "image20201105":1}
#     folders = os.listdir(r"E:\phase2_data\phase2\math_sample/")
#     for x in folders:
#         image_name_lst = []
#         expected_seg_all, text_all, sp_image_all, Unit_image_all, segments_all, object_count_all, recog_all, conf_all = [],[],[],[],[],[],[],[]
#         img_dir = r"E:\phase2_data\phase2\math_sample/"+x+"/"
#         for i in os.listdir(img_dir):
#             try:
#                 # image = cv2.imread(img_dir+i)
#                 with open(img_dir+i, "rb") as img_file:
#                     my_string = base64.b64encode(img_file.read())
                
#                 recog,conf, text, sp_image, Unit_image, segments, object_count =recognition_process("M",my_string,i)
#                 image_name_lst.append(img_dir+i)
#                 expected_seg_all.append(seg_expected[x])
#                 text_all.append(text)
#                 sp_image_all.append(sp_image)
#                 Unit_image_all.append(Unit_image)
#                 segments_all.append(segments)
#                 object_count_all.append(object_count)
#                 recog_all.append(recog)
#                 conf_all.append(conf)
#             except Exception as e:
#                 print(e)
#         excel_name = x+".xlsx"
#         output_math(image_name_lst,segments_all,excel_name,x,obj_expected[x],object_count_all,text_all,sp_image_all,Unit_image_all,seg_expected[x],recog_all,conf_all)

#         print(image_name_lst,text_all, sp_image_all, Unit_image_all, segments_all)

def english(path):
    folders = os.listdir(path)
    image_name_lst = []
    expected_seg_all, text_all, sp_image_all, Unit_image_all, segments_all,  recog_all, conf_all,x_all = [],[],[],[],[],[],[] , []
    counter_excel = 0
    image_counter= 0
    for x in folders:
        img_dir = path+x+"/"
        for i in os.listdir(img_dir):
            print(i)
            # image = cv2.imread(img_dir+i)
            with open(img_dir+i, "rb") as img_file:
                    image = base64.b64encode(img_file.read())
            #expected_seg_all.append(len(x))
            try:
                Recognized_text_string, confidence_score_list, text, segments =recognition_process("E",image,i)
                image_name_lst.append(img_dir+i)
                text_all.append(text)
                x_all.append(x)
                #sp_image_all.append(sp_image)
                #Unit_image_all.append(Unit_image)
                segments_all.append(segments)
                #object_count_all.append(object_count)
                recog_all.append(Recognized_text_string)
                conf_all.append(confidence_score_list)
                image_counter+=1
                excel_name = "english_"+str(counter_excel)+".xlsx"
                if image_counter == 500:
                    output_english(image_name_lst,segments_all,excel_name,x_all,recog_all,conf_all,text_all)
                    expected_seg_all, text_all, sp_image_all, Unit_image_all, segments_all,  recog_all, conf_all ,x_all= [],[],[],[],[],[],[],[]
                    image_name_lst = []
                    counter_excel += 1
                    image_counter = 0
                elif os.listdir(path+folders[-1])[-1] == i:
                    output_english(image_name_lst,segments_all,excel_name,x_all,recog_all,conf_all,text_all)
            except Exception as e:
                print(e)
          


def jpn(path,reading_direction):
    count=1
    folders = os.listdir(path)
    image_name_lst = []
    Od_output_llist=[]
    lst1 = []
    lst2 = []
    expected_seg_all, text_all, sp_image_all, expected_all, segments_all,  recog_all, conf_all = [],[],[],[],[],[],[]
    for x in folders:
        img_dir = path+x+"/"
        with open(img_dir+"char.txt", "r",encoding="utf-8") as f:
            expected=f.read()
            expected=str(expected)
            len_expected=len(expected)
        for i in os.listdir(img_dir):
            print(i)
            with open(img_dir+i, "rb") as img_file:
                    my_string = base64.b64encode(img_file.read())
            
            #expected_seg_all.append(len(x))
            try:
                if i!="char.txt" and i!="Thumbs.db":

                    Recognized_text_string, confidence_score_list,text,sp_image, segments,Od_output,output2,prob2 =recognition_process("J",my_string,i,len_expected,reading_direction)
                    image_name_lst.append(img_dir+i)
                    text_all.append(text)
                    sp_image_all.append(sp_image)
                    expected_all.append(expected)
                    #Unit_image_all.append(Unit_image)
                    segments_all.append(segments)
                    Od_output_llist.append(Od_output)
                    # object_count_all.append(object_count)
                    recog_all.append(Recognized_text_string)
                    lst1.append(output2)
                    lst2.append(prob2)
                    conf_all.append(confidence_score_list)
                    excel_name = "japanese"+str(count)+".xlsx"
                    print(len(image_name_lst))
                    
                    
        
                    
                    

                    
                    
                    
                    if len(image_name_lst) ==500:
                        # print("hello")
                        output_jpn(image_name_lst,segments_all,excel_name,expected_all,sp_image_all,recog_all,conf_all,text_all,Od_output_llist,lst1,lst2)
                        image_name_lst = []
                        expected_seg_all, text_all, sp_image_all, expected_all, segments_all,  recog_all, conf_all ,Od_output_llist= [],[],[],[],[],[],[],[]
                        count+=1
                        lst1 = []
                        lst2 = []
                    '''elif os.listdir(path+folders[-1])[-1] == i:
                         output_jpn(image_name_lst,segments_all,excel_name,expected_all,sp_image_all,recog_all,conf_all,text_all,Od_output_llist)'''
            except Exception as e:
                image = cv2.imread(path+x+"/"+i)
                cv2.imwrite("error_images/"+x+"_"+i,image)
                with open("Exception_Jap_Images.csv","a") as f:
                    writer = csv.writer(f)
                    writer.writerow([x+"/"+i,e])
                    print (e)
    output_jpn(image_name_lst,segments_all,excel_name,expected_all,sp_image_all,recog_all,conf_all,text_all,Od_output_llist,lst1,lst2)
        
        
        # print(len(image_name_lst),len(text_all), len(expected_all),len(sp_image_all), len(segments_all))



         
# jpn()

jpn(sys.argv[1],sys.argv[2])

# df.to_csv("client.csv")




