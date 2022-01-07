import xlsxwriter
import os
import sys
# import xlrd
import matplotlib.pyplot as plt
import cv2


def output_math(orig,pre,excel_name,fld_name,expected_obj,actual_obj,text_image,special_image,unit_image,excepted_segment,recog,conf):
	workbook = xlsxwriter.Workbook(excel_name)
	worksheet1 = workbook.add_worksheet("result")
	# loc = "english_result.xlsx"
	# wb = xlrd.open_workbook(loc)
	# sheet = wb.sheet_by_index(0)

	# lst=[]
	# lst1=[]
	# lst2=[]
	# expected=[]
	# for i in range(1,sheet.nrows):
	# 	lst1.append(sheet.cell_value(i, 6))
	# 	lst.append(sheet.cell_value(i, 4))
	# 	lst2.append(sheet.cell_value(i, 1))
	# 	expected.append(sheet.cell_value(i, 3))
	# print(len(lst1),len(pre))
	# worksheet1.set_column('A:I',20)



	worksheet1.set_default_row(len(pre))

	cell_format = workbook.add_format({'bold': True,})
	cell_format1 = workbook.add_format({'bold': True,})
	cell_format1.set_bg_color('red')
	cell_format2 = workbook.add_format({'bold': True,})
	cell_format1.set_bg_color('green')
	worksheet1.set_column(0,40)
	worksheet1.set_column("A:U",18)
	worksheet1.set_row(0,20,cell_format)
	worksheet1.write('A1', 'image',)
	worksheet1.write('B1','image name')
	worksheet1.write('C1','o_d_text_image' )
	worksheet1.write('D1','o_d_sp1_image' )
	worksheet1.write('E1','o_d_sp2_image' )
	worksheet1.write('F1','o_d_unit_image' )
	worksheet1.write('G1','o_d_actual_count' )
	worksheet1.write('H1','o_d_exp_count' )
	worksheet1.write('I1','o_d_Result' )
	worksheet1.write('J1', 'seg1')
	worksheet1.write('K1', 'seg2')
	worksheet1.write('L1', 'seg3')
	worksheet1.write('M1', 'seg4')
	worksheet1.write('N1', 'seg5')
	worksheet1.write('O1', 'seg result')
	worksheet1.write('P1', 'seg exp')
	worksheet1.write('Q1','Seg_Result' )
	worksheet1.write('R1', 'recog')
	worksheet1.write('S1', 'word')
	worksheet1.write('T1', 'confidence')
	worksheet1.write('U1', 'result')
	



	#worksheet.set_header(["image","name","reconized","accuracy"])
	row=1
	images_col= 0
	image_name_col=1
	text_img_col=2
	sp_img_col=3
	unit_img_col=5
	actual_obj_col=6
	exp_obj_col=7
	o_d_result = 8
	seg_res_col=14
	exp_seg_col=15
	recog_col=17
	word_col=18
	conf_col=19
	result_col=20
	
	
	

	pre_col=2
	expected_col=3

	
	Recognised_accuracy_col=5
	result_col=3
	Previous_col=7
	delta_col=8


	for i in range(len(orig)):
		if i<5000:
			try:
				
				worksheet1.set_row(row,40)
				worksheet1.insert_image(row,
				                    images_col,
				                   orig[i],
				                     {'x_scale': 0.3, 'y_scale': 0.3,
				                            'x_offset':5,'y_offset':5,
				                           'positioning': 1})
				worksheet1.write(row,image_name_col,orig[i])
				worksheet1.write(row,exp_obj_col,len(expected_obj))
				worksheet1.write(row,actual_obj_col,actual_obj[i])
				worksheet1.insert_image(row,
				                    text_img_col,
				                   text_image[i],
				                     {'x_scale': 0.3, 'y_scale': 0.3,
				                            'x_offset':5,'y_offset':5,
				                           'positioning': 1})
				if len(actual_obj[i])==expected_obj:
					worksheet1.write(row,o_d_result,"1")

				if len(special_image[i]) != 0:
					for j in special_image[i]:
						# print("image_name",pre[i][l],orig[i])
						worksheet1.insert_image(row,
						                    sp_img_col,
						                    j,
						                     {'x_scale': .7, 'y_scale': .7,
						                            'x_offset':0,'y_offset':0,
						                            'positioning': 1})
						sp_img_col+=1
				worksheet1.insert_image(row,
				                    unit_img_col,
				                   unit_image[i],
				                     {'x_scale': 0.5, 'y_scale': 0.5,
				                            'x_offset':5,'y_offset':5,
				                           'positioning': 1})
				
				worksheet1.write(row,exp_seg_col,excepted_segment)
				worksheet1.write(row,seg_res_col,len(pre[i]))
				worksheet1.write(row,word_col,fld_name)
				worksheet1.write(row,recog_col,recog[i])
				worksheet1.write(row,conf_col,conf[i])
				seg_img_col=9
				for l in range(len(pre[i])):
					# print("image_name",pre[i][l],orig[i])
					worksheet1.insert_image(row,
					                    seg_img_col,
					                    pre[i][l],
					                     {'x_scale': .7, 'y_scale': .7,
					                            'x_offset':0,'y_offset':0,
					                            'positioning': 1})
					seg_img_col+=1
			except Exception as e: 
				print(e)
			row+=1

	workbook.close()



def output_english(orig,pre,excel_name,fld_name,recog_all,conf_all,text_all):
	workbook = xlsxwriter.Workbook("english_excels/"+excel_name)
	worksheet1 = workbook.add_worksheet("result")
	# loc = "english_result.xlsx"
	# wb = xlrd.open_workbook(loc)
	# sheet = wb.sheet_by_index(0)

	# lst=[]
	# lst1=[]
	# lst2=[]
	# expected=[]
	# for i in range(1,sheet.nrows):
	# 	lst1.append(sheet.cell_value(i, 6))
	# 	lst.append(sheet.cell_value(i, 4))
	# 	lst2.append(sheet.cell_value(i, 1))
	# 	expected.append(sheet.cell_value(i, 3))
	# print(len(lst1),len(pre))
	# worksheet1.set_column('A:I',20)



	worksheet1.set_default_row(len(pre))

	cell_format = workbook.add_format({'bold': True,})
	cell_format1 = workbook.add_format({'bold': True,})
	cell_format1.set_bg_color('red')
	cell_format2 = workbook.add_format({'bold': True,})
	cell_format1.set_bg_color('green')
	worksheet1.set_column(0,40)
	worksheet1.set_row(0,20,cell_format)
	worksheet1.write('A1', 'image',)
	worksheet1.write('B1','image name' )
	worksheet1.write('C1','object_detection_image' )
	worksheet1.write('D1', 'segment expected')
	worksheet1.write('E1', 'segment result')
	worksheet1.write('F1', 'expected_result')
	worksheet1.write('G1', 'word')
	worksheet1.write('H1', 'recognized')
	worksheet1.write('I1', 'confidence')
	worksheet1.write('J1', 'recognise_result')
	worksheet1.write('K1', 'seg1')
	worksheet1.write('L1', 'seg2')
	worksheet1.write('M1', 'seg3')



	#worksheet.set_header(["image","name","reconized","accuracy"])
	row=1
	images_col= 0

	image_name_col=1
	#exp_obj_col=2
	#actual_obj_col=3
	text_img_col=2
	#sp_img_col=5
	#unit_img_col=7
	exp_seg_col=3
	seg_res_col=4
	word_col=6
	result_col=5
	recog_col=7
	conf_col=8
	recognise_result = 9
	

	#pre_col=2
	#expected_col=3

	
	#Recognised_accuracy_col=5
	#result_col=3
	Previous_col=7
	delta_col=8


	for i in range(len(orig)):
		if i<1000:
			try:
				
				worksheet1.set_row(row,15)
				worksheet1.insert_image(row,
				                    images_col,
				                   orig[i],
				                     {'x_scale': 0.3, 'y_scale': 0.3,
				                            'x_offset':5,'y_offset':5,
				                           'positioning': 1})
				worksheet1.write(row,image_name_col,orig[i])
				worksheet1.insert_image(row,
				                    text_img_col,
				                   text_all[i],
				                     {'x_scale': 0.3, 'y_scale': 0.3,
				                            'x_offset':5,'y_offset':5,
				                           'positioning': 1})
				
				
				worksheet1.write(row,exp_seg_col,len(fld_name[i]))
				worksheet1.write(row,seg_res_col,len(pre[i]))
				worksheet1.write(row,word_col,fld_name[i])
				
				r = False
				if len(fld_name[i]) == len(pre[i]):
					r = True
				else:
					r = False
				worksheet1.write(row,result_col,r)

				worksheet1.write(row,recog_col,recog_all[i])
				worksheet1.write(row,conf_col,conf_all[i])
				# comparing results with foldername with recognise col in lowercase
				re_acc=False
				if recog_all[i].lower() == fld_name[i].lower():
					re_acc =True
				else:
					re_acc = False
				worksheet1.write(row,recognise_result,re_acc)



				seg_img_col=10
				for l in range(len(pre[i])):
					# print("image_name",pre[i][l],orig[i])
					worksheet1.insert_image(row,
					                    seg_img_col,
					                    pre[i][l],
					                     {'x_scale': .7, 'y_scale': .7,
					                            'x_offset':0,'y_offset':0,
					                            'positioning': 1})
					seg_img_col+=1
			except FileCreateError:
				pass
			except Exception as e: 
				print(e)
			row+=1

	workbook.close()

def output_jpn(orig,pre,excel_name,fld_name,sp,recog_all,conf_all,text_all,Od_output,lst1,lst2):
	workbook = xlsxwriter.Workbook(os.path.join("jap_excels",excel_name))
	worksheet1 = workbook.add_worksheet("result")
	# loc = "english_result.xlsx"
	# wb = xlrd.open_workbook(loc)
	# sheet = wb.sheet_by_index(0)

	# lst=[]
	# lst1=[]
	# lst2=[]
	# expected=[]
	# for i in range(1,sheet.nrows):
	# 	lst1.append(sheet.cell_value(i, 6))
	# 	lst.append(sheet.cell_value(i, 4))
	# 	lst2.append(sheet.cell_value(i, 1))
	# 	expected.append(sheet.cell_value(i, 3))
	# print(len(lst1),len(pre))
	# worksheet1.set_column('A:I',20)



	worksheet1.set_default_row(len(pre))

	cell_format = workbook.add_format({'bold': True,})
	cell_format1 = workbook.add_format({'bold': True,})
	cell_format1.set_bg_color('red')
	cell_format2 = workbook.add_format({'bold': True,})
	cell_format1.set_bg_color('green')
	worksheet1.set_column("A:C",18)
	worksheet1.set_column("G:L",6)
	worksheet1.set_column("D:O",10)
	worksheet1.set_row(0,20,cell_format)
	worksheet1.write('A1', 'image',)
	worksheet1.write('B1','image name' )
	worksheet1.write('C1','object_detection_image')
	worksheet1.write('D1', 'segment expected')
	worksheet1.write('E1', 'segment output')
	worksheet1.write('F1', 'segment result')
	worksheet1.write('G1', 'seg1')
	worksheet1.write('H1', 'seg2')
	worksheet1.write('I1', 'seg3')
	worksheet1.write('J1', 'seg4')
	worksheet1.write('K1', 'seg5')
	worksheet1.write('L1', 'seg6')
	worksheet1.write('M1', 'expected')
	worksheet1.write('N1', 'recognized')
	worksheet1.write('O1','wrong str')
	worksheet1.write('P1', 'confidence')
	worksheet1.write('Q1', 'Final result')
	worksheet1.write('R1', 'second result')
	worksheet1.write('S1', 'second conf')
	worksheet1.write('T1', 'second recog')	
	



	#worksheet.set_header(["image","name","reconized","accuracy"])
	row=1
	images_col= 0

	image_name_col=1
	# exp_obj_col=2
	#actual_obj_col=3
	text_img_col=2
	# sp_img_col=6
	#unit_img_col=7
	exp_seg_col=3
	seg_res_col=4
	result_col=5
	
	
	

	#pre_col=2
	#expected_col=3

	
	# Recognised_accuracy_col=5
	# result_col=3
	# Previous_col=7
	# delta_col=8


	for i in range(len(orig)):
		if i<5000:
			try:
				
				worksheet1.set_row(row,40)
				print("enterd in cell")
				worksheet1.insert_image(row,
				                    images_col,
				                   orig[i],
				                     {'x_scale': 0.3, 'y_scale': 0.3,
				                            'x_offset':5,'y_offset':5,
				                           'positioning': 1})
				worksheet1.write(row,image_name_col,orig[i])
				worksheet1.insert_image(row,
				                    text_img_col,
				                   Od_output[i],
				                     {'x_scale': 0.3, 'y_scale': 0.3,
				                            'x_offset':5,'y_offset':5,
				                           'positioning': 1})
				
				
				worksheet1.write(row,exp_seg_col,len(fld_name[i]))
				worksheet1.write(row,seg_res_col,len(pre[i]))
				if len(fld_name[i])==len(pre[i]):
					worksheet1.write(row,result_col,"1")
				else:
					worksheet1.write(row,result_col,"0")	
				seg_img_col=6
				for l in range(len(pre[i])):
					# print("image_name",pre[i][l],orig[i])
					worksheet1.insert_image(row,
					                    seg_img_col,
					                    pre[i][l],
					                     {'x_scale': .7, 'y_scale': .7,
					                            'x_offset':0,'y_offset':0,
					                            'positioning': 1})
					seg_img_col+=1
				word_col=12
				# print(word_col)
				recog_col=word_col+1
				wrong_str = recog_col+1
				conf_col=wrong_str+1
				
				rresult_col=conf_col+1
				o2 = rresult_col +1
				co2= o2 + 1
				recog_sec = co2+1
				worksheet1.write(row,word_col,fld_name[i])
				worksheet1.write(row,recog_col,recog_all[i])
				#print("Done")
				if fld_name[i] != recog_all[i] and len(fld_name[i]) == len(recog_all[i]):
					temp_wrn = []
					for exp2 in range(len(fld_name[i])):
						if fld_name[i][exp2] != recog_all[i][exp2]:
							temp_wrn.append( "("+fld_name[i][exp2]+","+recog_all[i][exp2]+")")
							temp_wrn1 = ",".join(temp_wrn)
							worksheet1.write(row,wrong_str,temp_wrn1)
				#print("Done1")
				if fld_name[i]==recog_all[i]:
					worksheet1.write(row,rresult_col,"1")
				else:
					worksheet1.write(row,rresult_col,"0")
				worksheet1.write(row,conf_col,str(conf_all[i]))
				worksheet1.write(row,o2,"".join(lst1[i]))
				worksheet1.write(row,co2,str(lst2[i]))
				temp_str = []
				#print("Done3")
				if len(fld_name[i]) == len(recog_all[i]):
					for re in range(len(lst1[i])):
						if recog_all[i][re] != fld_name[i][re]:
							temp_str.append(lst1[i][re])
						else:
							temp_str.append(fld_name[i][re])
					worksheet1.write(row,recog_sec,"".join(temp_str))
				#print("Done4")
				sp_img_col=co2+1
				for m in range(len(sp[i])):
					# print("image_name",pre[i][l],orig[i])
					worksheet1.insert_image(row,
					                    sp_img_col,
					                    sp[i][m],
					                     {'x_scale': .7, 'y_scale': .7,
					                            'x_offset':0,'y_offset':0,
					                            'positioning': 1})
					sp_img_col+=1
				#print("Done5")
			except Exception as e: 
				print(e)
			row+=1

	workbook.close()
