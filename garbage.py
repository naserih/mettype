import os
import pandas as pd 
total_len = 0
src = '/data/hossein/DATA/points/'
src = '/data/hossein/DATA/patients_documents/TS_MET_pdfs'
folders_list = [os.path.join(src,f) for f in os.listdir(src)]
#folders_list = [os.path.join(src,f) for f in os.listdir(src) if len(f)==11 and 'GA' in f]

#print (folders_list)
for folder in folders_list:	
	files_list = [os.path.join(folder,f) for f in os.listdir(folder) if 'ConsultNote' in f]
	total_len += len(files_list)

print (total_len)
#for folder in folders_list:
#	if len(os.listdir(folder)) != 1:
#		print 'Error'
#	sub_folder = os.path.join(folder, os.listdir(folder)[0])	
#	files_list = [os.path.join(sub_folder,f) for f in os.listdir(sub_folder)]
#	# print(files_list)
#	for file in files_list:
#		# print '>>>>, ', file
#		try:
#			df = pd.read_table(file, sep=",",header=None)
#		except:
#			df = []
#		print(len(df))
#		total_len += len(df)

#print (total_len)