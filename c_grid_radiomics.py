# locate_bm.py
from dotenv import load_dotenv
import a_run_radiomics as rr
from datetime import datetime
import numpy as np
import os

load_dotenv()
RADIOMICS_FEATURES_PATH_OUT = os.environ.get("RADIOMICS_FEATURES_PATH")

label = 'MET'
cts_path, lcs_list, lc_metadata = rr.get_lc_metadata(label=label)
# print(lc_metadata)
case_ids = [f.split('/')[-2] for f in cts_path]
test_case_id = '1774119_20160809_1' 
test_ct_path = cts_path[case_ids.index(test_case_id)]
lc = lcs_list[case_ids.index(test_case_id)]
treshold = 0
size = [15,15]
contour = 'CY'
contour_name = contour+str(size[0])+'_'+str(case_ids.index(test_case_id))+'_'+test_case_id
radiomics_out = os.path.join(RADIOMICS_FEATURES_PATH_OUT, 'GRIDSEARCH', contour_name)
if not os.path.exists(radiomics_out):
        os.makedirs(radiomics_out)
x_scale = 7 #pixels in 7.5mm
y_scale = 7 #pixels in 7.5mm
z_scale = 3 #pixels in 7.5mm
n_x = 7
n_y = 7
n_z = 7
cnt =  0
t_cnt = n_x*n_y*n_z
medians = [0]
for x_inx in range(n_x):
	x_pxl_shift = x_scale*(x_inx-3)
	for y_inx in range(n_y):
		y_pxl_shift = y_scale*(y_inx-3)
		for z_inx in range(n_z):
			processed_files = os.listdir(radiomics_out)
			z_pxl_shift = z_scale*(z_inx-3)
			cnt += 1
			print (cnt)
			probe_c = [lc[0]+x_pxl_shift,lc[1]+y_pxl_shift,lc[2]+z_pxl_shift]
			# print (probe_c)
			t1 = datetime.now()
			label_name = '_'.join([test_case_id]+[str(f) for f in probe_c])
			if '_'+label_name+'.csv' not in processed_files:
				features = rr.process(test_case_id, test_ct_path, probe_c,contour, size, treshold)
				rr.write_csv(radiomics_out, features, '', label_name)
				t2 = datetime.now()
				medians.append(((t2-t1).total_seconds())/60.0)
			else:
				print('In Processed!')
			# print(t_cnt)
			print('remaining time: %i m'%(np.median(medians)*(t_cnt-cnt)))
