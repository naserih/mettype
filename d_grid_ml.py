from dotenv import load_dotenv
import os
import pickle
import b_ml_classifier as cl
import a_run_radiomics as rr
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
load_dotenv()

RADIOMICS_FEATURES_PATH_OUT = os.environ.get("RADIOMICS_FEATURES_PATH")
ML_MODELS_PATH_OUT = os.environ.get("PAINDICATOR_RESULTS")
radiomics_out = os.path.join(RADIOMICS_FEATURES_PATH_OUT, 'GRIDSEARCH')

########################
target_case_id = '1774119_20160809_1'
target_center = [272,334,69]
database =  'HM'
roi = 'CY15'
fs_method = ['NONE','TREE'][0]
rs_method = ['NONE', 'SMOTE'][0]
n = 1
model_name = [ 
    "Gaussian Process",      #0
    "Linear SVM",            #1
    "Neural Net",            #2
    "Neural Net relu lbfgs", #3
    "AdaBoost",              #4
    "Random Forest 100",     #5
    "RBF SVM",               #6
    "Nearest Neighbors",     #7    
    "Decision Tree",         #8
    "Naive Bayes",           #9
    "QDA",                   #10
    "Bagging"                #11
    ][n]
    ###########
feature_path = [f for f in os.listdir(radiomics_out) if roi in f and target_case_id in f]
if len(feature_path) == 1:
	feature_path = os.path.join(radiomics_out, feature_path[0])
else:
	print ('Error: more than one Feature file for target')


'''
This segment is to load original CT image
'''

label = 'MET'
cts_path, lcs_list, lc_metadata = rr.get_lc_metadata(label=label)
case_ids = [f.split('/')[-2] for f in cts_path]
target_ct_path = cts_path[case_ids.index(target_case_id)]
target_ct_slices = [os.path.join(target_ct_path,f) for f in os.listdir(os.path.join(target_ct_path)) if 'CT.' in f]

ct_array, ct_x,ct_y,ct_z, ct_spacing = rr.get_cts(target_ct_slices)
    

'''
This segment of the script is to fit the pretrained model
to the target ROI grid
'''

rf_csvs = [os.path.join(feature_path, f) for f in os.listdir(feature_path) if 'FAILED' not in f]
rf_locs = [[int(f.split('_')[4]),
			int(f.split('_')[5]),
			len(ct_z)-int(f.split('_')[6].split('.')[0])] for f in os.listdir(feature_path) if 'FAILED' not in f]

# print(rf_locs)
print(len(rf_locs))
# print(len(rf_csvs))

ml_tag = '%s_%s_%s_%s'%(database,roi,fs_method,rs_method)
model_filename = '%s%s/%s_%s.sav'%(ML_MODELS_PATH_OUT,ml_tag,ml_tag,model_name)    
loaded_model = pickle.load(open(model_filename, 'rb'))
# print(loaded_model)

radiomics_fetures = cl.features(rf_csvs)
radiomics_labels = np.array(radiomics_fetures.labels)
X = np.array(radiomics_fetures.df)
X = StandardScaler().fit_transform(X)
# X = cl.feature_selection(X,[1]*len(rf_locs), '', fs_method)
                                                                                      

# print(X)
# result = loaded_model.score(X, Y_test)
result = loaded_model.predict(X)
result = np.array(result)
# print(result)


'''
This segment is to generate 2D presentations for a selected plane
'''

xs = sorted(set([rf[0] for rf in rf_locs]))
ys = sorted(set([rf[1] for rf in rf_locs]))
zs = sorted(set([rf[2] for rf in rf_locs]))

# print (xs, ys,zs)
inx_x = 3 # 0-6 set 3 for middle slice
inx_y = 3 # 0-6 set 3 for middle slice
inx_z = 3 # 0-6 set 3 for middle slice

x_array = []
y_array = []
z_array = []

for i in range(len(rf_locs)):
	x_array.append(rf_locs[i][0])
	y_array.append(rf_locs[i][1])
	z_array.append(rf_locs[i][2])

x_array = np.array(x_array)
y_array = np.array(y_array)
z_array = np.array(z_array)

# print(z_array==zs[inx_z])
x_array_z = x_array[z_array==zs[inx_z]]
y_array_z = y_array[z_array==zs[inx_z]]
val_array_z = result[z_array==zs[inx_z]]

x_array_y = x_array[y_array==ys[inx_y]]
z_array_y = z_array[y_array==ys[inx_y]]
val_array_y = result[y_array==ys[inx_y]]

y_array_x = y_array[x_array==xs[inx_x]]
z_array_x = z_array[x_array==xs[inx_x]]
val_array_x = result[x_array==xs[inx_x]]


plt.imshow(ct_array[zs[inx_z],:,:], cmap='gray', origin='lower')
plt.scatter(x_array_z, y_array_z, c=val_array_z, cmap='rainbow',edgecolors='none', marker = 's', alpha = 0.5)
plt.scatter(xs[inx_x], ys[inx_y], edgecolors='r', marker = 's', c='none' )

# plt.imshow(ct_array[:,:,xs[inx_x]], cmap='gray', origin='lower',aspect=3)
# plt.scatter(y_array_x, z_array_x, c=val_array_z, cmap='rainbow',edgecolors='none', marker = 's', alpha = 0.5)
# plt.scatter(ys[inx_y], zs[inx_z], edgecolors='r', marker = 's', c='none' )

# plt.imshow(ct_array[:,ys[inx_y],:], cmap='gray', origin='lower', aspect=3)
# plt.scatter(x_array_y, z_array_y, c=val_array_y, cmap='rainbow',edgecolors='none', marker = 's', alpha = 0.5)
# plt.scatter(xs[inx_x], zs[inx_z], edgecolors='r', marker = 's', c='none' )

plt.show()

