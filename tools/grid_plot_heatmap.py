import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import os
import csv
from dotenv import load_dotenv
load_dotenv()
paindicator_results = os.environ.get("PAINDICATOR_RESULTS")

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
ANNOT_SIZE = 12
database =   ['LBM','HM','PnPNLP' ,'PnP', 'PnPg3', 'PnPg4'][3]
label = ['','lytic','blastic','mix'][0]
param = 2 # 0:AUC 1:R2 2:PRECISION 3:RECALL 4:F1-SCORE

GRID_SEARCH = [0, 3] # [0:'ROI' 1:'RS' 2:'FS' 3:'ML']

roi_i = 0 # 9 #   0:C70 1:C100 2:S70 3:S100 
          #   5:C50 1:C30 2:C20 3:C15 4:C10 
          #     S100 S70 5:S50 6:S30 7:S20 8:S15 9:S10
          #     10:C5x3 11:C3x5 12:C3x2 13:C2x3
          #     14:AC, 15:AS, 16:AA
roi_e = roi_i+12 # max 23

rs_i = 0 #    0:NONE 1:SMOTE 2:TL 3:ROS 4:RUS
rs_e = rs_i+2

fs_i = 3  #    0:NONE 1:TREE 
          # 2:LASSO_0.1', 3:LASSO 
          # 4: 'LASSO_1',
          # 5:PFECV 6:VT_0.0 7:VT_0.8,
          # 8:PCA_20 9:FastICA_20 10:PCA_2 11:FastICA_2 12:FastICA_10 13:PCA_10 
fs_e = fs_i+1

ml_i = 0#      
ml_e = ml_i+15

rois = [
                'SP50', 'CY50', #0:2
                'CY5030',
                'SP30', 'CY30', #2:4
                'SP20', 'CY20', #4:6
                'SP15', 'CY15', #6:8
                'E4CY', 'E4SP', 'E5CY', 'E9SC',
                'EN3', 'ENIT3',
                'EN6', 'ENIT6',
                'SP50', 'CY50', #0:2
                'CY5030',
                'SP30', 'CY30', #2:4
                'SP20', 'CY20', #4:6  
                'SP10', 'CY10', #8:10
                'SP7', 'CY7', #10:12
                'SPIT50', 'CYIT50', #12:14
                'SPIT30', 'CYIT30', #14:16
                'SPIT20', 'CYIT20', #16:18 
                'SPIT15', 'CYIT15', #18:20
                'SPIT10', 'CYIT10', #20:22
                'SPIT7', 'CYIT7', #22:24
                'E3CY', 'E3SP', 'E6SC', #24:27 
                'E6CY', 'E6SP',  'E12SC',
                'EIT3CY', 'EIT3SP', 'EIT6SC', #27:30 
                'EIT6CY', 'EIT6SP', 'EIT12SC', #33-36
                # 'E4CY', 'E4SP', 'E5CY', 'E9SC', #24:27 
                   ]

data_resampling_methods = ['NONE', 'SMOTE', 'TL', 'ROS', 'RUS']
feature_selection_methods = ['NONE', 'TREE',
              'LASSO_0.1','LASSO','LASSO_1',
              'PFECV', 'VT_0.8', 'VT_0.0', 
              'FastICA_20', 'FastICA_24', 'FastICA_30', 
              'PCA_20', 'PCA_24', 'PCA_30']
ml_models = [   
    "RBF SVM",               #0 
    "Naive Bayes",           #1
    "QDA",                   #2   
    "Decision Tree",         #3
    "Nearest Neighbors",     #4 
    "Bagging",               #5
    "AdaBoost",              #6
    "Random Forest 100",     #7
    "Neural Net relu lbfgs", #8
    "Linear SVM",            #9
    "Neural Net",            #10
    "Gaussian Process",      #11
]

ml_abrs = [
    "SVM",          #0
    "NB",           #1
    "QDA",          #2
    "DT",           #3
    "kNN",          #4    
    "Bagging"       #5
    "AdaBoost",     #6
    "RF",           #7
    "NNet_lbfgs",   #8
    "L_SVM",        #9
    "NNet",         #10
    "GPR",          #11
    ]

def get_hyperspace(database, label, rois, data_resampling_methods, feature_selection_methods, ml_models):
  hyperspace = []
  errorspace = []
  roi_names = []
  for roi_name in rois:
        roi_names.append(roi_name)
        hyperspace.append([])
        errorspace.append([])
        
        # rs_methods = []
        for rs_method in data_resampling_methods:
              hyperspace[roi_names.index(roi_name)].append([])
              errorspace[roi_names.index(roi_name)].append([])
              # rs_methods.append(rs_method)
              # fs_methods = []
              for fs_method in feature_selection_methods:
                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)].append([])
                  errorspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)].append([])
                  # model_names = []
                  auc_values = {}
                  auc_stdvs = {}
                  r2_values = {} 
                  pr_values = {} 
                  rc_values = {} 
                  f1_values = {}
                  # fs_methods.append(fs_method)
                  file_tag = '%s_%s_%s_%s'%(database, roi_name,fs_method,rs_method)
                  # print file_tag
                  file_full_path = os.path.join(paindicator_results, database, file_tag)

                  performance_files = [f for f in os.listdir(file_full_path) if '.npy' in f and label in f]
                  # print file_tag, performance_files
                  for performance_file in performance_files:
                      ml_model = performance_file.split('_')[4]
                      # removes FS arg
                      if ml_model in data_resampling_methods:
                        ml_model = performance_file.split('_')[5]
                      # adds ML arg
                      if performance_file.split('_')[6] != 'ROC':
                        ml_model = ml_model+'_'+performance_file.split('_')[6]
                      print (ml_model)
                      if ml_model in ml_models:
                        # print file_full_path, '//',performance_file
                        with open(os.path.join(file_full_path, performance_file), 'r') as textfile:
                            csvreader = csv.reader(textfile)
                            header = next(csvreader)
                            print (header)
                            ## this path keeps the value with best result
                            dataArray = next(csvreader)
                            # print dataArray
                            auc_sem = stats.sem([float(f) for f in dataArray[:5]])
                            if ml_models.index(ml_model) in auc_values:
                              if auc_values[ml_models.index(ml_model)] < float(dataArray[10]):
                                auc_values[ml_models.index(ml_model)] = float(dataArray[10])
                                auc_stdvs[ml_models.index(ml_model)] = '%i$\pm$%i'%(round(float(dataArray[10])*100),round(auc_sem*100))
                                r2_values[ml_models.index(ml_model)] = float(dataArray[11])
                                pr_values[ml_models.index(ml_model)] = float(dataArray[12])
                                rc_values[ml_models.index(ml_model)] = float(dataArray[13])
                                f1_values[ml_models.index(ml_model)] = float(dataArray[14])
                            else:
                              auc_values[ml_models.index(ml_model)] = float(dataArray[10])
                              auc_stdvs[ml_models.index(ml_model)] = '%i$\pm$%i'%(round(float(dataArray[10])*100),round(auc_sem*100))
                              r2_values[ml_models.index(ml_model)] = float(dataArray[11])
                              pr_values[ml_models.index(ml_model)] = float(dataArray[12])
                              rc_values[ml_models.index(ml_model)] = float(dataArray[13])
                              f1_values[ml_models.index(ml_model)] = float(dataArray[14])
                      else:
                        # print "ESC: ", ml_model
                        pass
                  # print (model_names)
                  if len(auc_values.keys()) != len(ml_models):
                    print ('Error in ML models: ', auc_values.keys())
                  errorspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][feature_selection_methods.index(fs_method)].append(np.array(auc_stdvs.values()))       
                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][feature_selection_methods.index(fs_method)].append(np.array(auc_values.values()))
                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][feature_selection_methods.index(fs_method)].append(np.array(r2_values.values()))
                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][feature_selection_methods.index(fs_method)].append(np.array(pr_values.values()))
                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][feature_selection_methods.index(fs_method)].append(np.array(rc_values.values()))
                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][feature_selection_methods.index(fs_method)].append(np.array(f1_values.values()))
                       
                          # print dataArray

  hyperspace = np.array(hyperspace)
  errorspace = np.array(errorspace)
  
  return hyperspace, errorspace

'''
VARIABLES:

rois =  [ 'CY70','CY100', 'SP70', 'SP100',
        'CY50', 'CY30','CY20', 'CY15', 'CY10',
        'SP50', 'SP30','SP20', 'SP15', 'SP10',
        'CY5030','CY3050','CY3020', 'CY2030',
        'E7CY','E11CY', 'E7SP', 'E18CS',]

data_resampling_methods = ['NONE', 'SMOTE', 'TL', 'ROS', 'RUS']

feature_selection_methods = ['NONE', 'LASSO','TREE','PFECV', 'VT_0.0', 'VT_0.8',
                          'PCA_20', 'FastICA_20', 'PCA_2',  
                          'FastICA_2', 'FastICA_10', 'PCA_10']

ml_models = [    
    "Gaussian Process",      #0
    "Linear SVM",            #1
    "Neural Net",            #2
    "Neural Net relu lbfgs", #3
    # "Neural Net reg",        #4
    # "Neural Net reg lbfgs ", #5
    "AdaBoost",              #6
    "Random Forest 100",     #7
    # "Random Forest",         #8
    # "Balanced Linear SVM",   #9
    "RBF SVM",               #10
    "Nearest Neighbors",     #11    
    "Decision Tree",         #12
    "Naive Bayes",           #13
    "QDA",                   #14
    "Bagging"                #15
]


HYPER SPACE
[roi][rs][fs][param][ml]

[params]: [0] ROC-AUC values
          [1] R2 values
          [2] Precision
          [3] Recall
          [4] F-1 Score 


'''
# print len(roi_names), len(data_resampling_methods),len(feature_selection_methods)



rois = rois[roi_i:roi_e]
data_resampling_methods = data_resampling_methods[rs_i:rs_e]
feature_selection_methods = feature_selection_methods[fs_i:fs_e]
ml_models = ml_models[ml_i:ml_e]
ml_abrs = ml_abrs[ml_i:ml_e]

hyperspace, errorspace = get_hyperspace(database, label, rois, data_resampling_methods, feature_selection_methods, ml_models)
print (hyperspace.shape)
'''
hyperspace[ROI][RS][FS]=[]
'''
x_variable_name = ["ROI", "Resampling Method", "Feature Selection Method", "ML Model"][GRID_SEARCH[0]]
x_variable = [rois, data_resampling_methods, feature_selection_methods, ml_abrs][GRID_SEARCH[0]]
y_variable_name = ["ROI", "Resampling Method", "Feature Selection Method", "ML Model"][GRID_SEARCH[1]]
y_variable = [rois, data_resampling_methods, feature_selection_methods, ml_abrs][GRID_SEARCH[1]]

# print hyperspace
if    GRID_SEARCH[0] == 0 and GRID_SEARCH[1] == 1:
  measure = hyperspace[:, :, 0, param, 0]  # ROI/RS
  measure_sdv = errorspace[:, :, 0, 0, 0]  # ROI/RS
  color_map = "YlGn"

elif  GRID_SEARCH[0] == 0 and GRID_SEARCH[1] == 2:
  measure = hyperspace[:, 0, :, param, 0]  # ROI/FS
  measure_sdv = errorspace[:, 0, :, 0, 0]  # ROI/FS
  color_map = 'YlOrRd'
  color_map = 'Purples'
  
elif  GRID_SEARCH[0] == 0 and GRID_SEARCH[1] == 3:
  measure = hyperspace[:, 0, 0, param, :]  # ROI/ML
  measure_sdv = errorspace[:, 0, 0, 0, :]  # ROI/ML
  color_map = "YlOrBr"
  
elif  GRID_SEARCH[0] == 1 and GRID_SEARCH[1] == 2:
  measure = hyperspace[0, :, :, param, 0]  # RS/FS
  measure_sdv = errorspace[0, :, :, 0, 0]  # RS/FS
  color_map = 'Purples'

elif  GRID_SEARCH[0] == 1 and GRID_SEARCH[1] == 3:
  measure = hyperspace[0, :, 0, param, :]  # RS/ML
  measure_sdv = errorspace[0, :, 0, 0, :]  # RS/ML
  color_map = 'Blues'

elif  GRID_SEARCH[0] == 2 and GRID_SEARCH[1] == 3:
  measure = hyperspace[0, 0, :, param, :]  # FS/ML
  measure_sdv = errorspace[0, 0, :, 0, :]  # FS/ML
  color_map = 'Greens'

if param == 4:
  color_map = 'Pastel2_r'
  color_map = 'RdYlGn'
# color_map = 'rainbow'
# color_map = 'seismic_r'

#color_map = 'YlGn'
'''
Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, 
CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, 
Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, 
Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, 
PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, 
RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, 
Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, 
Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, 
YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, 
bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, 
coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, 
flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, 
gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, 
gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, 
gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, 
jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, 
ocean_r, pink, pink_r, plasma, plasma_r, 
prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, 
summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, 
tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r
'''

 
print (measure.shape, len(x_variable), len(y_variable))
df = pd.DataFrame(data=measure, columns=y_variable, index=x_variable)
if param == 0:
  df_sdv = pd.DataFrame(data=measure_sdv, columns=y_variable, index=x_variable)
else:
  df_sdv = pd.DataFrame(data=measure, columns=y_variable, index=x_variable).round(decimals=3)


df['m1'] =  df.median(axis=1)
df.loc['m0'] = df.median()
df_sdv['m1'] =  df.median(axis=1)
df_sdv.loc['m0'] = df.median()

df['m1'] =  df.mean(axis=1)
df.loc['m0'] = df.mean()
df_sdv['m1'] =  df.mean(axis=1)
df_sdv.loc['m0'] = df.mean()


#df.loc['m0'] = df.max()
df['m1'] =  df.max(axis=1)
#df_sdv.loc['m0'] = df.max()
df_sdv['m1'] =  df.max(axis=1)

# print df.mean()



## sorts x 
#df = df.sort_values(by='m0', axis=1, ascending=True, inplace=False, kind='quicksort', na_position='last')
## sorts y
# df = df.sort_values(by='m1', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
df = df.drop('m1', axis=1)
df = df.drop('m0', axis=0)

## sorts x
#df_sdv = df_sdv.sort_values(by='m0', axis=1, ascending=True, inplace=False, kind='quicksort', na_position='last')
## sorts y
# df_sdv = df_sdv.sort_values(by='m1', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
df_sdv = df_sdv.drop('m1', axis=1)
df_sdv = df_sdv.drop('m0', axis=0)
# print int(round(df.shape[1]*3/4))
# print int(round(df.shape[0]*3/4))
plt.figure(figsize=(int(round(df.shape[1]*4/4)), int(round(df.shape[0]*2/4))))
plt.gcf().subplots_adjust(bottom=.2)
g = sns.heatmap(df, annot=df_sdv, fmt='',annot_kws={"size": ANNOT_SIZE}, 
 cmap=color_map, cbar=False,  vmin=0.40, vmax=.92)

plt.text(-.1, -.1, database+'_'+label+'_roi:'+rois[0]+' rs:'+data_resampling_methods[0]+' fs:'+feature_selection_methods[0]+' ml:'+ml_abrs[0])
plt.xticks(rotation=30,  rotation_mode="anchor", ha='right')

file_extension = ''
if len(data_resampling_methods) == 1:
  file_extension += '_RS.'+data_resampling_methods[0]
else:
  file_extension += '_RS.'+data_resampling_methods[0]+'-'+data_resampling_methods[-1]
if len(feature_selection_methods) == 1:
  file_extension += '_FS.'+feature_selection_methods[0]
else:
  file_extension += '_FS.'+feature_selection_methods[0]+'-'+feature_selection_methods[-1]
if len(ml_abrs) == 1:
  file_extension += '_ML.'+ml_abrs[0]
else:
  file_extension += '_ML.'+ml_abrs[0]+'-'+ml_abrs[-1]

df.to_csv(path_or_buf=paindicator_results+database+'/'+database+file_extension+'.csv', sep=',')


plt.show()

