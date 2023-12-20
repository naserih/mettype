import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
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
database =   ['LBM','HM','HM_RV','PnPNLP' ,'PnP', 'PnPg3', 'PnPg4'][2]
label = ['','lytic','blastic','mix'][0]
param_names = [ 'v_A', 'v_P', 'v_R', 'v_S', 'v_F1', 'v_AUC',
                't_A', 't_P', 't_R', 't_S', 't_F1', 't_AUC',
                'md_v_A', 'md_v_P', 'md_v_R', 'md_v_S', 'md_v_F1', 'md_v_AUC',
                'md_t_A', 'md_t_P', 'md_t_R', 'md_t_S', 'md_t_F1', 'md_t_AUC',
                'mx_v_A', 'mx_v_P', 'mx_v_R', 'mx_v_S', 'mx_v_F1', 'mx_v_AUC',
                'mx_t_A', 'mx_t_P', 'mx_t_R', 'mx_t_S', 'mx_t_F1', 'mx_t_AUC',
                'sd_v_A', 'sd_v_P', 'sd_v_R', 'sd_v_S', 'sd_v_F1', 'sd_v_AUC',
                'sd_t_A', 'sd_t_P', 'sd_t_R', 'sd_t_S', 'sd_t_F1', 'sd_t_AUC',] 


ms = 0 # ['mean','median','max','SD']
db = 1 # ['validation', 'test']
mt = 4 # ['AC','PR', 'RC','SP', 'F1','AUC']

param = (12*ms+6*db)+mt # 0:v_A 1:v_P 2:v_R 3:v_S 4:v_F1 5:v_AUC 
          # 6:t_A 7:t_P 8:t_R 9:t_S 10:t_F1 11:t_AUC 
          # 12:v_A 13:v_P 14:v_R 15:v_S 16:v_F1 17:v_AUC 
          # 18:t_A 19:t_P 20:t_R 21:t_S 22:t_F1 23:t_AUC 
print (param)
print(param_names[param])
optimize = False
GRID_SEARCH = [0, 3] # [0:'ROI' 1:'RS' 2:'FS' 3:'ML']

roi_i = 0 # 9 #   0:C70 1:C100 2:S70 3:S100 
          #   5:C50 1:C30 2:C20 3:C15 4:C10 
          #     S100 S70 5:S50 6:S30 7:S20 8:S15 9:S10
          #     10:C5x3 11:C3x5 12:C3x2 13:C2x3
          #     14:AC, 15:AS, 16:AA
roi_e = roi_i+13 # max 23

rs_i = 0 #    0:NONE 1:SMOTE 2:TL 3:ROS 4:RUS
rs_e = rs_i+1

fs_i =  2 #    0:NONE 1:TREE 
          # 2:LASSO_0.1', 3:LASSO 
          # 4: 'LASSO_1',
          # 5:PFECV 6:VT_0.0 7:VT_0.8,
          # 8:PCA_20 9:FastICA_20 10:PCA_2 11:FastICA_2 12:FastICA_10 13:PCA_10 
fs_e = fs_i+1

ml_i = 0#      
ml_e = ml_i+13

rois = [
'SP50', 'CY50', #0:2
                'CY5030',
                'SP30', 'CY30', #2:4
                'SP20', 'CY20', #4:6
                'SP15', 'CY15', #6:8
'E4CY', 'E4SP', 'E5CY', 'E9SC', #24:27 
                'EN3', 
                'EN6', 'ENIT3', 'ENIT6',
                'SP50', 'CY50', #0:2
                # 'CY5030',
                'SP30', 'CY30', #2:4
                'SP20', 'CY20', #4:6
                'SP15', 'CY15', #6:8
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
                # 
                   ]

data_resampling_methods = ['NONE', 'SMOTE', 'TL', 'ROS', 'RUS']
feature_selection_methods = ['NONE', 
              'LASSO_0.1','LASSO','LASSO_1',
              'TREE', 'PFECV', 'VT_0.8', 'VT_0.0', 
              'FastICA_20', 'FastICA_24', 'FastICA_30', 
              'PCA_20', 'PCA_24', 'PCA_30']
ml_models = [    
    "SVM",               #0
    "NB",           #1
    "QDA",                   #2
    "DT",         #3
    "kNN",    #4    
    "Bagging" ,              #5
    "AdaBoost",              #6
    "RF",     #7
    # "RF_b",     #7
    # "RF_bs",     #7
    "NNet_lbfgs", #8
    "L_SVM",            #11
    # "NNet_b",            #9
    "NNet",            #9
    "GPR",      #10
    ]
ml_abrs = [
    "SVM",               #0
    "NB",           #1
    "QDA",                   #2
    "DT",         #3
    "kNN",    #4    
    "Bagging" ,              #*5
    "AdaBoost",              #6
    "RF",     #7
    # "RF_b",     #7
    # "RF_bs",     #7
    "NNet_lbfgs", #8
    "L_SVM",            #11
    # "NNet_b",            #9
    "NNet",            #9
    "GPR",      #10
    ]

def get_hyperspace(database, label, rois, data_resampling_methods, feature_selection_methods, ml_models):
  hyperspace = []
  roi_names = []
  for roi_name in rois:
        roi_names.append(roi_name)
        hyperspace.append([])
        
        # rs_methods = []
        for rs_method in data_resampling_methods:
              hyperspace[roi_names.index(roi_name)].append([])
              # rs_methods.append(rs_method)
              # fs_methods = []
              for fs_method in feature_selection_methods:
                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)].append([])
                  # model_names = []
                  v_ac_values =  {}
                  v_sp_values = {}
                  v_auc_values = {}
                  v_pr_values = {} 
                  v_rc_values = {} 
                  v_f1_values = {}
                  v_tp_values = {}
                  v_tn_values = {}
                  v_fp_values = {}
                  v_fn_values = {}
                  t_ac_values =  {}
                  t_sp_values = {}
                  t_auc_values = {}
                  t_pr_values = {} 
                  t_rc_values = {} 
                  t_f1_values = {}
                  t_tp_values = {}
                  t_tn_values = {}
                  t_fp_values = {}
                  t_fn_values = {}
                  v_ac_sdv = {} 
                  v_sp_sdv = {} 
                  v_auc_sdv = {}
                  v_pr_sdv = {} 
                  v_rc_sdv = {} 
                  v_f1_sdv = {}
                  v_tp_sdv = {}
                  v_tn_sdv = {}
                  v_fp_sdv = {}
                  v_fn_sdv = {}
                  t_ac_sdv = {} 
                  t_sp_sdv = {} 
                  t_auc_sdv = {}
                  t_pr_sdv = {} 
                  t_rc_sdv = {} 
                  t_f1_sdv = {}
                  t_tp_sdv = {}
                  t_tn_sdv = {}
                  t_fp_sdv = {}
                  t_fn_sdv = {}
                  cnts = [0]*len(ml_models)
                  # fs_methods.append(fs_method)
                  file_tag = '%s_%s_%s_%s'%(database, roi_name,fs_method,rs_method)
                  # print (file_tag)
                  file_full_path = os.path.join(paindicator_results, database, file_tag)

                  performance_files = [f for f in os.listdir(file_full_path) if '.npy' in f and label in f]
                  # print  (performance_files)
                  for performance_file in performance_files:
                      ml_model = performance_file.split('_')[4]    
                      # removes FS arg
                      if ml_model in data_resampling_methods:
                        ml_model = performance_file.split('_')[5]
                      # adds ML arg
                        if performance_file.split('_')[7] == 'ROC':
                          ml_model = ml_model+'_'+performance_file.split('_')[6]
                      else:
                        if performance_file.split('_')[6] == 'ROC':
                          ml_model = ml_model+'_'+performance_file.split('_')[5]
                      if ml_model in ml_models:
                        # print file_full_path, '//',performance_file
                        # print(ml_model,'>>>', performance_file)
                        with open(os.path.join(file_full_path, performance_file), 'r') as textfile:
                            csvreader = csv.reader(textfile)
                            header = next(csvreader)
                            # print (header)
                            ## this path keeps the value with best result
                            dataArray = next(csvreader)
                            t_sp = float(dataArray[13])/(float(dataArray[13])+float(dataArray[14]))
                            t_ac = (float(dataArray[12])
                                  +float(dataArray[13]))/(float(dataArray[12])
                                  +float(dataArray[13])+float(dataArray[14])+float(dataArray[15]))
                            v_sp = float(dataArray[5])/(float(dataArray[5])+float(dataArray[6]))
                            v_ac = (float(dataArray[4])
                                  +float(dataArray[5]))/(float(dataArray[4])
                                  +float(dataArray[5])+float(dataArray[6])+float(dataArray[7]))
                            # print (dataArray)
                            # auc_sem = stats.sem([float(f) for f in dataArray[:5]])
                            if ml_models.index(ml_model) in t_auc_values:
                              v_ac_sdv[ml_models.index(ml_model)].append(v_ac)
                              v_sp_sdv[ml_models.index(ml_model)].append(v_sp)
                              v_pr_sdv[ml_models.index(ml_model)].append(float(dataArray[0]))
                              v_rc_sdv[ml_models.index(ml_model)].append(float(dataArray[1]))
                              v_f1_sdv[ml_models.index(ml_model)].append(float(dataArray[2]))
                              v_auc_sdv[ml_models.index(ml_model)].append(float(dataArray[3]))
                              v_tp_sdv[ml_models.index(ml_model)].append(float(dataArray[4]))
                              v_tn_sdv[ml_models.index(ml_model)].append(float(dataArray[5]))
                              v_fp_sdv[ml_models.index(ml_model)].append(float(dataArray[6]))
                              v_fn_sdv[ml_models.index(ml_model)].append(float(dataArray[7]))                 
                              t_ac_sdv[ml_models.index(ml_model)].append(t_ac)
                              t_sp_sdv[ml_models.index(ml_model)].append(t_sp)
                              t_pr_sdv[ml_models.index(ml_model)].append(float(dataArray[8]))
                              t_rc_sdv[ml_models.index(ml_model)].append(float(dataArray[9]))
                              t_f1_sdv[ml_models.index(ml_model)].append(float(dataArray[10]))
                              t_auc_sdv[ml_models.index(ml_model)].append(float(dataArray[11]))
                              t_tp_sdv[ml_models.index(ml_model)].append(float(dataArray[12]))
                              t_tn_sdv[ml_models.index(ml_model)].append(float(dataArray[13]))
                              t_fp_sdv[ml_models.index(ml_model)].append(float(dataArray[14]))
                              t_fn_sdv[ml_models.index(ml_model)].append(float(dataArray[15])) 

                              if optimize and t_f1_values[ml_models.index(ml_model)] < float(dataArray[10]):
                                # print(ml_model, float(dataArray[10]), '>',t_f1_values[ml_models.index(ml_model)])
                                  v_sp_values[ml_models.index(ml_model)] = v_sp
                                  v_ac_values[ml_models.index(ml_model)] = v_ac
                                  v_pr_values[ml_models.index(ml_model)] = float(dataArray[0])
                                  v_rc_values[ml_models.index(ml_model)] = float(dataArray[1])
                                  v_f1_values[ml_models.index(ml_model)] = float(dataArray[2])
                                  v_auc_values[ml_models.index(ml_model)] = float(dataArray[3])
                                  v_tp_values[ml_models.index(ml_model)] = float(dataArray[4])
                                  v_tn_values[ml_models.index(ml_model)] = float(dataArray[5])
                                  v_fp_values[ml_models.index(ml_model)] = float(dataArray[6])
                                  v_fn_values[ml_models.index(ml_model)] = float(dataArray[7])                  
                                  t_sp_values[ml_models.index(ml_model)] = t_sp
                                  t_ac_values[ml_models.index(ml_model)] = t_ac
                                  t_pr_values[ml_models.index(ml_model)] = float(dataArray[8])
                                  t_rc_values[ml_models.index(ml_model)] = float(dataArray[9])
                                  t_f1_values[ml_models.index(ml_model)] = float(dataArray[10])
                                  t_auc_values[ml_models.index(ml_model)] = float(dataArray[11])
                                  t_tp_values[ml_models.index(ml_model)] = float(dataArray[12])
                                  t_tn_values[ml_models.index(ml_model)] = float(dataArray[13])
                                  t_fp_values[ml_models.index(ml_model)] = float(dataArray[14])
                                  t_fn_values[ml_models.index(ml_model)] = float(dataArray[15])
                                  
                              elif not optimize:
                                  cnts[ml_models.index(ml_model)] += 1
                                  v_sp_values[ml_models.index(ml_model)] += v_sp
                                  v_ac_values[ml_models.index(ml_model)] += v_ac
                                  v_pr_values[ml_models.index(ml_model)] += float(dataArray[0])
                                  v_rc_values[ml_models.index(ml_model)] += float(dataArray[1])
                                  v_f1_values[ml_models.index(ml_model)] += float(dataArray[2])
                                  v_auc_values[ml_models.index(ml_model)] += float(dataArray[3])
                                  v_tp_values[ml_models.index(ml_model)] += float(dataArray[4])
                                  v_tn_values[ml_models.index(ml_model)] += float(dataArray[5])
                                  v_fp_values[ml_models.index(ml_model)] += float(dataArray[6])
                                  v_fn_values[ml_models.index(ml_model)] += float(dataArray[7])                  
                                  t_sp_values[ml_models.index(ml_model)] += t_sp
                                  t_ac_values[ml_models.index(ml_model)] += t_ac
                                  t_pr_values[ml_models.index(ml_model)] += float(dataArray[8])
                                  t_rc_values[ml_models.index(ml_model)] += float(dataArray[9])
                                  t_f1_values[ml_models.index(ml_model)] += float(dataArray[10])
                                  t_auc_values[ml_models.index(ml_model)] += float(dataArray[11])
                                  t_tp_values[ml_models.index(ml_model)] += float(dataArray[12])
                                  t_tn_values[ml_models.index(ml_model)] += float(dataArray[13])
                                  t_fp_values[ml_models.index(ml_model)] += float(dataArray[14])
                                  t_fn_values[ml_models.index(ml_model)] += float(dataArray[15])
                                  
                            else:
                                cnts[ml_models.index(ml_model)] += 1
                                v_sp_values[ml_models.index(ml_model)] = v_sp
                                v_ac_values[ml_models.index(ml_model)] = v_ac
                                v_pr_values[ml_models.index(ml_model)] = float(dataArray[0])
                                v_rc_values[ml_models.index(ml_model)] = float(dataArray[1])
                                v_f1_values[ml_models.index(ml_model)] = float(dataArray[2])
                                v_auc_values[ml_models.index(ml_model)] = float(dataArray[3])
                                v_tp_values[ml_models.index(ml_model)] = float(dataArray[4])
                                v_tn_values[ml_models.index(ml_model)] = float(dataArray[5])
                                v_fp_values[ml_models.index(ml_model)] = float(dataArray[6])
                                v_fn_values[ml_models.index(ml_model)] = float(dataArray[7])                  
                                t_sp_values[ml_models.index(ml_model)] = t_sp
                                t_ac_values[ml_models.index(ml_model)] = t_ac
                                t_pr_values[ml_models.index(ml_model)] = float(dataArray[8])
                                t_rc_values[ml_models.index(ml_model)] = float(dataArray[9])
                                t_f1_values[ml_models.index(ml_model)] = float(dataArray[10])
                                t_auc_values[ml_models.index(ml_model)] = float(dataArray[11])
                                t_tp_values[ml_models.index(ml_model)] = float(dataArray[12])
                                t_tn_values[ml_models.index(ml_model)] = float(dataArray[13])
                                t_fp_values[ml_models.index(ml_model)] = float(dataArray[14])
                                t_fn_values[ml_models.index(ml_model)] = float(dataArray[15])
                                
                                v_sp_sdv[ml_models.index(ml_model)] = [v_sp]
                                v_ac_sdv[ml_models.index(ml_model)] = [v_ac]
                                v_pr_sdv[ml_models.index(ml_model)] = [float(dataArray[0])]
                                v_rc_sdv[ml_models.index(ml_model)]= [float(dataArray[1])]
                                v_f1_sdv[ml_models.index(ml_model)]= [float(dataArray[2])]
                                v_auc_sdv[ml_models.index(ml_model)]= [float(dataArray[3])]
                                v_tp_sdv[ml_models.index(ml_model)]= [float(dataArray[4])]
                                v_tn_sdv[ml_models.index(ml_model)]= [float(dataArray[5])]
                                v_fp_sdv[ml_models.index(ml_model)]= [float(dataArray[6])]
                                v_fn_sdv[ml_models.index(ml_model)]= [float(dataArray[7])]                
                                t_sp_sdv[ml_models.index(ml_model)] = [t_sp]
                                t_ac_sdv[ml_models.index(ml_model)] = [t_ac]
                                t_pr_sdv[ml_models.index(ml_model)]= [float(dataArray[8])]
                                t_rc_sdv[ml_models.index(ml_model)]= [float(dataArray[9])]
                                t_f1_sdv[ml_models.index(ml_model)]= [float(dataArray[10])]
                                t_auc_sdv[ml_models.index(ml_model)]= [float(dataArray[11])]
                                t_tp_sdv[ml_models.index(ml_model)]= [float(dataArray[12])]
                                t_tn_sdv[ml_models.index(ml_model)]= [float(dataArray[13])]
                                t_fp_sdv[ml_models.index(ml_model)]= [float(dataArray[14])]
                                t_fn_sdv[ml_models.index(ml_model)]= [float(dataArray[15])]               
                      else:
                        # print "ESC: ", ml_model
                        pass
                  # print model_names
                  if len(t_auc_values.keys()) != len(ml_models):
                    print ('Error in ML models: ', t_auc_values.keys())
                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.mean(v_ac_sdv[key]) for key in sorted(v_ac_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.mean(v_pr_sdv[key]) for key in sorted(v_pr_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.mean(v_rc_sdv[key]) for key in sorted(v_rc_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.mean(v_sp_sdv[key]) for key in sorted(v_sp_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.mean(v_f1_sdv[key]) for key in sorted(v_f1_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.mean(v_auc_sdv[key]) for key in sorted(v_auc_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.mean(t_ac_sdv[key]) for key in sorted(t_ac_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.mean(t_pr_sdv[key]) for key in sorted(t_pr_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.mean(t_rc_sdv[key]) for key in sorted(t_rc_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.mean(t_sp_sdv[key]) for key in sorted(t_sp_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.mean(t_f1_sdv[key]) for key in sorted(t_f1_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.mean(t_auc_sdv[key]) for key in sorted(t_auc_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.median(v_ac_sdv[key]) for key in sorted(v_ac_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.median(v_pr_sdv[key]) for key in sorted(v_pr_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.median(v_rc_sdv[key]) for key in sorted(v_rc_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.median(v_sp_sdv[key]) for key in sorted(v_sp_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.median(v_f1_sdv[key]) for key in sorted(v_f1_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.median(v_auc_sdv[key]) for key in sorted(v_auc_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.median(t_ac_sdv[key]) for key in sorted(t_ac_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.median(t_pr_sdv[key]) for key in sorted(t_pr_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.median(t_rc_sdv[key]) for key in sorted(t_rc_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.median(t_sp_sdv[key]) for key in sorted(t_sp_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.median(t_f1_sdv[key]) for key in sorted(t_f1_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.median(t_auc_sdv[key]) for key in sorted(t_auc_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.max(v_ac_sdv[key]) for key in sorted(v_ac_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.max(v_pr_sdv[key]) for key in sorted(v_pr_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.max(v_rc_sdv[key]) for key in sorted(v_rc_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.max(v_sp_sdv[key]) for key in sorted(v_sp_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.max(v_f1_sdv[key]) for key in sorted(v_f1_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.max(v_auc_sdv[key]) for key in sorted(v_auc_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.max(t_ac_sdv[key]) for key in sorted(t_ac_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.max(t_pr_sdv[key]) for key in sorted(t_pr_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.max(t_rc_sdv[key]) for key in sorted(t_rc_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.max(t_sp_sdv[key]) for key in sorted(t_sp_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.max(t_f1_sdv[key]) for key in sorted(t_f1_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.max(t_auc_sdv[key]) for key in sorted(t_auc_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.std(v_ac_sdv[key]) for key in sorted(v_ac_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.std(v_pr_sdv[key]) for key in sorted(v_pr_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.std(v_rc_sdv[key]) for key in sorted(v_rc_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.std(v_sp_sdv[key]) for key in sorted(v_sp_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.std(v_f1_sdv[key]) for key in sorted(v_f1_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.std(v_auc_sdv[key]) for key in sorted(v_auc_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.std(t_ac_sdv[key]) for key in sorted(t_ac_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.std(t_pr_sdv[key]) for key in sorted(t_pr_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.std(t_rc_sdv[key]) for key in sorted(t_rc_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.std(t_sp_sdv[key]) for key in sorted(t_sp_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.std(t_f1_sdv[key]) for key in sorted(t_f1_sdv.keys())]))

                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][
                  feature_selection_methods.index(fs_method)].append(
                    np.array([np.std(t_auc_sdv[key]) for key in sorted(t_auc_sdv.keys())]))

                  #print('t_auc_sdv...',roi_name, rs_method, fs_method, [np.mean(t_auc_sdv[key]) for key in sorted(t_auc_sdv.keys())])
                  
  hyperspace = np.array(hyperspace)

  return hyperspace

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

[params]: [0] validation Precision
          [1] validation Recall
          [2] validation F1score
          [3] validation AUC
          [4] test Precision
          [5] test Recall
          [6] test F1score
          [7] test AUC


'''
# print len(roi_names), len(data_resampling_methods),len(feature_selection_methods)


rois = rois[roi_i:roi_e]
data_resampling_methods = data_resampling_methods[rs_i:rs_e]
feature_selection_methods = feature_selection_methods[fs_i:fs_e]
ml_models = ml_models[ml_i:ml_e]
ml_abrs = ml_abrs[ml_i:ml_e]

hyperspace = get_hyperspace(database, label, rois, data_resampling_methods, feature_selection_methods, ml_models)
print ('shape: ', hyperspace.shape)
'''
hyperspace[ROI][RS][FS][metric]=[]
'''
#print(print (hyperspace[0][0][0][0]))
x_variable_name = ["ROI", "Resampling Method", "Feature Selection Method", "ML Model"][GRID_SEARCH[0]]
x_variable = [rois, data_resampling_methods, feature_selection_methods, ml_abrs][GRID_SEARCH[0]]
y_variable_name = ["ROI", "Resampling Method", "Feature Selection Method", "ML Model"][GRID_SEARCH[1]]
y_variable = [rois, data_resampling_methods, feature_selection_methods, ml_abrs][GRID_SEARCH[1]]

# print hyperspace
if    GRID_SEARCH[0] == 0 and GRID_SEARCH[1] == 1:
  measure = hyperspace[:, :, 0, param, 0]  # ROI/RS
  color_map = "YlGn"

elif  GRID_SEARCH[0] == 0 and GRID_SEARCH[1] == 2:
  measure = hyperspace[:, 0, :, param, 0]  # ROI/FS
  color_map = 'YlOrRd'
  color_map = 'Purples'
  
elif  GRID_SEARCH[0] == 0 and GRID_SEARCH[1] == 3:
  measure = hyperspace[:, 0, 0, param, :]  # ROI/ML
  color_map = "YlOrBr"
  
elif  GRID_SEARCH[0] == 1 and GRID_SEARCH[1] == 2:
  measure = hyperspace[0, :, :, param, 0]  # RS/FS
  color_map = 'Purples'

elif  GRID_SEARCH[0] == 1 and GRID_SEARCH[1] == 3:
  measure = hyperspace[0, :, 0, param, :]  # RS/ML
  color_map = 'Blues'

elif  GRID_SEARCH[0] == 2 and GRID_SEARCH[1] == 3:
  measure = hyperspace[0, 0, :, param, :]  # FS/ML
  color_map = 'Greens'

if mt == 4:
  color_map = 'Pastel2_r'
  color_map = 'RdYlGn'
# color_map = 'rainbow'
# color_map = 'seismic_r'

# color_map = 'YlGn'
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
print(param)
print(measure)
print (measure.shape, len(x_variable), len(y_variable))
df = pd.DataFrame(data=measure, columns=y_variable, index=x_variable)


# df['m1'] =  df.median(axis=1)
# df.loc['m0'] = df.median()

# df['m1'] =  df.mean(axis=1)
# df.loc['m0'] = df.mean()

#df.loc['m0'] = df.max()
# df['m1'] =  df.max(axis=1)

# print df.mean()



## sorts x 
#df = df.sort_values(by='m0', axis=1, ascending=True, inplace=False, kind='quicksort', na_position='last')
## sorts y
# df = df.sort_values(by='m1', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
# df = df.drop('m1', axis=1)
# df = df.drop('m0', axis=0)

## sorts x
#df_sdv = df_sdv.sort_values(by='m0', axis=1, ascending=True, inplace=False, kind='quicksort', na_position='last')
## sorts y
# df_sdv = df_sdv.sort_values(by='m1', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
# print int(round(df.shape[1]*3/4))
# print int(round(df.shape[0]*3/4))
plt.figure(figsize=(int(round(df.shape[1]*4/4)), int(round(df.shape[0]*2/4))))
plt.gcf().subplots_adjust(bottom=.2)
g = sns.heatmap(df, annot=df, fmt='0.1%',annot_kws={"size": ANNOT_SIZE}, 
 cmap=color_map, cbar=False,  vmin=0.40, vmax=.92)

plt.text(-.1, -.1, database+'_'+label+param_names[param]+'_roi:'+rois[0]+' rs:'+data_resampling_methods[0]+' fs:'+feature_selection_methods[0]+' ml:'+ml_abrs[0])
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

