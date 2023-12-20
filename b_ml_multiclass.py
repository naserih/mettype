#ml_multiclassificatier
import csv
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import resample
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFECV, SelectFromModel #, SequentialFeatureSelector
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.multiclass import OneVsRestClassifier
import gc
import pickle
from datetime import datetime
from dotenv import load_dotenv
from scipy import interpolate
load_dotenv()
t0 = datetime.now()


# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.serif"] = ['Times']
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
feature_root = os.environ.get("RADIOMICS_FEATURES_PATH")
label_file = os.environ.get("LESION_CENTERS_WITH_LABEL")
output_results = os.environ.get("PAINDICATOR_RESULTS")
class features():
    def __init__(self, filepaths):
        self.values = {}
        self.labels = []
        self.df = []
        self.filenames = []
        for filepath in filepaths:
            value = []
            self.filenames.append(filepath)
            with open(filepath, 'r') as csv_file:
                csvreader = csv.reader(csv_file)
                # print filepath
                header = csvreader.next()
                for row in csvreader:
                    value.append(float(row[1]))
                    if row[0] not in self.values:
                        self.labels.append(row[0])
                        self.values[row[0]] = [float(row[1])]
                    else:
                        self.values[row[0]].append(float(row[1]))
            self.df.append(value)
            # print self.labels



def plot_1D(X, y, names, i_index):
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#10f972', '#ff0000']) 
    plt.scatter(names, X[:, i_index], c=y, cmap=cm_bright,
                       edgecolors='k')
    plt.show()
    

def get_label_metadata(label_file, label_column, label_tag):
    label_metadata = {}
    print('label_file: ', label_file)
    with open(label_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        # print header
        for row in csvreader:
            file_id = "_".join([row[13]]+row[14][1:-1].split(', '))
            label = row[label_column]
            if 'MET' in label:
                y_label = 'MET'
            elif 'CTRL' in label:
                y_label = 'CTRL'
            elif 'none' in label:
                y_label = 'no pain'
            elif 'mild' in label and 'g3' in label_tag:
                y_label = 'no pain'
            elif 'mild' in label and 'g4' in label_tag:
                y_label = 'mild'
            elif 'moderate' in label or 'severe' in label:
                y_label = 'pain'
            else:
                y_label = label
                # print 'UNK label', label
            if y_label in label_metadata:
                label_metadata[y_label]['row'].append(row)
                label_metadata[y_label]['file_id'].append(file_id)
            else:
                label_metadata[y_label] = {'row':[row],
                                                    'file_id' : [file_id]}
    # print label_metadata
    return label_metadata


def get_feature_space(feature_path, label_metadata, labels):
    X = None
    y = None

    radiomics_labels_0 = None
    radiomics_fetures_0 = None
    radiomics_fetures_1 = None
    radiomics_fetures_2 = None
    # print label_metadata[labels[0]]['file_id']
    # print ['_'.join(f.split('_')[:-1]) for f in os.listdir(feature_path)]
    label_0_csvs = [os.path.join(feature_path, f) for f in os.listdir(feature_path) if '_'.join(f.split('_')[:-1]) in label_metadata[labels[0]]['file_id'] and 'FAILED' not in f]
    label_1_csvs = [os.path.join(feature_path, f) for f in os.listdir(feature_path) if '_'.join(f.split('_')[:-1]) in label_metadata[labels[1]]['file_id'] and 'FAILED' not in f] 
    if len(labels) == 3:
        label_2_csvs = [os.path.join(feature_path, f) for f in os.listdir(feature_path) if '_'.join(f.split('_')[:-1]) in label_metadata[labels[2]]['file_id'] and 'FAILED' not in f] 
    
    # print label_0_csvs  
    # print label_1_csvs     
    radiomics_fetures_0 = features(label_0_csvs)
    radiomics_fetures_1 = features(label_1_csvs)
    if len(labels) == 3:
        radiomics_fetures_2 = features(label_2_csvs)
    
    radiomics_labels_0 = np.array(radiomics_fetures_0.labels)
    # radiomics_labels_1 = np.array(radiomics_fetures_1.labels)
    file_names = radiomics_fetures_0.filenames + radiomics_fetures_1.filenames
    if len(labels) == 3:
        file_names += radiomics_fetures_2.filenames
    # print 'feature space size:', np.array(radiomics_fetures_0.df).shape, np.array(radiomics_fetures_1.df).shape
    if len(labels) == 2:
        X = np.array(radiomics_fetures_0.df+radiomics_fetures_1.df)
        y = np.array([0]*len(radiomics_fetures_0.df)+[1]*len(radiomics_fetures_1.df))
    if len(labels) == 3:
        X = np.array(radiomics_fetures_0.df+radiomics_fetures_1.df+radiomics_fetures_2.df)
        y = np.array([0]*len(radiomics_fetures_0.df)+[1]*len(radiomics_fetures_1.df)
            +[2]*len(radiomics_fetures_2.df))
        
    # radiomics_labels = radiomics_labels_0
    # print 'HERE: ', X.shape, y.shape, radiomics_labels_0.shape
    return X, y, radiomics_labels_0, file_names


def run_classifiers(X, y, labels, features, rs_method, database, name_tag):
    classifiers = None
    X_train = None 
    X_test = None  
    y_train = None  
    y_test = None 
    md = 0 # start 
    mc = 16 # 16 total
    pt = datetime.now()
    # print(X)
    names = [ 
    "Gaussian Process",      #0
    "Linear SVM",            #1
    "Neural Net",            #2
    "Neural Net relu lbfgs", #3
    "AdaBoost",              #6
    "Random Forest 100",     #7
    "RBF SVM",               #10
    "Nearest Neighbors",     #11    
    "Decision Tree",         #12
    "Naive Bayes",           #13
    "QDA",                   #14
    "Bagging"                #15
    # "Random Forest",         #8
    # "Balanced Linear SVM",   #9
    # "Neural Net reg",        #4
    # "Neural Net reg lbfgs ", #5
    ][md:md+mc]
    classifiers = [
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        SVC(kernel="linear", C=1),
        MLPClassifier(alpha=1, max_iter=1000),
        MLPClassifier(solver='lbfgs', alpha=0.001,
        hidden_layer_sizes=(15,)),
        AdaBoostClassifier(),
        RandomForestClassifier(n_estimators=100, max_features="auto"),
        SVC(gamma=2, C=1),
        KNeighborsClassifier(3),
        DecisionTreeClassifier(max_depth=8),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        BaggingClassifier(KNeighborsClassifier(),
                       max_samples=0.5, max_features=0.5),
        # RandomForestClassifier(max_depth=8, n_estimators=4, max_features=2),
        # SVC(kernel="linear", class_weight="balanced", probability=True),
        # MLPClassifier(alpha=0.001, activation='logistic',
        # hidden_layer_sizes=(15,), max_iter=1000),
        # MLPClassifier(solver='lbfgs', alpha=0.001, activation='logistic',
        # hidden_layer_sizes=(15,), max_iter=1000),
       ][md:md+mc]
    # print labels
    
    #  Split dataset into training and test part
    # pt = datetime.now()
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.3, random_state=None, stratify = y)
    X_train, y_train = resampling(X_train, y_train, name_tag,rs_method)
    # print 
    print 'TRAN / TEST: \t %i : %i'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds())) 
    pt = datetime.now() 
    # print y_test

    # for i in range(len(labels)):
    #     plot_1D(X_test, y_test, range(len(y_test)), i)
    # print 'Total: ',len(y_test), '\t Class 0: ',len(y_test[y_test==0]),' \t Class 1: ', len(y_test[y_test==1])
    #i_index = 0
    #j_index = 1
    #X_mesh = []
    # iterate over classifiers
    #h = 300  # meshsize
    # print(X)
    #x_min, x_max = X[:, i_index].min() - .5, X[:, i_index].max() + .5
    #y_min, y_max = X[:, j_index].min() - .5, X[:, j_index].max() + .5
    # print X.shape
    #xx, yy = np.meshgrid(np.linspace(x_min, x_max, h),
    #                     np.linspace(y_min, y_max, h))
    # print X_mesh

    #X_mesh = np.array(X_mesh)

   
    # just plot the dataset first
    cm = plt.cm.RdYlGn_r
    cm_bright = ListedColormap(['#00ff00', '#ff0000']) 
    # cm_bright = ListedColormap(['#10f972', '#f60915']) ff0000
    
    # plt.show()
    clf = None
    processed_files = [f for f in 
            os.listdir(os.path.join(output_results,database, name_tag)) if '.npy' in f]
    for name, clf in zip(names, classifiers):
        #print(name)
        processed = False
        for processed_file in processed_files:
            if "%s_%s"%(name_tag, name) in processed_file:
                if skip_processed:
                    print "%s_%s, in processed"%(name_tag, name)
                    processed = True
                if reprocess_again:
                    processed = False
                    os.remove(os.path.join(output_results,database, name_tag, processed_file))
                    print "%s_%s, reprocessing"%(name_tag, name)
        if processed:
            print 'PROCESSED'
            continue
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        #print(scores.shape)
        cv = StratifiedKFold(n_splits=5)
        tprs = []
        fprs = []
        roc_aucs = []
        for i, (train, validation) in enumerate(cv.split(X_train, y_train)):
            # pt = datetime.now()
            #print(i)
            model = clf.fit(X_train[train], y_train[train])
            y_valid_multiclass = []
            for v in range(len(y_train[validation])):
                if y_train[validation][v] == 0:
                    y_valid_multiclass.append([1,0,0])
                elif y_train[validation][v] == 1:
                    y_valid_multiclass.append([0,1,0])
                elif y_train[validation][v] == 2:
                    y_valid_multiclass.append([0,0,1])
            y_valid_multiclass = np.array(y_valid_multiclass)
            
            if hasattr(clf, "decision_function"):
                y_probs = model.decision_function(X_train[validation])
            else:
                y_probs = model.predict_proba(X_train[validation])
            #print y_probs.shape
            #print y_probs 
            roc_auc = roc_auc_score(y_valid_multiclass, y_probs, average=None)
            #print("auc", auc)
            #print(y_valid_multiclass[:,1],y_probs[:,1])
            fpr = []
            tpr = []
            for v in range(y_probs.shape[1]):
                #print (v)
                fpr_s, tpr_s, _ = roc_curve(y_valid_multiclass[:,v], y_probs[:,v])
                err = 0
                for ix in range(len(fpr_s)):
                    fpr_s[ix] += err
                    tpr_s[ix] += err
                    err += 0.00001
                fit = interpolate.interp1d(fpr_s,tpr_s)
                fpr_fit = np.linspace(0, 1, 25)
                tpr_fit = fit(fpr_fit)
                fpr.append(fpr_fit)
                tpr.append(tpr_fit)
                #plt.plot(fpr_s,tpr_s, color='navy', lw=2, marker='d')
                #plt.plot(xp,yp, color='red', lw=1, marker='o')
                #plt.show()                
            tprs.append(tpr)
            fprs.append(fpr)
            roc_aucs.append(roc_auc)

        roc_aucs = np.array(roc_aucs)
        fprs = np.array(fprs)
        tprs = np.array(tprs)
        model = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        y_pred = model.predict(X_test)
        
        y_test_multiclass = []
        for v in range(len(y_test)):
            if y_test[v] == 0:
                y_test_multiclass.append([1,0,0])
            elif y_test[v] == 1:
                y_test_multiclass.append([0,1,0])
            elif y_test[v] == 2:
                y_test_multiclass.append([0,0,1])
        
        y_test_multiclass = np.array(y_test_multiclass)
        #print(y_test_multiclass)
        if hasattr(clf, "decision_function"):
                y_probs = model.decision_function(X_test)
        else:
                y_probs = model.predict_proba(X_test)

        #print(y_probs)

        roc_auc = roc_auc_score(y_test_multiclass, y_probs, average=None)
        p_r_f1 = precision_recall_fscore_support(y_test, y_pred, average=None)
        #print 'p_r_f1', p_r_f1
        fpr = []
        tpr = []
        for v in range(y_probs.shape[1]):
            #print (v)
            fpr_s, tpr_s, _ = roc_curve(y_test_multiclass[:,v], y_probs[:,v])

            fpr.append(fpr_s)
            tpr.append(tpr_s)
            
        
        fig_roc = plt.figure()
        #print(roc_auc)
        #print(roc_aucs)
        #inplot_label_v = 'AUCs (validation): %s' % ('_'.join( ['%0.3f'%(roc_aucs[:,i].mean()) 
        #                                        for i in range(roc_aucs.shape[1])]))
        #print(inplot_label_v)
        cnt = 0
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", 
                    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        for v in range(tprs.shape[1]): 
            inplot_label_t = '{:10s} AUC={:0.3f}'.format(labels[v], roc_auc[v])
            plt.fill_between(fprs[0,0], tprs[0,v],tprs[1,v], color=colors[v], alpha=.1)
            plt.fill_between(fprs[0,0], tprs[1,v],tprs[2,v], color=colors[v], alpha=.1)
            plt.fill_between(fprs[0,0], tprs[2,v],tprs[3,v], color=colors[v], alpha=.1)
            plt.fill_between(fprs[0,0], tprs[3,v],tprs[4,v], color=colors[v], alpha=.1)
            plt.plot(fpr[v] , tpr[v], color=colors[v], marker = 's',
                mfc='None', ms = 4,
                     lw=1, label=inplot_label_t)

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('%s \n %s'%(' '.join(name_tag.split('_')[1:]), name))
        plt.legend(loc="lower right")
            
        cnt += 1
        #plt.show()
        out_file_name = "%s/%s/%s/%s_%s_ROC_%0.3f_%0.3f"%(output_results,database,name_tag,name_tag,name,roc_auc[0],score)
        plt.savefig(out_file_name+'.png')
        plt.close(fig_roc)
        fig_roc.clear()
        fig_roc.clf()
        gc.collect()

        print 'PLOT ROCS : \t %i %i %s'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()), name)
        for v in range(tprs.shape[1]):
            out_file_name = "%s/%s/%s/%s_%s_%s_ROC_%0.3f_%0.3f"%(output_results, database,name_tag,name_tag, name,
                labels[v],roc_auc[v], score)
            # print '--->>>', labels[v]
            with open (out_file_name+'.csv', 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                for j in range(len(tprs)):
                    csvwriter.writerow(tprs[j,v])
                    csvwriter.writerow(fprs[j,v])
                csvwriter.writerow(tpr[v])
                csvwriter.writerow(fpr[v])
            #print score
            #print p_r_f1
            with open (out_file_name+'.npy', 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['AUC1','AUC2','AUC3','AUC4','AUC5',
                        'R2_1','R2_2','R2_3','R2_4','R2_5',
                        'AUC', 'R2','P','R','F1'])
                #print('v_AUC', roc_aucs[:,v])
                #print('v_R2', scores)
                #print('AUC', roc_auc[v])
                #print('R2', score)
                #print('P','R','F1', p_r_f1[v])
                csvwriter.writerow(np.concatenate((roc_aucs[:,v], scores, 
                    [roc_auc[v] , score],np.array(p_r_f1[v]))))
        
    clf = None
    name = None    
    X_mesh = None
    xx = None
    yy = None   
    fig_roc = None
    c_m = None  
    cset1 = None
    Z = None

def resampling(X,y,name_tag,method):
    y_0 = np.where(y == 0)[0]
    y_1 = np.where(y == 1)[0]
    # print y_0.shape
    X_0 = X[y_0]
    X_1 = X[y_1]
    # print 'before resampling: ', X_0.shape, X_1.shape
    if method == 'UP':
        y_0 = np.array([0]*len(y_1))
        X_0 = resample(X_0, 
            replace=True,     # sample with replacement
            n_samples=len(y_1),    # to match majority class
            random_state=123) # reproducible results
        X = np.concatenate((X_0, X_1)) 
        y = np.concatenate((y_0, y_1)) 
    elif method == 'DOWN':
        y_1 = np.array([1]*len(y_0))
        X_1 = resample(X_1, 
            replace=True,     # sample with replacement
            n_samples=len(y_0),    # to match majority class
            random_state=123)
        X = np.concatenate((X_0, X_1))
        y = np.concatenate((y_0, y_1)) 
    elif method == 'ROS':
        sampler = RandomOverSampler()
        X, y = sampler.fit_sample(X, y)
    elif method == 'SMOTE':
        sampler = SMOTE(ratio='minority')
        X, y = sampler.fit_sample(X, y)
    elif method == 'RUS':
        sampler = RandomUnderSampler(return_indices=False)
        X, y = sampler.fit_sample(X, y)
    elif method == 'TL':
        sampler = TomekLinks(return_indices=False, ratio='majority')
        X, y = sampler.fit_sample(X, y)
    elif method == 'NONE':
        return X, y
    else:
        print '''WARNING: INVALID RESAMPLING METHOD.
        resampling methodes are; 
        NONE: No resampling
        DOWN: Random reproducable DOWN sampling
        RUS:Random under-sampling 
        TL: Tomek links
        UP: Random reproducable UP sampling
        ROS: random over-sampling
        SMOTE: Synthetic Minority Oversampling TEchnique
        '''

    sampler = None    
    return X, y




def feature_selection(X,y, database, name_tag, method):
    # Create the RFE object and compute a cross-validated score.
    svc = None
    knn = None
    svc = SVC(kernel="linear")
    knn = KNeighborsClassifier(3),
    
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    if method == 'PCA':
        n_components = 'mle'
        transformer = PCA(n_components=n_components,
            # svd_solver = 'arpack'
            )
    elif method == 'PCA_24':
        n_components = 24
        transformer = PCA(n_components=n_components)
    elif method == 'PCA_20':
        n_components = 20
        transformer = PCA(n_components=n_components)
    elif method == 'PCA_30':
        n_components = 30
        transformer = PCA(n_components=n_components)
    elif method == 'FastICA_24':
        n_components = 24
        transformer = FastICA(n_components=n_components, random_state=0)
    elif method == 'FastICA_30':
        n_components = 30
        transformer = FastICA(n_components=n_components, random_state=0)
    elif method == 'FastICA_20':
        n_components = 20
        transformer = FastICA(n_components=n_components, random_state=0)
    elif method == 'PFECV':
        min_features_to_select = 10
        transformer = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                  scoring='accuracy',
                  min_features_to_select=min_features_to_select)
    elif method == 'LASSO':
        transformer = LinearSVC(C=0.5, penalty="l1", dual=False)
    elif method == 'LASSO_0.1':
        transformer = LinearSVC(C=0.1, penalty="l1", dual=False)
    elif method == 'LASSO_1':
        transformer = LinearSVC(C=1, penalty="l1", dual=False)
    elif method == 'TREE':
        transformer = ExtraTreesClassifier(n_estimators=50)
    elif method == 'VT_0.8':    
        transformer = VarianceThreshold(threshold=(.8 * (1 - .8)))
    elif method == 'VT_0.0':    
        transformer = VarianceThreshold()
    elif method == 'NONE':
        return X
    else:
        print 'UNKNOWN FREATURE SELECTION METHOD' 
    try:
        model = SelectFromModel(transformer.fit(X,y), prefit=True)
        X = model.transform(X)
        model = None
    except:
        print 'INFO: transformer used directrly'
        transformer.fit(X,y)
        X = transformer.transform(X)


    # if method == 'PFECV':
    #     fig = None
    #     # print "Optimal number of features : %d" %transformer.n_features_
    #     fig = plt.figure()
    #     plt.xlabel("Number of features selected")
    #     plt.ylabel("Cross validation score (nb of correct classifications)")
    #     plt.plot(range(min_features_to_select,
    #                    len(transformer.grid_scores_) + min_features_to_select),
    #              transformer.grid_scores_)
    #     # plt.show()
    #     plt.savefig("%s%s/%s_%s.jpg"%(output_results, name_tag,name_tag, transformer.n_features_))
    #     fig.clear()
    #     fig.clf()
    #     plt.close('all')
    #     plt.close(fig)
    #     gc.collect()
    transformer = None

    return X

def plot_2d_space(X, y, name_tag):   
    markers = ['D', 's']
    label_names = ['Pain', 'No Pain']
    colors = ['#FF0000', '#00FF00']
    edgecolors = ['#800000', '#008000']
    fig = plt.figure()
    for l, ln, c, m, ec, in zip([1,0], label_names, colors, markers, edgecolors):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=ln, marker=m, edgecolor=ec
        )
    plt.title(name_tag)
    plt.legend(loc='upper right')
    # print output_results
    plt.savefig("%s/%s/%s/%s_resampling.jpg"%(output_results,database, name_tag,name_tag))
    plt.close()
    plt.close('all')
    plt.close(fig)
    fig.clear()
    fig.clf()
    gc.collect()

def plot_feature_vs_class(X,y,feature_names, file_names, name_tag):
    # print set(y)
    marker = ['D', 's']
    label_names = ['Pain', 'No Pain']
    colors = ['#fc3339', '#008001']
    edgecolors = ['r', 'g']
    for iy in range(X.shape[1])[:]:
        # print ix
        data = []
        for label in [1,0]: 
            row_iy = np.where(y == label)[0]
            data.append(X[row_iy, iy])
        fig = plt.figure()
        plt.ylabel(feature_names[iy])
        plt.violinplot(data)
        plt.xticks( [1,2], label_names)
        plt.savefig("%s%s/%s_Viol_%i_%s.jpg"%(output_results, name_tag, name_tag,iy,feature_names[iy]))
        plt.close()
        plt.close('all')
        plt.close(fig)
        fig.clear()
        fig.clf()
        gc.collect()

def main():
    database = 'LBM' #'PnPg4'# 'HM' # 'PnP' #'PnPNLP', PnPg4, PnPg4
    r = 0 # roi
    re = r+36 # 30 total
    o = 0 # 5 RS  0:NONE 1:SMOTE 2:TL 3:ROS 4:RUS 
    oe =o+2 # 5 #RS methods
    n = 5 # 14 FS    
    ne = n+14 #14 FS methods
    '''
            0:NONE 1:TREE
            2:LASSO 3:LASSO_1 4:LASSO_0.1, 
            5:PFECV, 6:VT_0.8 7:VT_0.0',
            8:FastICA_20 9:FastICA_24 10:FastICA_30, 
            11:PCA_20 12:PCA_24 13:PCA_30

    '''
    rois = [    'SP50', 'CY50', #0:2
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
                'EIT3CY', 'EIT3SP', 'EIT6SC', #27:30 
                'E6CY', 'E6SP', 'E12SC', #30:33 
                'EIT6CY', 'EIT6SP', 'EIT12SC', #33-36 
                ][r:re]
    
    data_resampling_methods = ['NONE', 'SMOTE', 'TL', 'ROS', 'RUS'][o:oe]
    feature_selection_methods = ['PFECV', 'NONE', 'LASSO', 'LASSO_1','LASSO_0.1', 'TREE',
              'VT_0.8', 'VT_0.0',
            'FastICA_20', 'FastICA_24', 'FastICA_30', 
            'PCA_20', 'PCA_24', 'PCA_30'][n:ne]
    
    
    all_rois = { 
            
            'E3SP' :    [ 'SP15','SP10','SP7'],
            'E3CY' :    ['CY15','CY10','CY7'],
            'E6SP' :    ['SP50', 'SP30', 'SP20', 'SP15','SP10','SP7'],
            'E6CY' :    ['CY50', 'CY30','CY20','CY15','CY10','CY7'],
            'E6SC' :   ['SP15','SP10','SP7','CY15','CY10','CY7'],
            'E12SC' :   ['SP50', 'SP30', 'SP20', 'SP15','SP10','SP7',
                        'CY50', 'CY30','CY20','CY15','CY10','CY7'],
            'EIT3SP' :  ['SPIT15','SPIT10', 'SPIT7'],
            'EIT3CY' :  ['CYIT15','CYIT10','CYIT7'],
            'EIT6SP' :  ['SPIT50', 'SPIT30', 'SPIT20', 'SPIT15','SPIT10', 'SPIT7'],
            'EIT6CY' :  ['CYIT50', 'CYIT30','CYIT20','CYIT15','CYIT10','CYIT7'],
            'EIT6SC' : [ 'SPIT15','SPIT10', 'SPIT7','CYIT15','CYIT10','CYIT7'],
            'EIT12SC' : ['SPIT50', 'SPIT30', 'SPIT20', 'SPIT15','SPIT10', 'SPIT7',
                            'CYIT50', 'CYIT30','CYIT20','CYIT15','CYIT10','CYIT7'],
            
            # 'E4SP' :    ['SP50', 'SP30', 'SP20', 'SP15'],
            # 'E4CY' :    ['CY50', 'CY30','CY20','CY15'],
            # 'E5CY' :    ['CY50', 'CY30', 'CY20','CY15', 'CY5030'],
            # 'E9SC' :    [ 'SP50','CY50', 'SP30','CY30', 
            #             'SP20', 'CY20', 'CY15','SP15', 'CY5030',
            #             ],
            # 'EIT4SP' :    ['SPIT50', 'SPIT30', 'SPIT20', 'SPIT15'],
            # 'EIT4CY' :    ['CYIT50', 'CYIT30','CYIT20','CYIT15'],
            # 'EIT8SC' :    [ 'SPIT50','CYIT50', 'SPIT30','CYIT30', 
            #             'SPIT20', 'CYIT20', 'CYIT15','SPIT15'],
            }
    t_cnt = len(feature_selection_methods)*len(data_resampling_methods)*len(rois)
    dts = [0] 
    cnt = 0

    if database == 'HM':
        label_column = 3 # 3 cntrl/met for hm  # 13 met type
        labels = ['CTRL', 'MET']  
    elif database == 'PnP':
        label_column = 8 #  8 for pnp
        labels = ['pain', 'no pain'] 
    elif database == 'PnPNLP':
        label_column = 9 #  9 for pnp_nlp
        labels = ['pain', 'no pain']  

    elif database == 'PnPg3':
        label_column = 10 #  9 for pnp_nlp
        labels = ['pain', 'no pain']  
    elif database == 'PnPg4':
        label_column = 10 #  9 for pnp_nlp
        labels = ['pain', 'no pain']  
    elif database == 'LBM':
        label_column = 16 #  9 for pnp_nlp
        labels = ['mix', 'blastic', 'lytic']  
    
    else:
        print('label_column undefined: ', label_column)
    pt = datetime.now()
    label_metadata = get_label_metadata(label_file,label_column, database)

    # print label_metadata
    print('labels:',label_metadata.keys())
    
    print '--------------'
    print 'GOT LABELS: \t %i : %i'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds())) 
    for label in labels:
        # print label, len(label_metadata[label]['file_id'])
        if label not in label_metadata.keys():
            print 'LABEL ERROR: ', label
            break
    '''
    resampling: 0, 
    feature selection: 0-8
    '''
 
    # print rois
    for roi in rois:
            feature_path = '%s/%s'%(feature_root,roi)
            for rs_method in data_resampling_methods:
                gc.collect()
                # print roi
                if os.path.exists(feature_path) or 'E' in roi:
                    # print 'shape: ', X.shape, y.shape, feature_names.shape
                    # print y
                    # Standardize features by removing the mean and scaling to unit variance
                    # print 'RESAMPELED: ', X.shape, y.shape
                    # plot_feature_vs_class(X,y,feature_names, file_names, name_tag)
                    for fs_method in feature_selection_methods:
                        cnt +=1 
                        name_tag = '%s_%s_%s_%s'%(database,roi,fs_method,rs_method)
                        print '-------------'
                        print name_tag
                        pt = datetime.now()
                        t1 = datetime.now()
                        X = None 
                        y = None
                        if  'E' in roi: # 'E stands for ensemble ROIs'
                            Xs = []
                            ys = []
                            rfile_names = []
                            rfeature_names = []
                            f_dic = {}
                            f_names = []

                            for r in all_rois[roi]: 
                                r_feature_path = '%s/%s'%(feature_root,r)
                                rX, ry, rfeature_name, rfile_name = get_feature_space(r_feature_path, label_metadata, labels)
                                f_names.append(rfeature_name)
                                #print len(rfile_name)
                                #print rX.shape
                                for fi in range(len(rfile_name)):
                                    point_name = '_'.join((rfile_name[fi].split('/')[-1]).split('_')[:-1])
                                    if point_name in f_dic:
                                        f_dic[point_name][0].append(rX[fi,:])
                                        f_dic[point_name][1].append(ry[fi])
                                    else:
                                        f_dic[point_name] = [[rX[fi,:]],[ry[fi]]]
                            X = []
                            y = []
                            for point_name in f_dic:
                                if len(f_dic[point_name][0]) == len(all_rois[roi]):
                                    # print len(f_dic[point_name][0][0])
                                    all_f = np.concatenate(f_dic[point_name][0], axis=0)
                                    # print len(all_f)
                                    # print f_dic[point_name][1]
                                    X.append(all_f)
                                    y.append(f_dic[point_name][1][0])

                                
                            # print rfile_name
                            X = np.array(X)
                            y = np.array(y)
                            #print X.shape, y.shape
                            feature_names = np.concatenate(f_names, axis=0)


                        else:
                            X, y, feature_names, _ = get_feature_space(feature_path, label_metadata, labels)
                        
                       
                        # print 'FEATURE SPACE: \t %i : %i'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds())) 
                        # pt = datetime.now()
                        
                        X = StandardScaler().fit_transform(X)
                        # print 'X NRMILIZED: \t %i : %i'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds())) 
                        # print ('X', X)
                        pt = datetime.now()
                        if not os.path.exists(os.path.join(output_results,database)):
                            os.mkdir(os.path.join(output_results, database))
                        if not os.path.exists(os.path.join(output_results,database, name_tag)):
                            os.mkdir(os.path.join(output_results, database,name_tag))
                        pt = datetime.now()
                        X = feature_selection(X,y, database,name_tag, fs_method)
                        print  'X.shape', X.shape
                        print 'FEATURE REDUC: \t %i : %i'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds())) 
                        # plot_2d_space( X, y,name_tag)
                        run_classifiers(X, y, labels, feature_names, rs_method, database,name_tag)
                        dt = (datetime.now() - t1).total_seconds()
                        # print X.shape
                        if dt > 20:
                            dts.append(dt)
                        print  '%i/%i REMAINING TIME: %6.2f hrs'%(cnt, t_cnt, np.median(dts)*(t_cnt-cnt)/3600.0)




skip_processed = False      ## TRUE: SKIP FALSE: DO NOT REMOVE OLD RESULTS


### DO NOT MAKE THIS TRUE  UNLESS IF YOU WANT TO DELETE
reprocess_again = False    ## TRUE: REMOVES OLD RESULTS
# save_model = False
###
###
if __name__ == "__main__":
    main()
