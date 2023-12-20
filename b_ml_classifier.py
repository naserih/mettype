#ml_classificatier
import csv
import random
import os
import numpy as np
import pandas as pd
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
from sklearn.feature_selection import f_regression, VarianceThreshold, SelectKBest, chi2, RFECV, SelectFromModel #, SequentialFeatureSelector
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import gc
import pickle
from datetime import datetime
from dotenv import load_dotenv
#from keras.layers import Dense
#from keras.models import Sequential
load_dotenv()
t0 = datetime.now()


# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.serif"] = ['Times']
plt.rcParams['font.size'] = 14
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
            with open(filepath, 'r', newline='') as csv_file:
                csvreader = csv.reader(csv_file)
                # print filepath
                header = next(csvreader)
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
    with open(label_file, 'r', newline='') as csvfile:
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
    label_0_csvs = [os.path.join(feature_path, f) for f in os.listdir(feature_path) if '_'.join(f.split('_')[:-1]) in label_metadata[labels[0]]['file_id'] and 'FAILED' not in f]
    label_1_csvs = [os.path.join(feature_path, f) for f in os.listdir(feature_path) if '_'.join(f.split('_')[:-1]) in label_metadata[labels[1]]['file_id'] and 'FAILED' not in f] 
    if len(labels) == 3:
        label_2_csvs = [os.path.join(feature_path, f) for f in os.listdir(feature_path) if '_'.join(f.split('_')[:-1]) in label_metadata[labels[2]]['file_id'] and 'FAILED' not in f] 
    
    radiomics_fetures_0 = features(label_0_csvs)
    radiomics_fetures_1 = features(label_1_csvs)
    if len(labels) == 3:
        radiomics_fetures_2 = features(label_2_csvs)
    
    radiomics_labels_0 = np.array(radiomics_fetures_0.labels)
    # radiomics_labels_1 = np.array(radiomics_fetures_1.labels)
    file_names = radiomics_fetures_0.filenames + radiomics_fetures_1.filenames
    if len(labels) == 3:
        file_names += radiomics_fetures_1.filenames
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

def NNet_imb(n_input):
    # define model
    model = Sequential()
    # define first hidden layer and visible layer
    model.add(Dense(100, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
    # define output layer
    model.add(Dense(1, activation='sigmoid'))
    # define loss and optimizer
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    return model

def run_classifiers(X, y, labels, rs_method, database, name_tag):
    classifiers = None
    X_train = None 
    X_test = None  
    y_train = None  
    y_test = None 
    # md = 0 # start 
    # mc = 16 # 16 total
    pt = datetime.now()
    # print(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.3, random_state=None, stratify = y)
    X_train, y_train = resampling(X_train, y_train, name_tag,rs_method)
    # print 
    print ('TRAN / TEST: \t %i : %i'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds())))
    #print (y_test)
    pt = datetime.now() 
    
    classifiers = {
        "GPR" : GaussianProcessClassifier(1.0 * RBF(1.0)),
        "L_SVM" : SVC(kernel="linear", C=1),
        "NNet" : MLPClassifier(alpha=1, max_iter=1000),
        "NNet_lbfgs" : MLPClassifier(solver='lbfgs', alpha=0.001,
        hidden_layer_sizes=(15,)),
        # NNet_imb(n_input = X_train.shape[1]),
        "AdaBoost" : AdaBoostClassifier(),
        "RF": RandomForestClassifier(n_estimators=100, max_features="auto"),
        # RandomForestClassifier(n_estimators=100, max_features="auto",
            # class_weight='balanced'),
        # RandomForestClassifier(n_estimators=100, max_features="auto",
            # class_weight='balanced_subsample'),
        "SVM" : SVC(gamma=2, C=1),
        "kNN" : KNeighborsClassifier(3),
        "DT": DecisionTreeClassifier(max_depth=8),
        "NB" : GaussianNB(),
        "QDA" : QuadraticDiscriminantAnalysis(),
        "Bagging" : BaggingClassifier(KNeighborsClassifier(),
                       max_samples=0.5, max_features=0.5),
        # RandomForestClassifier(max_depth=8, n_estimators=4, max_features=2),
        # SVC(kernel="linear", class_weight="balanced", probability=True),
        # MLPClassifier(alpha=0.001, activation='logistic',
        # hidden_layer_sizes=(15,), max_iter=1000),
        # MLPClassifier(solver='lbfgs', alpha=0.001, activation='logistic',
        # hidden_layer_sizes=(15,), max_iter=1000),
       }
    clf = None
    processed_files = [f for f in 
            os.listdir(os.path.join(output_results,database, name_tag)) if '.npy' in f]
    for name, clf in classifiers.items():
        
        processed = False
        for processed_file in processed_files:
            if "%s_%s"%(name_tag, name) in processed_file:
                if skip_processed:
                    print ("%s_%s, in processed"%(name_tag, name))
                    processed = True
                if reprocess_again:
                    processed = False
                    os.remove(os.path.join(output_results,database, name_tag, processed_file))
                    print ("%s_%s, reprocessing"%(name_tag, name))
        if processed:
            print ('PROCESSED')
            continue
        # scores = cross_val_score(clf, X_train, y_train, cv=5)
        # score = clf.score(X_test, y_test)
        cv = StratifiedKFold(n_splits=5)
        tprs = []
        fprs = []
        tns = []
        tps = []
        fps = []
        fns = []
        roc_aucs = []
        ps = []
        rs = []
        f1s = []
        for i, (train, validation) in enumerate(cv.split(X_train, y_train)):
            # pt = datetime.now()
            # print(y_train[train])
            clf.fit(X_train[train], y_train[train])
            if name == 'NNet_b': 
                weights = {0:1, 1:5}
                model = clf.fit(X_train[train], y_train[train], 
                class_weight=weights, epochs=100, verbose=0)
            
            y_pred = clf.predict(X_train[validation]).ravel()
            # print 'FIT TRAIN %i: \t %i : %i'%(i, int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()))
            if hasattr(clf, "decision_function"):
                y_proba = clf.decision_function(X_train[validation])
            elif hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(X_train[validation])[:, 1]
            else:
                y_proba = y_pred
            y_pred[y_pred<0.5] = 0
            y_pred[y_pred>=0.5] = 1

            c_m = confusion_matrix(y_train[validation], y_pred)
            tns.append(c_m[0, 0])
            fps.append(c_m[0, 1])
            fns.append(c_m[1, 0])
            tps.append(c_m[1, 1])
            fpr, tpr, _ = roc_curve(y_train[validation], y_proba)
            tprs.append(tpr)
            fprs.append(fpr)
            roc_auc = auc(fpr, tpr)
            roc_aucs.append(roc_auc)
            p_r_f1 = precision_recall_fscore_support(y_train[validation], y_pred, average='macro')
            ps.append(p_r_f1[0])
            rs.append(p_r_f1[1])
            f1s.append(p_r_f1[2])


        # model = clf.fit(X_train, y_train)
        # score = clf.score(X_test, y_test)

        if save_mode:
            model_filename = "%s/%s/%s/%s_%s.sav"%(output_results, database,name_tag,name_tag, name)
            pickle.dump(clf, open(model_filename, 'wb'))

        y_pred = clf.predict(X_test).ravel()

        if hasattr(clf, "decision_function"):
            y_proba = clf.decision_function(X_test)
        elif hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred
        y_pred[y_pred<0.5] = 0
        y_pred[y_pred>=0.5] = 1
        # print(y_proba)
        # print(y_pred)
        # print('vvvvvvvvvv')
        # print(y_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        c_m = confusion_matrix(y_test, y_pred)

        p_r_f1 = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        #print(classification_report(y_test, y_pred, target_names=labels))
        #print(c_m)

        # print 'y_test', y_test
        # print 'y_pred', y_pred
        # print 'p_r_f1', p_r_f1
        # print 'PREC_RECAL: \t %i %i %s'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()), name)         
        tn = c_m[0, 0]
        fp = c_m[0, 1]
        fn = c_m[1, 0]
        tp = c_m[1, 1]

        # print 'model: %s score: %0.3f \t tp: %s  tn: %s  fp: %s  fn: %s '%(name, score, tp, tn, fp, fn)
        # std_tpr = np.std(tprs, axis=0) 
        # mean_tpr = np.mean(tprs, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
        #                 label=r'$\pm$ 1 std. dev.')
        # pt = datetime.now()
        fig_roc = plt.figure()
        
        inplot_label = 'Test, AUC: %0.3f, F1: %0.3f' % (roc_auc, p_r_f1[2])
        inplot_label_5fold = 'Validation, AUC: %0.3f, F1:%0.3f' % (np.mean(roc_aucs), np.mean(f1s))
        cnt = 0
        for v_tpr, v_fpr in zip(tprs, fprs):
            # print v_fpr 
        # for v in r
            if cnt != 0:
                inplot_label_5fold = ''
            plt.plot(v_fpr, v_tpr, color='pink',
                 lw=1, label=inplot_label_5fold, alpha=1)
            cnt += 1
        # plt.show()
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.plot(fpr, tpr, color='brown', marker = 's',
            mfc='None', ms = 7,
                 lw=1, label=inplot_label)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('%s \n %s'%(' '.join(name_tag.split('_')[1:]), name))
        plt.legend(loc="lower right")
        out_file_name = "%s/%s/%s/%s_%s_ROC_%0.3f_%0.3f"%(
            output_results,database ,name_tag,name_tag, name,roc_auc, p_r_f1[2])
        plt.savefig(out_file_name+'.png')
        out_file_name = "%s/%s/%s/%s_%s_ROC_%0.3f_%0.3f"%(
            output_results, database,name_tag,name_tag, name,roc_auc, p_r_f1[2])
        with open (out_file_name+'.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            for j in range(len(tprs)):
                csvwriter.writerow(tprs[j])
                csvwriter.writerow(fprs[j])
            csvwriter.writerow(tpr)
            csvwriter.writerow(fpr)

        # plt.show()
        plt.close(fig_roc)
        fig_roc.clear()
        fig_roc.clf()
        gc.collect()
        with open (out_file_name+'.npy', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['P_v','R_v','F1_v','AUC_v',
                    'TP_v','TN_v','FP_v','FN_v',
                    'P_t','R_t','F1_t','AUC_t',
                    'TP_t','TN_t','FP_t','FN_t',
                    ])
            #print(p_r_f1)
            array = np.concatenate(([np.mean(ps),
                np.mean(rs),
                np.mean(f1s),
                np.mean(roc_aucs), 
                # np.mean(scores),
                np.mean(tps),
                np.mean(tns),
                np.mean(fps),
                np.mean(fns)],
                p_r_f1[0:3] , [roc_auc],
                [tp, tn, fp, fn]))
            csvwriter.writerow(array)

        print ('PLOT ROCS : \t %i %i %s'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()), name))
         

        # Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
        pt = datetime.now()
        if X.shape[1] == 2:
            # print 'HERE'
            fig, ax = plt.subplots(nrows=1, ncols=1)
            fig.subplots_adjust(hspace=0.3)
            pt = datetime.now()
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            # print 'MESH CALC: \t %i %i %s'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()), name)
            pt = datetime.now()
            # print 'Z.shape', Z.shape
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            # print Z.shape
            cset1 = ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            fig.colorbar(cset1, ax=ax)

            # Plot the training points
            ax.set_title("%s_ROI:%s"%(name, name_tag))
            ax.scatter(X_train[:, i_index], X_train[:, j_index], c=y_train, cmap=cm_bright,
                   alpha=0.3, edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, i_index], X_test[:, j_index], c=y_test, cmap=cm_bright, 
                alpha=0.6,   edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            # ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    # size=15, horizontalalignment='right')
            
                # print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
            plt.tight_layout()
            saved_files = {}
            for f in os.listdir(output_results):
                saved_files['_'.join(f.split('_')[:-1])] = f.split('_')[-1] 
            # print saved_files
            if ("%s_%s"%(name, name_tag) not in saved_files  or saved_files["%s_%s"%(name, name_tag)] < score) \
                and score > 0.65:
                plt.savefig("%s/%s/%s/%s_%s_%0.3f.jpg"%(output_results,database, name_tag, name_tag, name, score))
            # plt.show()
            plt.close('all')
            plt.close(fig)
            fig.clear()
            fig.clf()
            gc.collect()
            fig = None
            ax = None
            # print 'PLOT 2D: \t %i %i %s'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()), name)
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
        sampler = SMOTE(sampling_strategy='minority')
        X, y = sampler.fit_resample(X, y)
    elif method == 'RUS':
        sampler = RandomUnderSampler(return_indices=False)
        X, y = sampler.fit_sample(X, y)
    elif method == 'TL':
        sampler = TomekLinks(return_indices=False, ratio='majority')
        X, y = sampler.fit_sample(X, y)
    elif method == 'NONE':
        return X, y
    else:
        print ('''WARNING: INVALID RESAMPLING METHOD.
        resampling methodes are; 
        NONE: No resampling
        DOWN: Random reproducable DOWN sampling
        RUS:Random under-sampling 
        TL: Tomek links
        UP: Random reproducable UP sampling
        ROS: random over-sampling
        SMOTE: Synthetic Minority Oversampling TEchnique
        ''')
    sampler = None    
    return X, y




def feature_selection(X,y, feature_names, database, name_tag, method):
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
                  scoring='f1_macro',
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
        print ('UNKNOWN FREATURE SELECTION METHOD') 
    try:
        model = SelectFromModel(transformer.fit(X,y), prefit=True)
        X = model.transform(X)
        model = None
    except:
        print ('INFO: transformer used directrly')
        model = transformer.fit(X,y)
        X = transformer.transform(X)

    #feat_importances = pd.Series(model.feature_importances_, index=feature_names)
    #model.feature_importances_
    f_stats, p_vals = f_regression(X,y)


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

    return X, f_stats, p_vals

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
    database = 'PnPNLP'#'LBM' #'PnPg4'# 'HM' # 'PnP' #'PnPNLP', PnPg4, PnPg4
    r = 0 # roi
    re = r+2 # 30 total
    o = 1 # 5 RS  0:NONE 1:SMOTE 2:TL 3:ROS 4:RUS 
    oe =o+1 # 5 #RS methods
    n = 2 # 14 FS    
    ne = n+1 #14 FS methods
    '''
            0:NONE 1:TREE
            2:LASSO 3:LASSO_1 4:LASSO_0.1, 
            5:PFECV, 6:VT_0.8 7:VT_0.0',
            8:FastICA_20 9:FastICA_24 10:FastICA_30, 
            11:PCA_20 12:PCA_24 13:PCA_30

    '''
    rois = [    'ENIT3', 'ENIT6', #27:30 
                'SP50', 'CY50', #0:2
                'CY5030',
                'SP30', 'CY30', #2:4
                'SP20', 'CY20', #4:6
                'SP15', 'CY15', #6:8
                'E4CY', 'E4SP', 'E5CY', 'E9SC', #24:27
                'EN3', 'EN6',
                'ENIT3', 'ENIT6',
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
    feature_selection_methods = ['NONE',
             'LASSO', 'LASSO_0.1', 'LASSO_1', 'PFECV',
              'VT_0.8', 'VT_0.0', 'TREE',
            'FastICA_20', 'FastICA_24', 'FastICA_30', 
            'PCA_20', 'PCA_24', 'PCA_30'][n:ne]
    
    
    all_rois = {  
            'ENIT3' :  ['SPIT15','SPIT10', 'SPIT7'],
            'ENIT6' :  ['SPIT50', 'SPIT30', 'SPIT20', 'SPIT15','SPIT10', 'SPIT7'],
            'EN3' :    [ 'SP15','SP10','SP7'],
            'EN6' :    ['SP15','SP10','SP7','CY15','CY10','CY7'],
            'EN12' :    ['SP15','SP10','SP7','CY15','CY10','CY7',
            'SPIT50', 'SPIT30', 'SPIT20', 'SPIT15','SPIT10', 'SPIT7'],
            
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
            
            'E4SP' :    ['SP50', 'SP30', 'SP20', 'SP15'],
            'E4CY' :    ['CY50', 'CY30','CY20','CY15'],
            'E5CY' :    ['CY50', 'CY30', 'CY20','CY15', 'CY5030'],
            'E9SC' :    [ 'SP50','CY50', 'SP30','CY30', 
                        'SP20', 'CY20', 'CY15','SP15', 'CY5030',
                        ],
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
    if database == 'HM_RV':
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
    
    print ('--------------')
    print ('GOT LABELS: \t %i : %i'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds())) )
    for label in labels:
        # print label, len(label_metadata[label]['file_id'])
        if label not in label_metadata.keys():
            print ('LABEL ERROR: ', label)
            break
    '''
    resampling: 0, 
    feature selection: 0-8)
    '''
 
    # print rois
    for roi in rois:
            feature_path = '%s/%s'%(feature_root,roi)
            for rs_method in data_resampling_methods:
                gc.collect()
                # print roi
                if os.path.exists(feature_path) or 'E' in roi:
                   for fs_method in feature_selection_methods:
                        cnt +=1 
                        name_tag = '%s_%s_%s_%s'%(database,roi,fs_method,rs_method)
                        print ('-------------')
                        print (name_tag)
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
                                #rX = np.array(rX)
                                f_names.append(rfeature_name)
                                for fi in range(len(rfile_name)):
                                    point_name = '_'.join((rfile_name[fi].split('/')[-1]).split('_')[:-1])
                                    #print (rfile_name[fi], fi, rX[0])
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
                        X, f_stats, p_vals = feature_selection(X,y, feature_names, database,name_tag, fs_method)
                        print  ('X.shape', X.shape)
                        print  ('f_stats', f_stats)
                        print  ('p_vals', p_vals)

                        print ('FEATURE REDUC: \t %i : %i'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds())) )
                        # plot_2d_space( X, y,name_tag)
                        run_classifiers(X, y, feature_names, rs_method, database,name_tag)
                        dt = (datetime.now() - t1).total_seconds()
                        # print X.shape
                        if dt > 20:
                            dts.append(dt)
                        print  ('%i/%i REMAINING TIME: %6.2f hrs'%(cnt, t_cnt, np.median(dts)*(t_cnt-cnt)/3600.0))




skip_processed = False      ## TRUE: SKIP FALSE: DO NOT REMOVE OLD RESULTS


### DO NOT MAKE THIS TRUE  UNLESS IF YOU WANT TO DELETE
reprocess_again = False    ## TRUE: REMOVES OLD RESULTS
save_mode = False
###
###
if __name__ == "__main__":
    main()
