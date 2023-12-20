import os
import csv
import json
from dotenv import load_dotenv
load_dotenv()


feature_root = os.environ.get("RADIOMICS_FEATURES_PATH")
label_file = os.environ.get("LESION_CENTERS_WITH_LABEL")
output_results = os.environ.get("PAINDICATOR_RESULTS")

def get_feature_space(label_metadata, deidentifier, feature_root):
    feature_metadata = label_metadata
    rois = os.listdir(feature_root)
    cnt = 0
    for roi in rois:
        csv_files = None
        feature_path = None
        feature_path = os.path.join(feature_root, roi)
        csv_files = [f for f in os.listdir(feature_path) if 'FAILED' not in f]
        # if 'FAILED' not in csv_file_name
        # print csv_files
        for csv_file_name in csv_files:
            csv_path = None
            ct_name = None
            csv_path = os.path.join(feature_root,roi,csv_file_name)
            ct_name = '_'.join(csv_file_name.split('_')[:-1])
            
            cnt += 1 
            if ct_name in deidentifier:
                label_index = deidentifier[ct_name]
                # print cnt, roi, ct_name
                feature_metadata[label_index][roi] = {}

                csv_data = open(csv_path, 'r')
                csvreader = csv.reader(csv_data)
                # print filepath
                header = csvreader.next()
                    # print header
                for row in csvreader:
                    # pass
                    # print row[0]
                    feature_metadata[label_index][roi][row[0]] = float(row[1])
                csv_data.close()
                csvreader = None
                        

                
    return feature_metadata

def get_label_metadata(label_file, label_column):
    label_metadata = {}
    deidentifier = {}
    with open(label_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        # print header
        cnt = 0
        for row in csvreader:
            file_id = "_".join([row[12]]+row[13][1:-1].split(', '))
            if file_id not in deidentifier:
                deidentifier[file_id] = 'p'+str(cnt)
                cnt += 1
            label = row[label_column]
            if deidentifier[file_id] not in label_metadata:
                label_metadata[deidentifier[file_id]] = {}
                label_metadata[deidentifier[file_id]]['type'] = row[15]
                if 'MET' in label:
                    label_metadata[deidentifier[file_id]]['label'] = 'metastatic'
                elif 'CTRL' in label:
                    label_metadata[deidentifier[file_id]]['label'] = 'healthy'

                else:
                    print 'Error: label unk'
            else:
                # doublicates are due to having multiple notes!
                # print 'Error: doublicate' 
                pass

    # print label_metadata
    return label_metadata, deidentifier

label_column = 3 # 3 cntrl/met # 13 met type #  VDP score 8 # 
label_metadata, deidentifier = get_label_metadata(label_file,label_column)    
feature_metadata = get_feature_space(label_metadata, deidentifier, feature_root)

valid_recs = []
inval_recs = []

first_key = feature_metadata.keys()[0]

print sorted(feature_metadata.keys())
print feature_metadata[first_key].keys()
print len (feature_metadata[first_key].keys())
first_att = feature_metadata[first_key].keys()[0]

for key in sorted(feature_metadata[first_key][first_att].keys()):
    print ">'",key,"',<"
print len(feature_metadata[first_key][first_att].keys())
types = []
for key in feature_metadata:
    if feature_metadata[key]['type'] not in types:
        types.append(feature_metadata[key]['type'])
print types

print '---------'
print first_key
print feature_metadata[first_key]

    # if len(feature_metadata[key].keys()) > 2:
#         print feature_metadata[key][feature_metadata[key].keys()[0]].keys()
#         valid_recs.append(len(feature_metadata[key].keys()))
#     else:
#         inval_recs.append(feature_metadata[key])
# print len(valid_recs)
# print inval_recs
   
# print feature_metadata
# with open(output_results+'featurespace_metadata.json','w') as jsonfile:
    # json.dump(feature_metadata, jsonfile)
