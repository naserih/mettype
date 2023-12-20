# README

featurespace_metadata.json includes radiomic features extracted from 1273 spinal lesion centers (healthy or metastatic) from simulation-ct images using geometrical regions of interest (ROIs). 

Lesion centers for metastatic bone lesions extracted by 1 radiation oncologists and 4 radiation oncology residents.
Lesion centers for healthy bones extracted by 1 medical physicist and 2 medical physics graduate studets.

All lesion centers validated by one independet expert for quality control.

Regions of interest includes: sperical and cylendrical(along-z-axes) with the following charactristics:
Sperical ROIs (diameter in mm): 10, 15, 20, 30, 50, 70, 100
Cylendrical RPIs (diameter x height in mm): 10x10, 15x15, 20x20, 30x30, 50x50, 70x70, 100x100
                                            20x30, 30x20, 30x50, 50x30 


107 radiomic features extracted from original simulation ct images using pyradiomics version 3.0.1 that implemented on Python 2.7.4.

DATA STRUCTURE:

Data is collected into the JSON file with the following structure:

{
    *ID* : {
            'label' : *''*,
            'type'  : *''*,
            *ROI* : {
                *RADIOMIC FEATURE* : *FEATURE VALUE*
            },
    }
}  

*ID*    :  Is a a unique ID assigned to each lesion center, 
    - each ID  key is a 'string'. List of IDs are ['p0', 'p1', 'p2', ..., 'p1281']

'label' : 'label' is the key:value that used for isindicating whether or not the lesion center is metastatic or healthy lesion. 
    - 'label' value is a 'string' from the following list ['healthy', 'metastatic']
        - 'label':'healthy' means that the lesion center is a healthy bone lesion
        - 'label':'metastatic' means that the lesion center is a metastatic bone lesion

'type'  : is the key:value that used for isindicating the type of the metastatic lesion. 
    - 'type' value is a 'string' from the following list ['', 'blastic', 'mix', 'lytic']
        - for healthy bone lesions type is empty string ''
        - for metastatic bone lesions type is either 'blastic', 'lytic', or 'mix'
        - 'type':'' means that the lesion is not metastatic
        - 'type':'blastic' means that the lesion is a blastic bone metastases
        - 'type':'lytic' means that the lesion is a lytic bone metastases
        - 'type':'mix' means that the lesion is a mixed bone metastases
 
*ROI*   : *ROI* is a key for each resion of interest (ROI). each ROI key is a 'string' from the following list:
                ['SP10', 'SP15', 'SP20', 'SP30', 'SP50', 'SP70', 'SP100',
                 'CY10', 'CY15', 'CY20', 'CY30', 'CY50', 'CY70', 'CY100', 
                 'CY2030', 'CY3020', 'CY3050', 'CY5030']
        - CY stand for Cylendrical and SP stands for spherical, numbers are sizes of ROI.

            'SP10', 10 mm diamenter spherical ROI,
            'SP15', 15 mm diamenter spherical ROI,
            'SP20', 20 mm diamenter spherical ROI,
            'SP30', 30 mm diamenter spherical ROI,
            'SP50', 50 mm diamenter spherical ROI,
            'SP70', 70 mm diamenter spherical ROI,
            'SP100', 100 mm diamenter spherical ROI,
            'CY10', 10 mm diamenter 10 mm height spherical ROI along z axis,
            'CY15', 15 mm diamenter 15 mm height spherical ROI along z axis,
            'CY20', 20 mm diamenter 20 mm height spherical ROI along z axis,
            'CY30', 30 mm diamenter 30 mm height spherical ROI along z axis,
            'CY50', 50 mm diamenter 50 mm height spherical ROI along z axis,
            'CY70', 70 mm diamenter 70 mm height spherical ROI along z axis,
            'CY100', 100 mm diamenter 100 mm height spherical ROI along z axis,
            'CY2030', 20 mm diamenter 30 mm height spherical ROI along z axis,
            'CY3020', 30 mm diamenter 20 mm height spherical ROI along z axis,
            'CY3050', 30 mm diamenter 50 mm height spherical ROI along z axis,
            'CY5030', 50 mm diamenter 30 mm height spherical ROI along z axis,

        - Each ROI is a key for a dictionery that contains extracted radiomic features 

*RADIOMIC FEATURE* : is a 'key':value. 'key' is a name of a radiomic feature from the following list, 
                    and value is the value of the radiomic feature( a'flating' number) that calculated 
                    for that specifid ROI around the specifid lesion center.


['firstOrder_10Percentile',
'firstOrder_90Percentile',
'firstOrder_Energy',
'firstOrder_Entropy',
'firstOrder_InterquartileRange',
'firstOrder_Kurtosis',
'firstOrder_Maximum',
'firstOrder_Mean',
'firstOrder_MeanAbsoluteDeviation',
'firstOrder_Median',
'firstOrder_Minimum',
'firstOrder_Range',
'firstOrder_RobustMeanAbsoluteDeviation',
'firstOrder_RootMeanSquared',
'firstOrder_Skewness',
'firstOrder_TotalEnergy',
'firstOrder_Uniformity',
'firstOrder_Variance',
'GLCM_Autocorrelation',
'GLCM_ClusterProminence',
'GLCM_ClusterShade',
'GLCM_ClusterTendency',
'GLCM_Contrast',
'GLCM_Correlation',
'GLCM_DifferenceAverage',
'GLCM_DifferenceEntropy',
'GLCM_DifferenceVariance',
'GLCM_Id',
'GLCM_Idm',
'GLCM_Idmn',
'GLCM_Idn',
'GLCM_Imc1',
'GLCM_Imc2',
'GLCM_InverseVariance',
'GLCM_JointAverage',
'GLCM_JointEnergy',
'GLCM_JointEntropy',
'GLCM_MCC',
'GLCM_MaximumProbability',
'GLCM_SumAverage',
'GLCM_SumEntropy',
'GLCM_SumSquares',
'GLDM_DependenceEntropy',
'GLDM_DependenceNonUniformity',
'GLDM_DependenceNonUniformityNormalized',
'GLDM_DependenceVariance',
'GLDM_GrayLevelNonUniformity',
'GLDM_GrayLevelVariance',
'GLDM_HighGrayLevelEmphasis',
'GLDM_LargeDependenceEmphasis',
'GLDM_LargeDependenceHighGrayLevelEmphasis',
'GLDM_LargeDependenceLowGrayLevelEmphasis',
'GLDM_LowGrayLevelEmphasis',
'GLDM_SmallDependenceEmphasis',
'GLDM_SmallDependenceHighGrayLevelEmphasis',
'GLDM_SmallDependenceLowGrayLevelEmphasis',
'GLRLM_GrayLevelNonUniformity',
'GLRLM_GrayLevelNonUniformityNormalized',
'GLRLM_GrayLevelVariance',
'GLRLM_HighGrayLevelRunEmphasis',
'GLRLM_LongRunEmphasis',
'GLRLM_LongRunHighGrayLevelEmphasis',
'GLRLM_LongRunLowGrayLevelEmphasis',
'GLRLM_LowGrayLevelRunEmphasis',
'GLRLM_RunEntropy',
'GLRLM_RunLengthNonUniformity',
'GLRLM_RunLengthNonUniformityNormalized',
'GLRLM_RunPercentage',
'GLRLM_RunVariance',
'GLRLM_ShortRunEmphasis',
'GLRLM_ShortRunHighGrayLevelEmphasis',
'GLRLM_ShortRunLowGrayLevelEmphasis',
'GLSZM_GrayLevelNonUniformity',
'GLSZM_GrayLevelNonUniformityNormalized',
'GLSZM_GrayLevelVariance',
'GLSZM_HighGrayLevelZoneEmphasis',
'GLSZM_LargeAreaEmphasis',
'GLSZM_LargeAreaHighGrayLevelEmphasis',
'GLSZM_LargeAreaLowGrayLevelEmphasis',
'GLSZM_LowGrayLevelZoneEmphasis',
'GLSZM_SizeZoneNonUniformity',
'GLSZM_SizeZoneNonUniformityNormalized',
'GLSZM_SmallAreaEmphasis',
'GLSZM_SmallAreaHighGrayLevelEmphasis',
'GLSZM_SmallAreaLowGrayLevelEmphasis',
'GLSZM_ZoneEntropy',
'GLSZM_ZonePercentage',
'GLSZM_ZoneVariance',
'NGTDM_Busyness',
'NGTDM_Coarseness',
'NGTDM_Complexity',
'NGTDM_Contrast',
'NGTDM_Strength',
'Shape_Elongation',
'Shape_Flatness',
'Shape_LeastAxisLength',
'Shape_MajorAxisLength',
'Shape_Maximum2DDiameterColumn',
'Shape_Maximum2DDiameterRow',
'Shape_Maximum2DDiameterSlice',
'Shape_Maximum3DDiameter',
'Shape_MeshVolume',
'Shape_MinorAxisLength',
'Shape_Sphericity',
'Shape_SurfaceArea',
'Shape_SurfaceVolumeRatio',
'Shape_VoxelVolume',
]
