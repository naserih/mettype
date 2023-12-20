import os, copy, pydicom
import numpy as np
from pydicom.tag import Tag
import SimpleITK as sitk
from skimage import draw
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label,regionprops,find_contours


class Dicom_to_Imagestack:
    def __init__(self, rewrite_RT_file=False, delete_previous_rois=True,Contour_Names=None,
                 template_dir=None, channels=3, get_images_mask=True, arg_max=True,
                 associations={}, **kwargs):
        if Contour_Names is not None:
            for name in Contour_Names:
                if name not in associations:
                    associations[name] = name
        else:
            Contour_Names = []
        self.arg_max = arg_max
        self.rewrite_RT_file = rewrite_RT_file
        if template_dir is None:
            template_dir = os.path.join('./', 'template_RS.dcm')
        self.template_dir = template_dir
        self.template = True
        self.delete_previous_rois = delete_previous_rois
        self.Contour_Names = Contour_Names
        self.channels = channels
        self.get_images_mask = get_images_mask
        keys = list(associations.keys())
        for key in keys:
            associations[key.lower()] = associations[key].lower()
        self.associations, self.hierarchy = associations, {}
        self.get_images_mask = get_images_mask
        self.reader = sitk.ImageSeriesReader()
        self.reader.MetaDataDictionaryArrayUpdateOn()
        self.reader.LoadPrivateTagsOn()
        self.all_RTs = {}
        self.all_RTp = {}
        self.all_s_rois = []
        self.all_p_rois = []
        self.all_paths = []

    def down_folder(self, input_path):
        files = []
        dirs = []
        file = []
        for root, dirs, files in os.walk(input_path):
            break
        for val in files:
            if val.find('.dcm') != -1:
                file = val
                break
        if file and input_path:
            self.all_paths.append(input_path)
            self.Make_Contour_From_directory(input_path)
        for dir in dirs:
            new_directory = os.path.join(input_path, dir)
            self.down_folder(new_directory)
        return None

    def make_array(self, PathDicom):
        self.PathDicom = PathDicom
        self.lstFilesDCM = []
        self.lstRSFile = None
        self.lstRPFile = None
        self.Dicom_info = []
        fileList = []
        self.RTs_in_case = {}
        self.RTp_in_case = {}
        for dirName, dirs, fileList in os.walk(PathDicom):
            break
        fileList = [i for i in fileList if i.find('.dcm') != -1]
        RTS_fileList = [os.path.join(dirName, i) for i in fileList if i.find('RT') == 0 or i.find('RS') == 0]
        RTP_fileList = [os.path.join(dirName, i) for i in fileList if i.find('RP') == 0]
        
        if not self.get_images_mask:
            # print '--------** ', RT_fileList
            # if RT_fileList:
            #     fileList = RT_fileList
            for filename in fileList:
                try:
                    ds = pydicom.read_file(os.path.join(dirName, filename))
                    self.ds = ds
                    if ds.Modality == 'CT' or ds.Modality == 'MR' or ds.Modality == 'PT':  # check whether the file's DICOM
                        self.lstFilesDCM.append(os.path.join(dirName, filename))
                        self.Dicom_info.append(ds)
                        self.ds = ds
                    elif ds.Modality == 'RTSTRUCT':
                        self.lstRSFile = os.path.join(dirName, filename)
                        self.RTs_in_case[self.lstRSFile] = []
                    elif ds.Modality == 'RTPLAN':
                        self.lstRPFile = os.path.join(dirName, filename)
                        self.RTp_in_case[self.lstRPFile] = []
                except:
                    continue
            if self.lstFilesDCM:
                self.RefDs = pydicom.read_file(self.lstFilesDCM[0])
        else:
            self.dicom_names = self.reader.GetGDCMSeriesFileNames(self.PathDicom)
            self.reader.SetFileNames(self.dicom_names)
            self.get_images()
            # image_files = [i.split(PathDicom)[1][1:] for i in self.dicom_names]
            # RT_Files = [os.path.join(PathDicom, file) for file in fileList if file not in image_files]
            # print 'III>>>> ', RT_Files
            # print '>>>>>>> ', RTS_fileList
            for self.lstRSFile in RTS_fileList:
                self.RTs_in_case[self.lstRSFile] = []
            for self.lstRPFile in RTP_fileList:
                self.RTp_in_case[self.lstRPFile] = []
            self.RefDs = pydicom.read_file(self.dicom_names[0])
            self.ds = pydicom.read_file(self.dicom_names[0])


        self.mask_exist = False
        self.s_rois_in_case = []
        self.p_rois_in_case = []
        self.all_RTs.update(self.RTs_in_case)
        self.all_RTp.update(self.RTp_in_case)
        if self.lstRSFile is not None:
            self.template = False
            for RTS in self.RTs_in_case:
                self.lstRSFile = RTS
                self.get_rois_from_RTS()
        elif self.get_images_mask:
            self.use_template()

        if self.lstRPFile is not None:
            self.template = False
            for RTP in self.RTp_in_case:
                self.lstRPFile = RTP
                self.get_rois_from_RTP()
        elif self.get_images_mask:
            self.use_template()

    def get_rois_from_RTS(self):
        rois_in_structure = []
        self.RS_struct = pydicom.read_file(self.lstRSFile)
        # print '--------------------------------------------------'
        # print self.RS_struct.keys()
        if Tag((0x3006, 0x020)) in self.RS_struct.keys():
            self.ROI_Structure = self.RS_struct.StructureSetROISequence
        else:
            self.ROI_Structure = []
        for Structures in self.ROI_Structure:
            if Structures.ROIName not in self.s_rois_in_case:
                self.s_rois_in_case.append(Structures.ROIName)
                rois_in_structure.append(Structures.ROIName)
        self.all_RTs[self.lstRSFile] = rois_in_structure
        # print 'UUUUUUUU    ', rois_in_structure

    def get_rois_from_RTP(self):
        rois_in_structure = []
        self.RP_struct = pydicom.read_file(self.lstRPFile)
        print 'v-------v-------v-------v-------v--------v--------v'
        # print self.RP_struct.keys()
        # for key in self.RP_struct.keys():
        #     print key
        print self.RP_struct.dir()
        self.dose_ref = self.RP_struct.DoseReferenceSequence
        print self.dose_ref[0]

        print '==================================='
        if Tag((0x300a, 0x0b0)) in self.RP_struct.keys():
            print self.RP_struct.keys()
            print 'MWMWMW ', self.RP_struct[Tag((0x300a, 0x010))]
            # if Tag((0x300a, 0x018)) in self.RP_struct[Tag((0x300a, 0x010))].keys():
            print '===========> ', self.RP_struct.DoseReferenceSequence
            self.ROI_Plan = self.RP_struct.DoseReferenceSequence
            # print '===========> ', self.ROI_Plan #, self.ROI_Plan.keys()
        else:
            self.ROI_Plan = []
        for Structures in self.ROI_Plan:
            print Structures
            print ',.,.,.,.,.,.,.,.,.,.,.,.,.'
            if Structures.ROIName not in self.p_rois_in_case:
                self.p_rois_in_case.append(Structures.ROIName)
                rois_in_structure.append(Structures.ROIName)
        self.all_RTp[self.lstRPFile] = rois_in_structure

    def get_mask(self):
        self.mask = np.zeros([len(self.dicom_names), self.image_size_1, self.image_size_2, len(self.Contour_Names) + 1],
                             dtype='int8')
        # print "YYYYYYY", len(self.dicom_names), self.image_size_1, self.image_size_2
        self.structure_references = {}
        # print '&&&&&&  ', self.RS_struct.ROIContourSequence
        for contour_number in range(len(self.RS_struct.ROIContourSequence)):
            self.structure_references[
                self.RS_struct.ROIContourSequence[contour_number].ReferencedROINumber] = contour_number
            # print contour_number
        found_rois = {}
        for Structures in self.ROI_Structure:
            ROI_Name = Structures.ROIName
            if Structures.ROINumber not in self.structure_references.keys():
                continue
            true_name = None
            if ROI_Name in self.associations:
                true_name = self.associations[ROI_Name]
            elif ROI_Name.lower() in self.associations:
                true_name = self.associations[ROI_Name.lower()]
            if true_name and true_name in self.Contour_Names:
                found_rois[true_name] = {'Hierarchy': 999, 'Name': ROI_Name, 'Roi_Number': Structures.ROINumber}
        for ROI_Name in found_rois.keys():
            if found_rois[ROI_Name]['Roi_Number'] in self.structure_references:
                index = self.structure_references[found_rois[ROI_Name]['Roi_Number']]
                mask = self.get_mask_for_contour(index)
                self.mask[..., self.Contour_Names.index(ROI_Name) + 1][mask == 1] = 1
        if self.arg_max:
            self.mask = np.argmax(self.mask, axis=-1)
        self.annotation_handle = sitk.GetImageFromArray(self.mask.astype('int8'))
        self.annotation_handle.SetSpacing(self.dicom_handle.GetSpacing())
        self.annotation_handle.SetOrigin(self.dicom_handle.GetOrigin())
        self.annotation_handle.SetDirection(self.dicom_handle.GetDirection())
        return None

    def get_mask_for_contour(self, i):
        self.Liver_Locations = self.RS_struct.ROIContourSequence[i].ContourSequence
        # print '12345     )))', i, self.Liver_Locations
        self.Liver_Slices = []
        for contours in self.Liver_Locations:
            # print contours
            # print '- - - - - - - -'
            data_point = contours.ContourData[2]
            if data_point not in self.Liver_Slices:
                self.Liver_Slices.append(contours.ContourData[2])
        print self.Liver_Slices
        return self.Contours_to_mask()

    def Contours_to_mask(self):
        mask = np.zeros([len(self.dicom_names), self.image_size_1, self.image_size_2], dtype='int8')
        Contour_data = self.Liver_Locations
        ShiftCols, ShiftRows, _ = [float(i) for i in self.reader.GetMetaData(0, "0020|0032").split('\\')]
        PixelSize = self.dicom_handle.GetSpacing()[0]
        Mag = 1 / PixelSize
        mult1 = mult2 = 1
        if ShiftCols > 0:
            mult1 = -1

        for i in range(len(Contour_data)):
            referenced_sop_instance_uid = Contour_data[i].ContourImageSequence[0].ReferencedSOPInstanceUID
            if referenced_sop_instance_uid not in self.SOPInstanceUIDs:
                print('Error here with instance UID')
                return None
            else:
                slice_index = self.SOPInstanceUIDs.index(referenced_sop_instance_uid)
            cols = Contour_data[i].ContourData[1::3]
            rows = Contour_data[i].ContourData[0::3]
            col_val = [Mag * abs(x - mult1 * ShiftRows) for x in cols]
            row_val = [Mag * abs(x - mult2 * ShiftCols) for x in rows]
            temp_mask = self.poly2mask(col_val, row_val, [self.image_size_1, self.image_size_2])
            mask[slice_index, :, :][temp_mask > 0] += 1
        mask[mask>1] = 0
        return mask

    def use_template(self):
        self.template = True
        if not self.template_dir:
            self.template_dir = os.path.join('\\\\mymdafiles', 'ro-admin', 'SHARED', 'Radiation physics', 'BMAnderson',
                                             'Auto_Contour_Sites', 'template_RS.dcm')
            if not os.path.exists(self.template_dir):
                self.template_dir = os.path.join('..', '..', 'Shared_Drive', 'Auto_Contour_Sites', 'template_RS.dcm')
        self.key_list = self.template_dir.replace('template_RS.dcm', 'key_list.txt')
        self.RS_struct = pydicom.read_file(self.template_dir)
        print('Running off a template')
        self.changetemplate()

    def get_images(self):
        self.dicom_handle = self.reader.Execute()
        sop_instance_UID_key = "0008|0018"
        self.SOPInstanceUIDs = [self.reader.GetMetaData(i, sop_instance_UID_key) for i in
                                range(self.dicom_handle.GetDepth())]
        slice_location_key = "0020|0032"
        self.slice_info = [self.reader.GetMetaData(i, slice_location_key).split('\\')[-1] for i in
                           range(self.dicom_handle.GetDepth())]
        self.ArrayDicom = sitk.GetArrayFromImage(self.dicom_handle)
        self.image_size_1, self.image_size_2, _ = self.dicom_handle.GetSize()

    def poly2mask(self, vertex_row_coords, vertex_col_coords, shape):
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
        mask = np.zeros(shape, dtype=np.bool)
        mask[fill_row_coords, fill_col_coords] = True
        return mask

    def with_annotations(self, annotations, output_dir, ROI_Names=None):
        annotations = np.squeeze(annotations)
        self.image_size_0, self.image_size_1 = annotations.shape[1], annotations.shape[2]
        self.ROI_Names = ROI_Names
        self.output_dir = output_dir
        if len(annotations.shape) == 3:
            annotations = np.expand_dims(annotations, axis=-1)
        self.annotations = annotations
        self.Mask_to_Contours()

    def Mask_to_Contours(self):
        self.RefDs = self.ds
        self.ShiftCols, self.ShiftRows, _ = [float(i) for i in self.reader.GetMetaData(0, "0020|0032").split('\\')]
        self.mult1 = self.mult2 = 1
        self.PixelSize = self.dicom_handle.GetSpacing()[0]
        current_names = []
        for names in self.RS_struct.StructureSetROISequence:
            current_names.append(names.ROIName)
        Contour_Key = {}
        xxx = 1
        for name in self.ROI_Names:
            Contour_Key[name] = xxx
            xxx += 1
        self.all_annotations = self.annotations
        base_annotations = copy.deepcopy(self.annotations)
        temp_color_list = []
        color_list = [[128, 0, 0], [170, 110, 40], [0, 128, 128], [0, 0, 128], [230, 25, 75], [225, 225, 25],
                      [0, 130, 200], [145, 30, 180],
                      [255, 255, 255]]
        self.struct_index = 0
        new_ROINumber = 1000
        for Name in self.ROI_Names:
            new_ROINumber -= 1
            if not temp_color_list:
                temp_color_list = copy.deepcopy(color_list)
            color_int = np.random.randint(len(temp_color_list))
            print('Writing data for ' + Name)
            self.annotations = copy.deepcopy(base_annotations[:, :, :, int(self.ROI_Names.index(Name) + 1)])
            self.annotations = self.annotations.astype('int')

            make_new = 1
            allow_slip_in = True
            if (Name not in current_names and allow_slip_in) or self.delete_previous_rois:
                self.RS_struct.StructureSetROISequence.insert(0,copy.deepcopy(self.RS_struct.StructureSetROISequence[0]))
                # if not self.template:
                #     self.struct_index = len(self.RS_struct.StructureSetROISequence) - 1
                # else:
                #     self.struct_index += 1
            else:
                print('Prediction ROI {} is already within RT structure'.format(Name))
                continue
            #     self.struct_index = current_names.index(Name) - 1
            self.RS_struct.StructureSetROISequence[self.struct_index].ROINumber = new_ROINumber
            self.RS_struct.StructureSetROISequence[self.struct_index].ReferencedFrameOfReferenceUID = \
                self.ds.FrameOfReferenceUID
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIName = Name
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIVolume = 0
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIGenerationAlgorithm = 'SEMIAUTOMATIC'
            if make_new == 1:
                self.RS_struct.RTROIObservationsSequence.insert(0,
                    copy.deepcopy(self.RS_struct.RTROIObservationsSequence[0]))
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ObservationNumber = new_ROINumber
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ReferencedROINumber = new_ROINumber
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ROIObservationLabel = Name
            self.RS_struct.RTROIObservationsSequence[self.struct_index].RTROIInterpretedType = 'ORGAN'

            if make_new == 1:
                self.RS_struct.ROIContourSequence.insert(0,copy.deepcopy(self.RS_struct.ROIContourSequence[0]))
            self.RS_struct.ROIContourSequence[self.struct_index].ReferencedROINumber = new_ROINumber
            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[1:] = []
            self.RS_struct.ROIContourSequence[self.struct_index].ROIDisplayColor = temp_color_list[color_int]
            del temp_color_list[color_int]

            contour_num = 0
            if np.max(self.annotations) > 0:  # If we have an annotation, write it
                image_locations = np.max(self.annotations, axis=(1, 2))
                indexes = np.where(image_locations > 0)[0]
                for point, i in enumerate(indexes):
                    print(str(int(point / len(indexes) * 100)) + '% done with ' + Name)
                    annotation = self.annotations[i, :, :]
                    regions = regionprops(label(annotation))
                    for ii in range(len(regions)):
                        temp_image = np.zeros([self.image_size_0, self.image_size_1])
                        data = regions[ii].coords
                        rows = []
                        cols = []
                        for iii in range(len(data)):
                            rows.append(data[iii][0])
                            cols.append(data[iii][1])
                        temp_image[rows, cols] = 1
                        points = find_contours(temp_image, 0)[0]
                        output = []
                        for point in points:
                            output.append(((point[1]) * self.PixelSize + self.mult1 * self.ShiftCols))
                            output.append(((point[0]) * self.PixelSize + self.mult2 * self.ShiftRows))
                            output.append(float(self.slice_info[i]))
                        if output:
                            if contour_num > 0:
                                self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence.append(
                                    copy.deepcopy(
                                        self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[0]))
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                                contour_num].ContourNumber = str(contour_num)
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                                contour_num].ContourImageSequence[0].ReferencedSOPInstanceUID = self.SOPInstanceUIDs[i]
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                                contour_num].ContourData = output
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                                contour_num].NumberofContourPoints = round(len(output) / 3)
                            contour_num += 1
                    hole_annotation = 1 - annotation
                    filled_annotation = binary_fill_holes(annotation)
                    hole_annotation[filled_annotation == 0] = 0
                    regions = regionprops(label(hole_annotation))
                    for ii in range(len(regions)):
                        temp_image = np.zeros([self.image_size_0, self.image_size_1])
                        data = regions[ii].coords
                        rows = []
                        cols = []
                        for iii in range(len(data)):
                            rows.append(data[iii][0])
                            cols.append(data[iii][1])
                        temp_image[rows, cols] = 1
                        points = find_contours(temp_image, 0)[0]
                        output = []
                        for point in points:
                            output.append(((point[1]) * self.PixelSize + self.mult1 * self.ShiftCols))
                            output.append(((point[0]) * self.PixelSize + self.mult2 * self.ShiftRows))
                            output.append(float(self.slice_info[i]))
                        if output:
                            if contour_num > 0:
                                self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence.append(
                                    copy.deepcopy(
                                        self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[0]))
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                                contour_num].ContourNumber = str(contour_num)
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                                contour_num].ContourImageSequence[0].ReferencedSOPInstanceUID = self.SOPInstanceUIDs[i]
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                                contour_num].ContourData = output
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                                contour_num].NumberofContourPoints = round(len(output) / 3)
                            contour_num += 1
        self.RS_struct.SOPInstanceUID += '.' + str(np.random.randint(999))
        # for i in range(len(self.RS_struct.StructureSetROISequence)):
        #     self.RS_struct.StructureSetROISequence[i].ROINumber = i + 1
        #     self.RS_struct.RTROIObservationsSequence[i].ReferencedROINumber = i + 1
        #     self.RS_struct.ROIContourSequence[i].ReferencedROINumber = i + 1
        if self.template or self.delete_previous_rois:
            for i in range(len(self.RS_struct.StructureSetROISequence),len(self.ROI_Names),-1):
                del self.RS_struct.StructureSetROISequence[-1]
            for i in range(len(self.RS_struct.RTROIObservationsSequence),len(self.ROI_Names),-1):
                del self.RS_struct.RTROIObservationsSequence[-1]
            for i in range(len(self.RS_struct.ROIContourSequence),len(self.ROI_Names),-1):
                del self.RS_struct.ROIContourSequence[-1]
            for i in range(len(self.RS_struct.StructureSetROISequence)):
                self.RS_struct.StructureSetROISequence[i].ROINumber = i + 1
                self.RS_struct.RTROIObservationsSequence[i].ReferencedROINumber = i + 1
                self.RS_struct.ROIContourSequence[i].ReferencedROINumber = i + 1
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        out_name = os.path.join(self.output_dir,
                                'RS_MRN' + self.RS_struct.PatientID + '_' + self.RS_struct.SeriesInstanceUID + '.dcm')
        if os.path.exists(out_name):
            out_name = os.path.join(self.output_dir,
                                    'RS_MRN' + self.RS_struct.PatientID + '_' + self.RS_struct.SeriesInstanceUID + '1.dcm')
        print('Writing out data...')
        pydicom.write_file(out_name, self.RS_struct)
        fid = open(os.path.join(self.output_dir, 'Completed.txt'), 'w+')
        fid.close()
        print('Finished!')
        # Raystation_dir = self.output_dir.split('Output_MRN')[0]+'Output_MRN_RayStation\\'+self.RS_struct.PatientID+'\\'
        # if not os.path.exists(Raystation_dir):
        # dicom.write_file(Raystation_dir + 'RS_MRN' + self.RS_struct.PatientID + '_' + self.ds.SeriesInstanceUID + '.dcm', self.RS_struct)
        # fid = open(Raystation_dir+'Completed.txt','w+')
        # fid.close()
        return None

    def changetemplate(self):
        keys = self.RS_struct.keys()
        for key in keys:
            # print(self.RS_struct[key].name)
            if self.RS_struct[key].name == 'Referenced Frame of Reference Sequence':
                break
        self.RS_struct[key]._value[0].FrameOfReferenceUID = self.ds.FrameOfReferenceUID
        self.RS_struct[key]._value[0].RTReferencedStudySequence[0].ReferencedSOPInstanceUID = self.ds.StudyInstanceUID
        self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
            0].SeriesInstanceUID = self.ds.SeriesInstanceUID
        for i in range(len(self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                               0].ContourImageSequence) - 1):
            del self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence[-1]
        fill_segment = copy.deepcopy(
            self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence[0])
        for i in range(len(self.SOPInstanceUIDs)):
            temp_segment = copy.deepcopy(fill_segment)
            temp_segment.ReferencedSOPInstanceUID = self.SOPInstanceUIDs[i]
            self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence.append(temp_segment)
        del \
        self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence[0]

        new_keys = open(self.key_list)
        keys = {}
        i = 0
        for line in new_keys:
            keys[i] = line.strip('\n').split(',')
            i += 1
        new_keys.close()
        for index in keys.keys():
            new_key = keys[index]
            try:
                self.RS_struct[new_key[0], new_key[1]] = self.ds[[new_key[0], new_key[1]]]
            except:
                continue
        return None
        # Get slice locations

    def Make_Contour_From_directory(self, PathDicom):
        self.make_array(PathDicom)
        if self.rewrite_RT_file:
            self.rewrite_RT()
        if not self.template and self.get_images_mask:
            self.get_mask()
        true_rois = []
        # print '|||||||||>   ', self.s_rois_in_case
        # print '|||||||||>   ', self.p_rois_in_case
        for roi in self.s_rois_in_case:
            if roi not in self.all_s_rois:
                self.all_s_rois.append(roi)
            if self.Contour_Names:
                if roi in self.associations:
                    true_rois.append(self.associations[roi])
                elif roi in self.Contour_Names:
                    true_rois.append(roi)
        for roi in self.Contour_Names:
            if roi not in true_rois:
                print('Lacking {} in {}'.format(roi, PathDicom))
                print('Found {}'.format(self.s_rois_in_case))
                break
        return None

    def rewrite_RT(self, lstRSFile=None):
        if lstRSFile is not None:
            self.RS_struct = pydicom.read_file(lstRSFile)
        if Tag((0x3006, 0x020)) in self.RS_struct.keys():
            self.ROI_Structure = self.RS_struct.StructureSetROISequence
        else:
            self.ROI_Structure = []
        if Tag((0x3006, 0x080)) in self.RS_struct.keys():
            self.Observation_Sequence = self.RS_struct.RTROIObservationsSequence
        else:
            self.Observation_Sequence = []
        self.s_rois_in_case = []
        for i, Structures in enumerate(self.ROI_Structure):
            if Structures.ROIName in self.associations:
                new_name = self.associations[Structures.ROIName]
                self.RS_struct.StructureSetROISequence[i].ROIName = new_name
            self.s_rois_in_case.append(self.RS_struct.StructureSetROISequence[i].ROIName)
        for i, ObsSequence in enumerate(self.Observation_Sequence):
            if ObsSequence.ROIObservationLabel in self.associations:
                new_name = self.associations[ObsSequence.ROIObservationLabel]
                self.RS_struct.RTROIObservationsSequence[i].ROIObservationLabel = new_name
        self.RS_struct.save_as(self.lstRSFile)

if __name__ == '__main__':
    xxx = 1
