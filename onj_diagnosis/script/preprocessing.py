import os
import os.path as osp
from glob import glob
import SimpleITK as sitk


# Configuration
DATA_ROOT = r'G:\txt+dcm'

OM2LABEL = {
    'chemical': 1,
    'drug': 2,
    'radiation': 3,
    'other': 4
}


def collect_classification_dataset(data_root, modality='CBCT'):
    dataset = []

    name = 'dcm' if modality != 'PR' else 'dcm.dcm'

    for om in OM2LABEL.keys():
        paths = glob(osp.join(data_root, 'gsy', om, '*', modality, name))
        for p in paths:
            dataset.append({'path': p, 'label': OM2LABEL[om]})

    paths = glob(osp.join(data_root, 'qt', '*', '*', modality, name))
    for p in paths:
        dataset.append({'path': p, 'label': 0})

    return dataset
    

def collect_segmentation_dataset(data_root, modality='CBCT'):
    dataset = []

    name = 'dcm' if modality != 'PR' else 'dcm.dcm'
    label_name = 'label' if modality != 'PR' else 'label.dcm'

    for om in OM2LABEL.keys():
        paths = glob(osp.join(data_root, 'gsy', om, '*', modality, name))
        for p in paths:
            label_p = osp.join(osp.dirname(p), label_name)
            # print(label_p)
            if not osp.exists(label_p): continue
            dataset.append({'path': p, 'label': label_p})

    return dataset


def collect_visual_language_dataset(data_root, visual_modality='CBCT'):
    dataset = []

    
    name = 'dcm' if visual_modality != 'PR' else 'dcm.dcm'

    for om in OM2LABEL.keys():
        visual_paths = glob(osp.join(data_root, 'gsy', om, '*', visual_modality, name))
        for p in visual_paths:
            # language_p = glob(osp.join(osp.dirname(osp.dirname(p)), '*.docx'))[0]
            language_p = glob(osp.join(osp.dirname(osp.dirname(p)), '*.txt'))[0]
            if not osp.exists(language_p): continue
            dataset.append({'visual_path': p, 'language_path': language_p, 'label': OM2LABEL[om]})

    return dataset


if __name__=='__main__':

    # Classification Dataset
    # CBCT_CLS_DATASET        = collect_classification_dataset(DATA_ROOT, 'CBCT')
    # CT_CLS_DATASET          = collect_classification_dataset(DATA_ROOT, 'CT')
    # PR_CLS_DATASET          = collect_classification_dataset(DATA_ROOT, 'PR')
    # # print(CBCT_CLS_DATASET)

    # Segmentation Dataset
    # CBCT_SEG_DATASET        = collect_segmentation_dataset(DATA_ROOT, 'CBCT')
    # CT_SEG_DATASET          = collect_segmentation_dataset(DATA_ROOT, 'CT')
    PR_SEG_DATASET          = collect_segmentation_dataset(DATA_ROOT, 'PR')
    # print(CT_SEG_DATASET)

    # Multi-Modality Classification Dataset
    # CBCT_TEXT_CLS_DATASET   = collect_visual_language_dataset(DATA_ROOT, 'CBCT')
    # CT_TEXT_CLS_DATASET     = collect_visual_language_dataset(DATA_ROOT, 'CT')
    # PR_TEXT_CLS_DATASET     = collect_visual_language_dataset(DATA_ROOT, 'PR')
    # # print(CBCT_TEXT_CLS_DATASET)

    # for item in CBCT_CLS_DATASET:
    #     path = item['path']
    #     reader = sitk.ImageSeriesReader()
    #     reader.SetFileNames(reader.GetGDCMSeriesFileNames(path))
    #     data = sitk.GetArrayFromImage(reader.Execute())


    #     # data = sitk.GetArrayFromImage(sitk.ReadImage(path))
    #     print(type(data), data.shape, path)

    print('DONE')