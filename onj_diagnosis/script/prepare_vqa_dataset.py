 
import os
import os.path as osp
import random
from glob import glob
from tqdm import tqdm

from docx import Document
import shutil


def parse_docx(path):
    # load text
    flag = True
    medical_record = ''
    treatment_plan = ''
    for p in Document(path).paragraphs:
        if p.text[:2] == '输出':
            flag = False
            continue
        
        if flag:
            medical_record += p.text + '\r\n'
        else:
            treatment_plan += p.text + '\r\n'

    medical_record = medical_record.strip()
    treatment_plan = treatment_plan.strip()

    # parse text
    qa_list = []
    # 诊断
    for item in treatment_plan.split('\r\n'):
        head, body = item.split('：')
        if head == '诊断':
            qa_list.append({'question': '临床诊断是什么', 'answer': body.strip()})
            break

    # 治疗方案
    flag = False
    answer = ''
    for item in treatment_plan.split('\r\n'):
        if item[:4] == '治疗方案':
            flag = True
            continue
        if flag:
            answer += item + '\r\n'
    qa_list.append({'question': '如何设计治疗方案', 'answer': answer.strip()})

    # 口腔护理
    for item in treatment_plan.split('\r\n'):
        head, body = item.split('：')
        if head == '口腔护理':
            qa_list.append({'question': '口腔科门诊需要进行哪些保守操作', 'answer': body.strip()[:-1]})
            break

    # 药物治疗
    for item in treatment_plan.split('\r\n'):
        head, body = item.split('：')
        if head == '药物治疗':
            qa_list.append({'question': '如何进行药物治疗', 'answer': body.strip()[:-1]})
            break

    # 手术治疗
    for item in treatment_plan.split('\r\n'):
        head, body = item.split('：')
        if head == '手术治疗':
            qa_list.append({'question': '如何进行手术治疗', 'answer': body.strip()[:-1]})
            break

    # 其他
    for item in treatment_plan.split('\r\n'):
        head, body = item.split('：')
        if head == '其他':
            qa_list.append({'question': '注意事项有哪些', 'answer': body.strip()[:-1]})
            break

    return medical_record, qa_list


if __name__=='__main__':

    src_data_root = r'G:\txt+dcm'
    dst_data_root = r'vqa_dataset'
    train_ratio = 0.8

    patient_paths = []
    for om in ['chemical', 'drug', 'radiation', 'other']:
        chinese_language_paths = glob(osp.join(src_data_root, 'gsy', om, '*', '*.docx'))
        for cns_p in tqdm(chinese_language_paths):
            patient_path = osp.dirname(cns_p)
            if osp.isdir(osp.join(patient_path, 'PR')):
                patient_paths.append(patient_path)

    random.seed(2024)
    random.shuffle(patient_paths)
    n_paths = len(patient_paths)
    train_paths, test_paths = \
        patient_paths[:int(n_paths*train_ratio)], \
        patient_paths[int(n_paths*train_ratio):]

    dst_train_root = osp.join(dst_data_root, 'train')
    i_train = 0
    for p in tqdm(train_paths):
        dcm_p = osp.join(p, 'PR', 'dcm.dcm')
        label_p = osp.join(p, 'PR', 'label.dcm')
        text_p = glob(osp.join(p, '*.docx'))[0]

        assert osp.exists(dcm_p), dcm_p
        assert osp.exists(label_p), label_p
        assert osp.exists(text_p), text_p

        medical_record, qa_list = parse_docx(text_p)

        for qa_item in qa_list:
            dst_patient_path = osp.join(dst_train_root, f'{i_train:03d}')
            if not osp.exists(dst_patient_path): os.makedirs(dst_patient_path)

            shutil.copy2(dcm_p, osp.join(dst_patient_path, 'panoramic_radiograph.dcm'))
            shutil.copy2(label_p, osp.join(dst_patient_path, 'segmentation.dcm'))

            with open(osp.join(dst_patient_path, 'medical_record.txt'), 'w') as f:
                f.write(medical_record)
            with open(osp.join(dst_patient_path, 'question.txt'), 'w') as f:
                f.write(qa_item['question'])
            with open(osp.join(dst_patient_path, 'answer.txt'), 'w') as f:
                f.write(qa_item['answer'])

            i_train += 1

    dst_test_root = osp.join(dst_data_root, 'test')
    dst_test_w_answer_root = osp.join(dst_data_root, 'test_with_answer')
    i_test = 0
    for p in tqdm(test_paths):
        dcm_p = osp.join(p, 'PR', 'dcm.dcm')
        label_p = osp.join(p, 'PR', 'label.dcm')
        text_p = glob(osp.join(p, '*.docx'))[0]

        assert osp.exists(dcm_p), dcm_p
        assert osp.exists(label_p), label_p
        assert osp.exists(text_p), text_p

        medical_record, qa_list = parse_docx(text_p)

        for qa_item in qa_list:
            dst_patient_path = osp.join(dst_test_root, f'{i_test:03d}')
            if not osp.exists(dst_patient_path): os.makedirs(dst_patient_path)

            dst_patient_path_with_answer = osp.join(dst_test_w_answer_root, f'{i_test:03d}')
            if not osp.exists(dst_patient_path_with_answer): os.makedirs(dst_patient_path_with_answer)

            shutil.copy2(dcm_p, osp.join(dst_patient_path, 'panoramic_radiograph.dcm'))
            shutil.copy2(label_p, osp.join(dst_patient_path, 'segmentation.dcm'))
            with open(osp.join(dst_patient_path, 'medical_record.txt'), 'w') as f:
                f.write(medical_record)
            with open(osp.join(dst_patient_path, 'question.txt'), 'w') as f:
                f.write(qa_item['question'])

            shutil.copy2(dcm_p, osp.join(dst_patient_path_with_answer, 'panoramic_radiograph.dcm'))
            shutil.copy2(label_p, osp.join(dst_patient_path_with_answer, 'segmentation.dcm'))
            with open(osp.join(dst_patient_path_with_answer, 'medical_record.txt'), 'w') as f:
                f.write(medical_record)
            with open(osp.join(dst_patient_path_with_answer, 'question.txt'), 'w') as f:
                f.write(qa_item['question'])
            with open(osp.join(dst_patient_path_with_answer, 'answer.txt'), 'w') as f:
                f.write(qa_item['answer'])

            i_test += 1
        
