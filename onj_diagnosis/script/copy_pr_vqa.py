import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk

try:
    from cv2 import cv2 
except:
    import cv2


if __name__=='__main__':

    src_root = r'G:\txt+dcm\gsy'
    tgt_root = r'G:\pr_vqa'

    for om in ['chemical', 'drug', 'radiation', 'other']:
        text_paths = glob(osp.join(src_root, om, '*', '*.txt'))

        for text_path in tqdm(text_paths):
            person_id = osp.basename(osp.dirname(text_path))

            pr_path = osp.join(src_root, om, person_id, 'PR', 'dcm.dcm')
            if not osp.exists(pr_path): continue

            image = sitk.GetArrayFromImage(sitk.ReadImage(pr_path))[0]
            image = cv2.resize(image, (224,224))

            tgt_pr_path = osp.join(tgt_root, om, person_id, 'pr.jpg')
            if not osp.exists(osp.dirname(tgt_pr_path)):
                os.makedirs(osp.dirname(tgt_pr_path))
            cv2.imwrite(tgt_pr_path, image)

            with open(text_path, 'r') as f:
                data = f.read()
            tgt_text_path = osp.join(tgt_root, om, person_id, 'text.txt')
            with open(tgt_text_path, 'w') as f:
                f.write(data)
