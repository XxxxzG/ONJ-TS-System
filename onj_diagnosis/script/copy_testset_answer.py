import os
import os.path as osp
import shutil
from tqdm import tqdm

from glob import glob
from googletrans import Translator

src_dir = '../vqa_dataset/test_with_answer'
dst_dir = '../vqa_dataset/test_answer'


translator = Translator()
def translate_text(text, src='zh-cn', dest='en'):
    try:
        translated = translator.translate(text, src=src, dest=dest)
        return translated.text
    except Exception as e:
        print(f"An error occurred while translating: {e}")
        return None


if __name__=='__main__':

    for p in tqdm(glob(osp.join(src_dir, '*', 'answer.txt'))):
        item_id = osp.basename(osp.dirname(p))
        dst_p = osp.join(dst_dir, item_id, 'answer.txt')
        if not osp.exists(osp.dirname(dst_p)):
            os.makedirs(osp.dirname(dst_p))
        shutil.copy2(p, dst_p)

        with open(dst_p, 'r', encoding='gbk') as f:
            text_cn = f.read()
        text_en  = translate_text(text_cn)
        assert text_en is not None
        with open(osp.join(osp.dirname(dst_p), 'answer_en.txt'), 'w') as f:
            f.write(text_en)