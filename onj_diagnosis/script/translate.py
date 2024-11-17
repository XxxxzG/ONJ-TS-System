
import os.path as osp
from glob import glob
from tqdm import tqdm

from docx import Document

from googletrans import Translator  



def read_docx(path):
    doc = Document(path)  
    
    ret = None
    # 遍历文档中的每一个段落  
    for i, para in enumerate(doc.paragraphs):
        if ret == None:
            ret = para.text
        else:
            ret += '\r\n' + para.text

        if i >= 5: break
    
    return ret


if __name__=='__main__':
      
    data_root = r'G:\txt+dcm'

    translator = Translator()

    # for om in ['chemical', 'drug', 'radiation', 'other']:
    for om in ['radiation', 'other']:
        chinese_language_paths = glob(osp.join(data_root, 'gsy', om, '*', '*.docx'))

        for cns_p in tqdm(chinese_language_paths):
            cns_fn = osp.basename(cns_p).split('.')[0]
            eng_p = osp.join(osp.dirname(cns_p), cns_fn+'.txt')

            cns_text = read_docx(cns_p)
  
            eng_text = translator.translate(cns_text, src='zh-cn', dest='en').text

            with open(eng_p, 'w') as f:
                f.write(eng_text)