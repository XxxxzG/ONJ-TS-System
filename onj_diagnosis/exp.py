# python exp.py

import os
import os.path as osp
import platform


# helpers
def add_pythonpath(p):

    if platform.system() == 'Windows':
        SEP = ';'
    elif platform.system() == 'Linux':
        SEP = ':'
    else:
        raise NotImplementedError
    
    pythonpath = os.environ.get('PYTHONPATH')
    pythonpath = p + SEP + pythonpath if pythonpath else p
    os.environ['PYTHONPATH'] = pythonpath


# configuration
PROJECT_ROOT    = osp.join('../../')
add_pythonpath(PROJECT_ROOT)


for model_name in [
    'swin_tiny_patch4_window7_224', 
    'crossvit_tiny_240'
    'vit_small_patch16_224',
    'vgg16_bn', 
    'resnet50', 
    'inception_v3', ]:
    for fold in [0, 1, 2, 3, 4]:

        command = f'python -m train --fold {fold} --model_name {model_name}'
        os.system(command)

# command = f'python -m tsne'
# os.system(command)

# command = f'python -m cam'
# os.system(command)