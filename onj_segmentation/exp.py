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


command = f'python -m train'
os.system(command)


# command = f'python -m eval'
# os.system(command)