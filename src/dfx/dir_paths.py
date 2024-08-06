import os
from .wd import working_dir

def get_path(dir: str):

    paths = {'dataset': os.path.join(working_dir, 'datasets'),\
             'data_robustness': os.path.join(working_dir, 'testing_robustness'),\
             'data_generalization': os.path.join(working_dir, 'testing_generalization'),\
             'guidance': os.path.join(working_dir, 'guidance.csv'),\
             'models': os.path.join(working_dir, 'models/unbalancing-approach')}

    return paths[dir]