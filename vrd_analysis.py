import numpy as np
import json as j

def get_counts(json_anno, size=(100,70,100)):
    count = np.zeros(size, dtype=np.int)
    for img_name in json_anno:
        for relation in json_anno[img_name]:
            sub = relation['subject']['category']
            pred = relation['predicate']
            obj = relation['object']['category']
            count[sub, pred, obj] += 1
    return count
# count of each subject-predicate-object relationship (100x70x100)
# per-predicate: count subjects, count objects, count 

#=========================================================================================
def t():
    cfg_fname = '/home/econser/research/active_refer/data/VRD/annotations_train.json'
    with open(cfg_fname, 'rb') as f:
        cfg = j.load(f)
    return get_counts(cfg)
