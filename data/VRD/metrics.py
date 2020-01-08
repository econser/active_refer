import numpy as np
import json as j

CLASSES = ["person", "sky", "building", "truck", "bus", "table", "shirt", "chair", "car", "train", "glasses", "tree", "boat", "hat", "trees", "grass", "pants", "road", "motorcycle", "jacket", "monitor", "wheel", "umbrella", "plate", "bike", "clock", "bag", "shoe", "laptop", "desk", "cabinet", "counter", "bench", "shoes", "tower", "bottle", "helmet", "stove", "lamp", "coat", "bed", "dog", "mountain", "horse", "plane", "roof", "skateboard", "traffic light", "bush", "phone", "airplane", "sofa", "cup", "sink", "shelf", "box", "van", "hand", "shorts", "post", "jeans", "cat", "sunglasses", "bowl", "computer", "pillow", "pizza", "basket", "elephant", "kite", "sand", "keyboard", "plant", "can", "vase", "refrigerator", "cart", "skis", "pot", "surfboard", "paper", "mouse", "trash can", "cone", "camera", "ball", "bear", "giraffe", "tie", "luggage", "faucet", "hydrant", "snowboard", "oven", "engine", "watch", "face", "street", "ramp", "suitcase"]

PREDICATES = ["on", "wear", "has", "next to", "sleep next to", "sit next to", "stand next to", "park next", "walk next to", "above", "behind", "stand behind", "sit behind", "park behind", "in the front of", "under", "stand under", "sit under", "near", "walk to", "walk", "walk past", "in", "below", "beside", "walk beside", "over", "hold", "by", "beneath", "with", "on the top of", "on the left of", "on the right of", "sit on", "ride", "carry", "look", "stand on", "use", "at", "attach to", "cover", "touch", "watch", "against", "inside", "adjacent to", "across", "contain", "drive", "drive on", "taller than", "eat", "park on", "lying on", "pull", "talk", "lean on", "fly", "face", "play with", "sleep on", "outside of", "rest on", "follow", "hit", "feed", "kick", "skate on"]

def count_relations(j_cfg, top_n=10):
    tracker = np.zeros((100,70,100), dtype=np.int)
    for key in j_cfg:
        j_anno = j_cfg[key]
        for rel in j_anno:
            pred_ix = int(rel['predicate'])
            sub_ix = int(rel['subject']['category'])
            obj_ix = int(rel['object']['category'])
            tracker[sub_ix, pred_ix, obj_ix] += 1
    
    flat_tracker = tracker.flatten()
    flat_argsort = np.argsort(flat_tracker)[::-1]
    ixs = np.unravel_index(flat_argsort, tracker.shape)
    
    for i in range(top_n):
        sub_ix = ixs[0][i]
        sub_str = CLASSES[sub_ix]
        pred_ix = ixs[1][i]
        pred_str = PREDICATES[pred_ix]
        obj_ix = ixs[2][i]
        obj_str = CLASSES[obj_ix]
        n_instances = tracker[sub_ix, pred_ix, obj_ix]
        print('{} {} {} : {}'.format(sub_str, pred_str, obj_str, n_instances))

def output_metrics(j_cfg, out_fname, min_instances=1):
    tracker = np.zeros((100,70,100), dtype=np.int)
    for key in j_cfg:
        j_anno = j_cfg[key]
        for rel in j_anno:
            pred_ix = int(rel['predicate'])
            sub_ix = int(rel['subject']['category'])
            obj_ix = int(rel['object']['category'])
            tracker[sub_ix, pred_ix, obj_ix] += 1
    
    flat_tracker = tracker.flatten()
    flat_argsort = np.argsort(flat_tracker)[::-1]
    ixs = np.unravel_index(flat_argsort, tracker.shape)

    f = open(out_fname, 'wb')
    i = 0
    n_instances = tracker[ixs[0][i], ixs[1][i], ixs[2][i]]
    while n_instances >= min_instances:
        sub_ix = ixs[0][i]
        pred_ix = ixs[1][i]
        obj_ix = ixs[2][i]
        
        sub_str = CLASSES[sub_ix]
        pred_str = PREDICATES[pred_ix]
        obj_str = CLASSES[obj_ix]
        n_instances = tracker[sub_ix, pred_ix, obj_ix]
        
        f.write('{},{},{},{}'.format(sub_str, pred_str, obj_str, n_instances))
        i += 1
    f.close()
        
def get_dataset(set_type='train'):
    fname = '/home/econser/research/active_refer/data/VRD/annotations_{}.json'.format(set_type)
    with open(fname,'rb') as f:
        j_cfg = j.load(f)
    return j_cfg
