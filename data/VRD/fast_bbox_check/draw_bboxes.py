import os
import json as j
import numpy as np
import scipy.io as sio

import matplotlib.image as image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt



#===============================================================================
VRD_CLASSES = ('__background', 'person', 'sky', 'building', 'truck', 'bus', 'table', 'shirt', 'chair', 'car', 'train', 'glasses', 'tree', 'boat', 'hat', 'trees', 'grass', 'pants', 'road', 'motorcycle', 'jacket', 'monitor', 'wheel', 'umbrella', 'plate', 'bike', 'clock', 'bag', 'shoe', 'laptop', 'desk', 'cabinet', 'counter', 'bench', 'shoes', 'tower', 'bottle', 'helmet', 'stove', 'lamp', 'coat', 'bed', 'dog', 'mountain', 'horse', 'plane', 'roof', 'skateboard', 'traffic light', 'bush', 'phone', 'airplane', 'sofa', 'cup', 'sink', 'shelf', 'box', 'van', 'hand', 'shorts', 'post', 'jeans', 'cat', 'sunglasses', 'bowl', 'computer', 'pillow', 'pizza', 'basket', 'elephant', 'kite', 'sand', 'keyboard', 'plant', 'can', 'vase', 'refrigerator', 'cart', 'skis', 'pot', 'surfboard', 'paper', 'mouse', 'trash can', 'cone', 'camera', 'ball', 'bear', 'giraffe', 'tie', 'luggage', 'faucet', 'hydrant', 'snowboard', 'oven', 'engine', 'watch', 'face', 'street', 'ramp', 'suitcase')



#===============================================================================
def get_gt_bboxes(anno, image_fname):
    if image_fname not in anno:
        return None

    bboxes = []
    for rel in j_anno[image_fname]:
        s = rel['subject']
        sub_ix = int(s['category']) + 1
        sub_cls_name = VRD_CLASSES[sub_ix]
        b = s['bbox'] # y0, y1, x0, x1
        sub_bbox = [b[2], b[0], b[3], b[1]] # x0, y0, x1, y1
        anno_tup = (sub_cls_name, sub_bbox)
        if anno_tup not in bboxes:
            bboxes.append(anno_tup)
        
        o = rel['object']
        obj_ix = int(o['category']) + 1
        obj_cls_name = VRD_CLASSES[obj_ix]
        b = o['bbox']
        obj_bbox = [b[2], b[0], b[3], b[1]]
        anno_tup = (obj_cls_name, obj_bbox)
        if anno_tup not in bboxes:
            bboxes.append(anno_tup)
    return bboxes

def get_ss_bboxes(ss_data, image_fname):
    image_key = '{:<24}'.format(image_fname.split('.')[0])
    ix = np.where(ss_data['images'] == image_key)[0][0]
    bboxes = ss_data['boxes'][0][ix]
    return bboxes



#===============================================================================
def draw_image(image_fname, gt_bboxes, ss_bboxes, outdir=None):
    # assume all bboxes are x0y0x1y1

    fig, ax = plt.subplots()
    
    ax.clear()
    ax.axis('off')
    image_only = image_fname.split('/')[-1]
    fig.suptitle(image_only, fontsize=11)
    img = image.imread(image_fname)
    ax.imshow(img)
    
    # draw ss bboxes in blue
    for bbox in ss_bboxes:
        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - x
        h = bbox[3] - y
        r = Rectangle((x,y),w,h, linewidth=1.0, alpha=0.9, edgecolor='b', facecolor='None')
        ax.add_patch(r)
    
    # draw GT bboxes in green
    for bbox_tup in gt_bboxes:
        cls_name = bbox_tup[0]
        b = bbox_tup[1]
        x = b[0]
        y = b[1]
        w = b[2] - x
        h = b[3] - y
        r = Rectangle((x,y),w,h, linewidth=1.5, alpha=0.9, edgecolor='g', facecolor='None')
        ax.add_patch(r)
        ax.text(x+6, y-10, '{:s}'.format(cls_name), bbox=dict(facecolor='g', alpha=0.5), fontsize=9, color='white')
    
    if outdir is None:
        plt.show()
    else:
        out_fname = os.path.join(outdir, image_only)
        plt.savefig(out_fname)
    plt.close()



#===============================================================================
def parse_args():
    pass

if __name__ == '__main__':
    # parse args
    image_dir = '/home/econser/research/irsg_psu_pdx/data/sg_dataset/sg_train_images/'
    image_fnames = os.listdir(image_dir)
    outdir = '/home/econser/research/active_refer/data/VRD/fast_bbox_check/output/'
    
    # open the anno file
    fq_fname_anno = '/home/econser/research/active_refer/data/VRD/annotations_train.json'
    with open(fq_fname_anno, 'rb') as f:
        j_anno = j.load(f)
        
    # open the ss file
    fq_fname_ss = '/home/econser/research/active_refer/data/VRD/fast_bbox_check/train.mat'
    ss_data = sio.loadmat(fq_fname_ss)
    
    # call draw_image
    n_images = len(image_fnames)
    for ix, image_fname in enumerate(image_fnames):
        print('rendering {:<28} --- {:04d}/{:04d}'.format(image_fname, ix+1, n_images))
        gt_bboxes = get_gt_bboxes(j_anno, image_fname)
        ss_bboxes = get_ss_bboxes(ss_data, image_fname)
        image_fname = os.path.join(image_dir, image_fname)
        draw_image(image_fname, gt_bboxes, ss_bboxes, outdir=outdir)
