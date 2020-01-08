#import _init_paths
#import matplotlib; matplotlib.use('agg') #when running remotely
import sys
sys.path.append('/home/econser/usr/py-faster-rcnn/lib')

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
#from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import json as j

CLASSES = ('__background__', 'person', 'sky', 'building', 'truck', 'bus', 'table', 'shirt', 'chair', 'car', 'train', 'glasses', 'tree', 'boat', 'hat', 'trees', 'grass', 'pants', 'road', 'motorcycle', 'jacket', 'monitor', 'wheel', 'umbrella', 'plate', 'bike', 'clock', 'bag', 'shoe', 'laptop', 'desk', 'cabinet', 'counter', 'bench', 'shoes', 'tower', 'bottle', 'helmet', 'stove', 'lamp', 'coat', 'bed', 'dog', 'mountain', 'horse', 'plane', 'roof', 'skateboard', 'traffic light', 'bush', 'phone', 'airplane', 'sofa', 'cup', 'sink', 'shelf', 'box', 'van', 'hand', 'shorts', 'post', 'jeans', 'cat', 'sunglasses', 'bowl', 'computer', 'pillow', 'pizza', 'basket', 'elephant', 'kite', 'sand', 'keyboard', 'plant', 'can', 'vase', 'refrigerator', 'cart', 'skis', 'pot', 'surfboard', 'paper', 'mouse', 'trash can', 'cone', 'camera', 'ball', 'bear', 'giraffe', 'tie', 'luggage', 'faucet', 'hydrant', 'snowboard', 'oven', 'engine', 'watch', 'face', 'street', 'ramp', 'suitcase')

def get_annotations(anno_path, dataset_name, classes):
    with open(os.path.join(anno_path, '{}_image_metadata.json'.format(dataset_name)), 'rb') as f:
        fsize_dict = j.load(f)
    
    train_dict = {}
    with open(os.path.join(anno_path, 'annotations_{}.json'.format(dataset_name)), 'rb') as f:
        j_anno = j.load(f)
    for anno_key in j_anno:
        anno_key = anno_key.encode('ascii', 'ignore')
        
        if anno_key not in fsize_dict:
            continue
        
        img_width = fsize_dict[anno_key]['width']
        img_height = fsize_dict[anno_key]['height']
        img_size = (img_width, img_height)
        
        bboxes = []
        anno_classes = set()
        for rel in j_anno[anno_key]:
            s = rel['subject']
            sub_ix = int(s['category']) + 1
            anno_classes.add(sub_ix)
            sub_cls_name = classes[sub_ix]
            b = s['bbox'] # y0, y1, x0, x1
            sub_bbox = [b[2], b[0], b[3], b[1]] # x0, y0, x1, y1
            anno_tup = (sub_cls_name, sub_bbox)
            if anno_tup not in bboxes:
                bboxes.append(anno_tup)
            
            o = rel['object']
            obj_ix = int(o['category']) + 1
            anno_classes.add(obj_ix)
            obj_cls_name = classes[obj_ix]
            b = o['bbox']
            obj_bbox = [b[2], b[0], b[3], b[1]]
            anno_tup = (obj_cls_name, obj_bbox)
            if anno_tup not in bboxes:
                bboxes.append(anno_tup)
                
        train_dict[anno_key] = (img_size, bboxes, anno_classes)
    return train_dict    



def get_ss_bboxes(ss_data, image_fname):
    image_key = '{:<24}'.format(image_fname.split('.')[0])
    ix = np.where(ss_data['images'] == image_key)[0][0]
    bboxes = ss_data['boxes'][0][ix]
    return bboxes



def vis_detections(im, class_name, dets, anno, thresh=0.5, top_n=None):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return False

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for box_ix, i in enumerate(inds):
        if top_n is not None and box_ix >= top_n:
            break
        
        bbox = dets[i, :4]
        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - x
        h = bbox[3] - y
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue', linewidth=3.5)
        ax.add_patch(rect)

        score = dets[i, -1]
        label = '{:s} {:.3f}'.format(class_name, score)
        ax.text(x + 3, y - 8, label, bbox=dict(facecolor='blue', alpha=0.5), fontsize=14, color='white')

    # draw any GT bboxes
    for obj in anno[1]:
        if obj[0] == class_name:
            bbox = obj[1]
            x = bbox[0]
            y = bbox[1]
            w = bbox[2] - x
            h = bbox[3] - y
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=4.0)
            ax.add_patch(rect)

    if top_n is not None:
        title = 'Top {} detections for {} objects'.format(top_n, class_name)
    else:
        title = '{} detections with ''p({} | box) >= {:.2f}'.format(class_name, class_name, thresh)
    ax.set_title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return True



#===============================================================================
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='rcnn vrd visualizer')
    parser.add_argument('--type', dest='mdl_type', help='[fast] or [faster] R-CNN', type=str, default='faster')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # configure GPU
    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0

    # get parameters
    args = parse_args()
    
    save_output = False
    TOP_N = 3
    NMS_THRESH = 0.5
    bbox_dict = {}

    import pdb;pdb.set_trace()
    if args.mdl_type == 'fast':
        # fast r-cnn setup
        cfg.TEST.HAS_RPN = False # Use RPN for proposals
        model_file = '/home/econser/research/active_refer/models/definitions/vrd_fast/test.prototxt'
        weights = '/home/econser/research/active_refer/models/weights/vrd_fast_100k.caffemodel'
        fq_fname_ss = '/home/econser/research/active_refer/data/VRD/selective_search/test.mat'
        bbox_mat = sio.loadmat(fq_fname_ss)
        
        for ix, bbox_key in enumerate(bbox_mat['images']):
            k = bbox_key.encode('ascii', 'ignore').split(' ')[0]
            v = bbox_mat['boxes'][0][ix]
            bbox_dict[k] = v
    elif args.mdl_type == 'faster':
        # faster r-cnn setup
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        model_file = '/home/econser/research/active_refer/models/definitions/vrd_faster/faster_rcnn_end2end/test.prototxt'
        weights = '/home/econser/research/active_refer/models/weights/vrd_faster_500k.caffemodel'
    else:
        sys.exit(0)
    
    out_dir = '/home/econser/research/active_refer/output/vrd_faster_viz/'
    img_dir = '/home/econser/research/irsg_psu_pdx/data/sg_dataset/sg_test_images'

    anno_path = '/home/econser/research/active_refer/data/VRD'
    
    # initialize
    img_fnames = os.listdir(img_dir)
    n_images = len(img_fnames)

    if save_output:
        for cls in CLASSES[1:]:
            out_cls_dir = os.path.join(out_dir, cls)
            if not os.path.exists(out_cls_dir):
                os.makedirs(out_cls_dir)

    annos = get_annotations(anno_path, 'test', CLASSES)

    # initialize the net
    net = caffe.Net(model_file, weights, caffe.TEST)

    # run each image
    for i, img_fname in enumerate(img_fnames):
        if img_fname.endswith('.gif'):
            continue
        
        print('generating image {} ({:0.2f}%)'.format(img_fname, i/float(n_images) * 100.0))
        
        img_fqname = os.path.join(img_dir, img_fname)
        
        image = cv2.imread(img_fqname)
        bboxes = bbox_dict.get(img_fname.split('.')[0])
        scores, boxes = im_detect(net, image, bboxes)

        # show top N class detections for annotated classes
        anno_classes = annos[img_fname][2]
        for cls_ind in anno_classes:
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            sort_ixs = np.argsort(dets[:,4])
            dets = dets[sort_ixs][::-1]
            keep_ixs = nms(dets, NMS_THRESH)
            dets = dets[keep_ixs]
            vis_detections(image, CLASSES[cls_ind], dets, annos[img_fname], thresh=0.0, top_n=TOP_N)
        
        # show RCNN classes with confidence > threshold that are not in annotations (false positives)
        CONF_THRESH = 0.75
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            #print('{} : {}'.format(cls, dets[:,-1][:5]))
            do_save = vis_detections(image, cls, dets, annos[img_fname], thresh=CONF_THRESH)
            
            if do_save and False:
                out_cls_dir = os.path.join(out_dir, cls)
                out_fname = img_fname
                out_fqname = os.path.join(out_cls_dir, out_fname)
                print('   {}   {}'.format(cls, out_cls_dir))
                plt.savefig(out_fqname)
                plt.close('all')
