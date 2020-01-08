#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
from __future__ import print_function
import numpy as np
import os
import sys
import argparse
import glob
import time

import caffe
from sklearn.metrics import classification_report

CLASSES = ('__background__', 'person', 'sky', 'building', 'truck', 'bus', 'table', 'shirt', 'chair', 'car', 'train', 'glasses', 'tree', 'boat', 'hat', 'trees', 'grass', 'pants', 'road', 'motorcycle', 'jacket', 'monitor', 'wheel', 'umbrella', 'plate', 'bike', 'clock', 'bag', 'shoe', 'laptop', 'desk', 'cabinet', 'counter', 'bench', 'shoes', 'tower', 'bottle', 'helmet', 'stove', 'lamp', 'coat', 'bed', 'dog', 'mountain', 'horse', 'plane', 'roof', 'skateboard', 'traffic light', 'bush', 'phone', 'airplane', 'sofa', 'cup', 'sink', 'shelf', 'box', 'van', 'hand', 'shorts', 'post', 'jeans', 'cat', 'sunglasses', 'bowl', 'computer', 'pillow', 'pizza', 'basket', 'elephant', 'kite', 'sand', 'keyboard', 'plant', 'can', 'vase', 'refrigerator', 'cart', 'skis', 'pot', 'surfboard', 'paper', 'mouse', 'trash can', 'cone', 'camera', 'ball', 'bear', 'giraffe', 'tie', 'luggage', 'faucet', 'hydrant', 'snowboard', 'oven', 'engine', 'watch', 'face', 'street', 'ramp', 'suitcase')

def main(argv):
    #args = parse(args)

    # pull and prep args
    mdl_arch = '/home/econser/research/active_refer/models/definitions/vrd_vgg16/deploy.prototxt'
    mdl_weights = '/home/econser/research/active_refer/models/weights/vrd_vgg16_3k.caffemodel'
    #mdl_weights = '/home/econser/research/active_refer/models/weights/vrd_vgg16_L0260287.caffemodel'
    #mdl_weights = '/home/econser/research/active_refer/models/weights/vrd_vgg16_iter_48000.caffemodel'
    #mdl_weights = '/home/econser/research/active_refer/models/weights/vrd_vgg16_iter_10000.caffemodel'

    do_multi_crop = False # single or multiple crops
    image_dims = [256, 256]#np.array([256,256], dtype=np.int)
    image_mean = np.array([86,185,238], dtype=np.int) # note: this is BGR
    pre_scale = 255.0
    post_scale = None
    channel_map = [2,1,0] # RGB -> BGR
    
    cpu_mode = True
    cpu_id = 0

    input_data = '/home/econser/research/active_refer/data/VRD/vrd_vgg/test/'
    out_fname = '/home/econser/research/active_refer/data/VRD/vrd_vgg/predictions.npy'
    
    # initialize
    if cpu_mode:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    
    # instantiate a classifier
    classifier = caffe.Classifier(mdl_arch, mdl_weights, image_dims = image_dims, mean = image_mean, input_scale = post_scale, raw_scale = pre_scale, channel_swap = channel_map)

    # handle image dir and single-image differently
    all_class_preds = []
    all_image_fnames = []
    if os.path.isdir(input_data):
        # run the images in batches
        all_image_fnames = os.listdir(input_data)
        all_image_fnames = filter(lambda x: x.endswith('.jpg'), all_image_fnames)
        n_total_images = len(all_image_fnames)
        images_per_batch = 500
        n_batches = (n_total_images / images_per_batch) + 1

        for batchnum in xrange(n_batches):
            mdl_input = []
            min_img_ix = batchnum * images_per_batch
            max_img_ix = min(min_img_ix + images_per_batch, n_total_images)
            print('starting batch {:03d} ({:04d}-{:04d})...'.format(batchnum,min_img_ix,max_img_ix), end='')
            for ix in range(min_img_ix, max_img_ix):
                image_fname = all_image_fnames[ix]
                img = caffe.io.load_image(os.path.join(input_data, image_fname))
                mdl_input.append(img)
            preds = classifier.predict(mdl_input, do_multi_crop)
            pred_cls = np.argmax(preds, axis=1)
            all_class_preds.append(pred_cls)
            print('done')
    else:
        all_image_fnames.append(input_data.split('/')[-1])
        mdl_input = [caffe.io.load_image(input_data)]
        preds = classifier.predict(mdl_input, do_multi_crop)
        pred_cls = np.argmax(preds, axis=1)
        all_class_preds.append(pred_cls)

    all_class_preds = np.concatenate(all_class_preds).ravel()
    cls_ixs = [int(fname.split('-')[-1].split('.')[0]) for fname in all_image_fnames]
    cls_ixs = np.array(cls_ixs, dtype=np.int)
    #import pdb;pdb.set_trace()
    print(classification_report(cls_ixs, all_class_preds, target_names=CLASSES))
    #pass

#    #==================================================
#    img_names = []
#    if os.path.isdir(input_data):
#        mdl_input = []
#        for img_ix, img_fname in enumerate():
#            # TODO: store input names & classes
#            img_names.append(img_fname)
#            img = caffe.io.load_image(os.path.join(input_data, img_fname))
#            mdl_input.append(img)
#            if img_ix > 10:
#                break
#        #mdl_input = [caffe.io.load_image(fname) for fname in glob.glob(input_data + '/*.jpg')]
#    else:
#        img_names.append(input_data.split('/')[-1])
#        mdl_input = [caffe.io.load_image(input_data)]
#
#    # classify the input
#    preds = classifier.predict(mdl_input, do_multi_crop)
#    #np.save(out_fame, preds)
#
#    # run performance metrics
#    import pdb;pdb.set_trace()
#    cls_ixs = [ix for img_name.split('-')[-1].split('.')[0] in img_names]
#    cls_ixs = np.array(cls_ixs, dtype=np.int)
#    pred_cls = np.argmax(preds, axis=1)
#    pass
    
if __name__ == '__main__':
    main(sys.argv)
