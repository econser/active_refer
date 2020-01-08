import json as j
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=240)
import model_set as m
import viz as v
from conditional_viz import ConditionalViz
import diagnostics as diag
import time
import math
import os
import random
#TODO: compare x0y0 (min-value) with xcyc (bbox-center)

import ar_utils as au

ITERATION_THERSHOLD = 1000
#===============================================================================
# calls for pulling relevant data from the full dataset
# TODO : paramaterize the subj/obj indices?
#
def get_full_matches(dataset, q_sub, q_pred, q_obj):
    full_matches = []
    if q_pred not in dataset:
        return full_matches
    
    for d in dataset[q_pred]:
        if d[0] == q_sub and d[5] == q_obj:
            if not sub_is_present(d[1:5], full_matches) and not obj_is_present(d[6:10], full_matches):
                full_matches.append(d)

    return full_matches

def get_subj_matches(dataset, q_sub, q_pred):
    subj_matches = []
    if q_pred not in dataset:
        return subj_matches
    
    for d in dataset[q_pred]:
        if d[0] == q_sub:
            subj_matches.append(d)

    return subj_matches

def get_obj_matches(dataset, q_pred, q_obj):
    obj_matches = []
    if q_pred not in dataset:
        return obj_matches
    
    for d in dataset[q_pred]:
        if d[5] == q_obj:
            obj_matches.append(d)

    return obj_matches

def sub_is_present(new_bbox, current_bboxes):
    for row in current_bboxes:
        if new_bbox == row[1:5]: # TODO: fix bogus code
            return True
    return False

def obj_is_present(new_bbox, current_bboxes):
    for row in current_bboxes:
        if new_bbox == row[6:10]: # TODO: fix bogus code
            return True
    return False



#===============================================================================
# calls for getting configuration data from json files
#
def get_query(config, cls_dict, predicate_dict):
    """ get a referring relationships query
    in:  json config (root),
         class name to ix dict,
         predicate to ix dict
    out: referring relationship string "person kick ball"
         seperate subject, predicate, & object strings: "person", "kick", "ball"
         seperate s,p,o indices: 0, 68, 85
    """
    return au.RRQuery(config, cls_dict, predicate_dict)

def get_dataset(config):
    # training set
    with open(config['train_sizes'], 'rb') as f:
        img_sizes = j.load(f)
    
    fname = config['train_anno']
    with open(fname, 'rb') as f:
        train_cfg = j.load(f)
        train_data = process_json_anno(train_cfg, img_sizes)
    train_imgs = config['train_imgs']

    # test set
    with open(config['test_sizes'], 'rb') as f:
        img_sizes = j.load(f)
        
    fname = config['test_anno']
    with open(fname, 'rb') as f:
        test_cfg = j.load(f)
        test_data = process_json_anno(test_cfg, img_sizes)
    test_imgs = config['test_imgs']

    # validation set
    if 'val_anno' in config:
        with open(config['val_sizes'], 'rb') as f:
            img_sizes = j.load(f)
            
        fname = config['val_anno']
        with open(fname, 'rb') as f:
            val_cfg = j.load(f)
            val_data = process_json_anno(val_cfg, img_sizes)
        val_imgs = config['val_imgs']
    else:
        val_data = None
        val_imgs = None
    
    return {'train_data' : train_data,
            'train_imgs' : train_imgs,
            'test_data'  : test_data,
            'test_imgs'  : test_imgs,
            'val_data'   : val_data,
            'val_imgs'   : val_imgs}

def process_json_anno(json_cfg, size_dict):
    """ Process a json annotation into a usable format
    """
    data_dict = {}
    for img_name in json_cfg:
        if img_name not in size_dict:
            continue # TODO: fix here --- this shouldn't be necessary
        
        im_data = size_dict[img_name]
        for relation in json_cfg[img_name]:
            predicate_ix = relation['predicate']
            object_cls = relation['object']['category']
            object_bbox = relation['object']['bbox']
            subject_cls = relation['subject']['category']
            subject_bbox = relation['subject']['bbox']
            rel_tuple = (subject_cls, subject_bbox[0], subject_bbox[1], subject_bbox[2], subject_bbox[3], object_cls, object_bbox[0], object_bbox[1], object_bbox[2], object_bbox[3], img_name, im_data['height'], im_data['width'])
            
            if not predicate_ix in data_dict:
                data_dict[predicate_ix] = []
            data_dict[predicate_ix].append(rel_tuple)
    return data_dict



#===============================================================================
def preprocess_tblr_data(data):
    # input data is:
    #      0,      1,      2,      3,    4,    5
    # bbox_t, bbox_b, bbox_l, bbox_r, im_h, im_w
    #     y0,     y1,     x0,     x1, im_h, im_w
    #
    # output data is:
    # nrm_x0, nrm_x1, nrm_y0, nrm_y1, im_w, im_h
    #
    # This converts top-bottom-left-right bboxes to normalized image coordinates
    # the normalized image center is 0,0 and its max extent is from (-0.5, -0.5) to
    # (+0.5, +0.5)
    
    #out = np.copy(data)
    #
    #out[:,0] = data[:,2]
    #out[:,1] = data[:,3]
    #out[:,2] = data[:,0]
    #out[:,3] = data[:,1]
    #
    #out[:,4] = data[:,5]
    #out[:,5] = data[:,4]
    #
    ## center each box
    #shift = np.array((out[:,4], out[:,4], out[:,5], out[:,5])).T / 2.
    #out[:,0:4] -= shift
    #
    ## scale the bboxes relative to unit size image
    ##scale_factor = np.sqrt(1. / np.prod((out[:,4], out[:,5]), axis=0))
    ##scale_factor = np.reshape(np.tile(inv_diag, 4), (4,-1)).T
    ##scale_factor = np.tile(scale_factor[:,np.newaxis],4)
    #
    #scale_factor = 1 / np.array((out[:,4], out[:,4], out[:,5], out[:,5])).T
    #out[:,0:4] *= scale_factor

    out = np.copy(data)
    out = au.iyyxx_to_nxxyy(out[:,0:4], out[:,5], out[:,4])
    out = np.hstack((out, data[:,5][:,np.newaxis], data[:,4][:,np.newaxis]))
    return out
    
    

#===============================================================================
def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser(description='Active Referring Relationships')
    parser.add_argument('--config', dest='cfg_fname', help='config filename')
    parser.add_argument('--train', dest='do_training', help='train the modelset?')
    parser.add_argument('--image', dest='img_fname', help='image filename')
    parser.add_argument('--query', dest='query_str', help='subject-relationship-object query')
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    
    cfg_fname = args.cfg_fname
    if cfg_fname is None: # TODO: remove this
        cfg_fname = '/home/econser/research/active_refer/data/vrd_cfg_.json' # TODO: remove this
    with open(cfg_fname, 'rb') as f:
        cfg = j.load(f)

    # get the referring relationship query
    with open(cfg['object_list'], 'rb') as f:
        obj_list = j.load(f)
        obj_dict = {k: v for v, k in enumerate(obj_list)}
    with open(cfg['predicate_list'], 'rb') as f:
        pred_list = j.load(f)
        pred_dict = {k: v for v, k in enumerate(pred_list)}
    query = get_query(cfg, obj_dict, pred_dict)
    
    # get the dataset
    dataset = get_dataset(cfg['dataset'])

    full_relations = get_full_matches(dataset['train_data'], query.subject_ix, query.predicate_ix, query.object_ix)
    subj_relation = get_subj_matches(dataset['train_data'], query.subject_ix, query.predicate_ix)
    obj_relation = get_obj_matches(dataset['train_data'], query.predicate_ix, query.object_ix)
    
    # get the relevant data for the relationship
    full_relations = np.array(full_relations, dtype=np.object)

    subj_cols = np.r_[1:5,11,12]
    subj_data = np.array(full_relations[:, subj_cols], dtype=np.float)
    subj_data = preprocess_tblr_data(subj_data) # TODO: parameterize preproc fn from cfg file
    
    obj_cols = np.r_[6:10,11,12]
    obj_data = np.array(full_relations[:, obj_cols], dtype=np.float)
    obj_data = preprocess_tblr_data(obj_data) # TODO: parameterize preproc fn from cfg file

    # full_data is in xyxy format:
    # sub_x0, sub_x1, sub_y0, sub_y1, img_w, img_h, obj_x0, obj_x1, obj_y0, obj_y1, img_w, img_h 
    full_data = np.hstack((subj_data, obj_data))
    
    # prep the models and train
    # TODO : save/load trained modelset
    models = m.ModelSet(cfg['models'])
    
    models.subject_priors.train(subj_data)
    models.object_priors.train(obj_data)
    models.relationship.train(full_data)

    #===========================================================================
    # conditional viz testing ==================================================
    image_fname = '/home/econser/Pictures/8438726923_b82debb62e_b.jpg'
    #image_fname = "/home/econser/research/active_refer/data/cond_test/images/test_grid.jpg"
    cviz = ConditionalViz(models.subject_priors, models.object_priors, models.relationship, query.subject_str, query.predicate_str, query.object_str, image_fname)
    while True:#for i in range(10):
        cviz.step()
        time.sleep(0.5)

    #===========================================================================
    # fire up the viz test window ==============================================
    do_viz = True # TODO: parameterize this
    if do_viz:
        viz = v.Viz(query.subject_str, query.predicate_str, query.object_str)
        base_dir = '/home/econser/research/irsg_psu_pdx/data/sg_dataset/sg_train_images/'
        img_ix = random.randint(0,len(full_relations))
        img_fnames = [os.path.join(base_dir, full_relations[img_ix, 10])]
        detections = models.roi_generator.get_rois(img_fnames[0], [query.subject_ix, query.object_ix])
        import pdb;pdb.set_trace()

        subj_gt = full_relations[img_ix,1:5].astype(np.int) # tblr -> xywh
        subj_gt = np.array((subj_gt[2], subj_gt[0], subj_gt[3] - subj_gt[2], subj_gt[1] - subj_gt[0])).astype(np.int)
        obj_gt = full_relations[img_ix,6:10].astype(np.int) # tblr -> xywh
        obj_gt = np.array((obj_gt[2], obj_gt[0], obj_gt[3] - obj_gt[2], obj_gt[1] - obj_gt[0])).astype(np.int)
        gt_bboxes = np.vstack((subj_gt, obj_gt))
        diag.viz_test_full(viz, img_fnames, models.obj_detector, models.subject_priors, models.relationship.subject_conditional, models.relationship.subject_calibration, models.object_priors, models.relationship.object_conditional, models.relationship.object_calibration, detections, query.subject_ix, query.object_ix, gt_bboxes=gt_bboxes, sleeptime=0.0)
    else:
        viz = None
    
    #==========================================================================
    # process the test set
    #
    pass
