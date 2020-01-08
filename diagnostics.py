import time
import numpy as np
import ar_utils as au
import matplotlib.pyplot as plt

_debug_level = 3 # 5: lots, ..., 1: little

def plot_scaling_performance(varset, have_mask, want_mask, cond_mvn, scaling_fn, scaling_params):
    wants = varset[:,want_mask]
    haves = varset[:,have_mask]
    densities = np.ones(len(wants))
    for ix, bbox in enumerate(wants):
        mvn = cond_mvn(bbox)
        all_densities = mvn.pdf(haves)
        densities[ix] = all_densities[ix]
    probs = scaling_fn(densities[:,np.newaxis])

    xs = np.linspace(np.min(densities), np.max(densities), 500)
    ys = 1. / (1. + np.exp(- scaling_params[0] - scaling_params[1] * xs))

    fig, ax1 = plt.subplots()
    ax1.hist(densities, bins=40)
    ax2 = ax1.twinx()
    ax2.plot(xs[:,np.newaxis], ys[:,np.newaxis], color='r')

    plt.show()
    plt.close()

#===============================================================================
# generate an image space bbox from a prior sampler
#
# returns: ixywh bbox
#
def get_prior_sample(sampler, img_w, img_h, box_name=None):
    nxysa_sample = np.array([-1,-1,-1,-1])
    while not au.valid_bbox_nxysa(nxysa_sample):
        nxysa_sample = sampler.sample(1)
        if __debug__ and _debug_level >= 5 and box_name is not None:
            ixywh_bbox = au.nxysa_to_ixywh(nxysa_sample, img_w, img_h)
            x,y,w,h = ixywh_bbox
            print('sampled {} bbox [{:4d}, {:4d}, {:4d}, {:4d}] ({}, {})'.format(box_name, x, y, w, h, img_w, img_h))
    
    if __debug__ and _debug_level >= 3: print('sampled {} from priors'.format(box_name))
    
    ixywh_bbox = au.nxysa_to_ixywh(nxysa_sample, img_w, img_h)
    return ixywh_bbox

#===============================================================================
# generate an image space bbox from a conditional sampler
#
# have_bbox: nxysa (log-space)
#    output: ixywh bbox (None if we couldn't get a valid sample)
#
def get_conditional_sample(sampler, have_bbox, img_w, img_h, box_name=None, max_attempts=50):
    nxysa_sample = np.array([-1,-1,-1,-1])
    n_attempts = 0
    while not au.valid_bbox_nxysa(nxysa_sample):
        n_attempts += 1
        nxysa_sample = sampler(have_bbox).rvs(1)
        nxysa_sample[2:4] = np.exp(nxysa_sample[2:4]) # log-space sample --> regular-space bbox
        if __debug__ and _debug_level >= 5 and box_name is not None:
            #ixywh_sample = au.nxysa_to_ixywh(nxysa_sample, img_w, img_h)
            #x,y,w,h = ixywh_sample
            #print('sampled {} bbox [{:4d}, {:4d}, {:4d}, {:4d}] ({}, {})'.format(box_name, x, y, w, h, img_w, img_h))
            x,y,w,h = nxysa_sample
            print('sampled {} bbox [{:0.6f}, {:0.6f}, {:0.6f}, {:0.6f}] ({}, {}) have-box [{:0.6f}, {:0.6f}, {:0.6f}, {:0.6f}]'.format(box_name, x, y, w, h, img_w, img_h, have_bbox[0], have_bbox[1], have_bbox[2], have_bbox[3]))
        if n_attempts > max_attempts:
            break
        
    if n_attempts > max_attempts:
        return None

    if __debug__ and _debug_level >= 3: print('sampled {} from conditional'.format(box_name))
    
    ixywh_sample = au.nxysa_to_ixywh(nxysa_sample, img_w, img_h)
    return ixywh_sample

#===============================================================================
# Generate an ixywh sample from a set of RCNN-based detections
#   Uses a stochastic acceptance method to select the winning sample
#
# returns: ixywh bbox
#
def get_detection_sample(bboxes, scores, max_attempts=50, box_name=None):
    max_confidence = np.max(scores)
    n_attempts = 0
    sbox = None
    while n_attempts < max_attempts:
        n_attempts += 1
        sample_ix = np.random.randint(len(scores))
        sample_confidence = scores[sample_ix]
        acceptance_val = sample_confidence / max_confidence
        if np.random.random() <= acceptance_val:
            sbox = np.copy(bboxes[sample_ix])
            if __debug__ and _debug_level >= 5 and box_name is not None:
                x, y, w, h = sbox
                print('sampled {} bbox from RCNN: [{:0.6f}, {:0.6f}, {:0.6f}, {:0.6f}]'.format(box_name, x, y, w, h))
            # convert ixyxy to ixywh
            sbox[2:4] -= sbox[0:2]
            break
    
    if (__debug__ and _debug_level >= 3 and sbox is not None): print('sampled {} from RCNN bboxes'.format(box_name))

    return sbox.astype(np.int64)

#===============================================================================
# Generate an ixywh sample from a bbox source
#-------------------------------------------------------------------------------
# 1st choice - conditional MVN
# 2nd choice - FRCNN
# 3rd choice - priors
#
# have_bbox is in nxysa_log format
# returns: ixywh bbox
#
def get_bbox(have_bbox, conditional_mvn, detections, priors, img_w, img_h, box_name=None):
    # try to sample a valid bbox from the conditional mvn N(want_bbox | have_bbox).sample()
    sbox = get_conditional_sample(conditional_mvn, have_bbox, img_w, img_h, box_name)
    
    # pull from rcnn if we didn't get a successful sample
    if sbox is None:
        scores = detections[:,4]
        bboxes = detections[:,0:4]
        sbox = get_detection_sample(bboxes, scores, box_name)
    
    # pull from the prior if we still haven't succeded
    if sbox is None:
        sbox = get_prior_sample(priors, img_w, img_h, box_name)
    
    return sbox

def get_rand_bbox(have_bbox, conditional_mvn, detections, priors, img_w, img_h, box_name=None):
    rnd = np.random.random() * 100.
    if rnd < 50.:
        sbox = get_conditional_sample(conditional_mvn, have_bbox, img_w, img_h, box_name, max_attempts=1000000)
    elif rnd < 75.:
        scores = detections[:,4]
        bboxes = detections[:,0:4]
        sbox = get_detection_sample(bboxes, scores, box_name=box_name)
    else:
        sbox = get_prior_sample(priors, img_w, img_h, box_name=box_name)
    
    return sbox

#===============================================================================
# Pretty print some bboxes
#
# bboxes should be tuples of (coords, name, format), e.g. ([0, 0, 100, 200], 'person', 'ixywh')
#
def print_bboxes(bbox_tups):
    box_strs = []
    for tup in bbox_tups:
        coords, name, fmt = tup
        if coords is None:
            box_str = '{}: ~~~,~~~,~~~,~~~'.format(name)
        else:
            if fmt == 'ixywh':
                if type(coords[0]) != np.int64:
                    import pdb;pdb.set_trace()
                box_str = '{}: {:3d}, {:3d}, {:3d}, {:3d}'.format(name, coords[0], coords[1], coords[2], coords[3])
            elif fmt == 'nxysa':
                box_str = '{}: {: 0.6f}, {: 0.6f}, {: 0.6f}, {: 0.6f}'.format(name, coords[0], coords[1], coords[2], coords[3])
            else:
                box_str = '{}: {}'.format(name, coords)
        box_strs.append(box_str)
    return ' --- '.join(box_strs)

#===============================================================================
# Viz test using priors and conditional MVNs
#
from collections import namedtuple
BoxDescriptor = namedtuple('BoxDescriptor', 'ixywh, nxysa, score, gt_iou') #nxysa should be log-space

def viz_test_full(viz, img_fnames, entity_cnn, sub_sampler, sub_conditional, sub_calibration, obj_sampler, obj_conditional, obj_calibration, detections, subj_cls_ix, obj_cls_ix, gt_bboxes=None, sleeptime=0.0):
    ALPHA = 0.75 # just a guess from situate AUROC based total support calc
    for img_fname in img_fnames:
        viz.set_image(img_fname)
        viz.set_gt_bboxes(gt_bboxes)
        img_w, img_h = viz.get_image_shape()
        viz.draw()

        sub_supports = np.zeros(3)
        obj_supports = np.zeros(3)

        best_bboxes = {'subject' : None,
                        'object' : None}

        while viz.get_state() != 'TERMINATED': #TODO: also check convergence condition
            if viz.get_state() == 'PAUSED':
                viz.draw()
                time.sleep(0.01)
                continue
                
            if None in best_bboxes.values():
                # sample sub & obj priors
                sub_ixywh = get_prior_sample(sub_sampler, img_w, img_h, box_name='subject') # 
                sub_nxysa = au.ixywh_to_nxysa(sub_ixywh[np.newaxis,:], img_w, img_h)[0] # regular space nxysa
                best_bboxes['subject'] = BoxDescriptor(sub_ixywh, sub_nxysa, 0.0, 0.0)
                
                obj_ixywh = get_prior_sample(obj_sampler, img_w, img_h, box_name='object')
                obj_nxysa = au.ixywh_to_nxysa(obj_ixywh[np.newaxis,:], img_w, img_h)[0]
                best_bboxes['object'] = BoxDescriptor(obj_ixywh, obj_nxysa, 0.0, 0.0)
            else:
                # generate the 'have' bbox from rcnn, best_bbox, or prior (randomly)
                sub_ixywh = get_rand_bbox(best_bboxes['subject'].nxysa, sub_conditional, detections[subj_cls_ix], sub_sampler, img_w, img_h, box_name='subject')
                sub_nxysa = au.ixywh_to_nxysa(sub_ixywh[np.newaxis,:], img_w, img_h)[0]
                obj_ixywh = get_rand_bbox(best_bboxes['object'].nxysa, obj_conditional, detections[obj_cls_ix], obj_sampler, img_w, img_h, box_name='object')
                obj_nxysa = au.ixywh_to_nxysa(obj_ixywh[np.newaxis,:], img_w, img_h)[0]


            if __debug__ and _debug_level >= 5:
                #bbox_str = print_bboxes([(sub_ixywh, 'subject', 'ixywh'), (obj_ixywh, 'object', 'ixywh')])
                bbox_str = print_bboxes([(sub_nxysa, 'subject', 'nxysa'), (obj_nxysa, 'object', 'nxysa')])
                print(bbox_str)
            
            # draw and score
            if sub_ixywh is not None:
                #sub_nxysa = au.ixywh_to_nxysa(sub_ixywh[np.newaxis,:], img_w, img_h)
                #sub_nxysa = sub_nxysa[0]
                sub_nxysa[2:4] = np.log(sub_nxysa[2:4])
                
                scores, crop = entity_cnn.detect(img_fname, sub_ixywh)
                cnn_prob = scores[subj_cls_ix]

                have_nxysa = best_bboxes['object'].nxysa
                conditional_prob = sub_calibration(sub_conditional(sub_nxysa).pdf(have_nxysa))[0]
                total_score = ALPHA * cnn_prob + (1. - ALPHA) * conditional_prob
                
                if gt_bboxes is not None:
                    gt_iou = au.calc_iou_ixywh(sub_ixywh, gt_bboxes[0])
                else:
                    gt_iou = None
                
                viz.set_subject_supports(cnn_prob, conditional_prob, total_score, gt_iou=gt_iou)

                viz.set_subj_patch(crop)
                viz.set_subject_bbox(sub_ixywh)

                if best_bboxes['subject'] is None or total_score > best_bboxes['subject'].score:
                    best_bboxes['subject'] = BoxDescriptor(sub_ixywh, sub_nxysa, total_score, gt_iou)
                    viz.set_best_subj_patch(crop)
                    viz.set_best_subject_bbox(best_bboxes['subject'].ixywh)
                    viz.set_best_subject_score(best_bboxes['subject'].score)
                    viz.set_best_subject_iou(best_bboxes['subject'].gt_iou)

                
            if obj_ixywh is not None:
                #obj_nxysa = au.ixywh_to_nxysa(obj_ixywh[np.newaxis,:], img_w, img_h)
                #obj_nxysa = obj_nxysa[0]
                obj_nxysa[2:4] = np.log(obj_nxysa[2:4])
                
                scores, crop = entity_cnn.detect(img_fname, obj_ixywh)
                cnn_prob = scores[obj_cls_ix]

                have_nxysa = best_bboxes['subject'].nxysa
                conditional_prob = obj_calibration(obj_conditional(obj_nxysa).pdf(have_nxysa))[0]
                total_score = ALPHA * cnn_prob + (1. - ALPHA) * conditional_prob

                if gt_bboxes is not None:
                    gt_iou = au.calc_iou_ixywh(obj_ixywh, gt_bboxes[1])
                else:
                    gt_iou = None
                
                viz.set_object_supports(cnn_prob, conditional_prob, total_score, gt_iou=gt_iou)
                
                viz.set_obj_patch(crop)
                viz.set_object_bbox(obj_ixywh)

                if best_bboxes['object'] is None or total_score > best_bboxes['object'].score:
                    best_bboxes['object'] = BoxDescriptor(obj_ixywh, obj_nxysa, total_score, gt_iou)
                    viz.set_best_obj_patch(crop)
                    viz.set_best_object_bbox(best_bboxes['object'].ixywh)
                    viz.set_best_object_score(best_bboxes['object'].score)
                    viz.set_best_object_iou(best_bboxes['object'].gt_iou)

            # calculate situation score
            sit_score = pow((best_bboxes['subject'].score * best_bboxes['object'].score), 0.5)
            viz.set_situation_score(sit_score)

            # draw and (brief) pause for pyplot to actually do it
            viz.draw()
            time.sleep(sleeptime)

#===============================================================================
# Viz test using only priors
#
def viz_test_priors(viz, img_fnames, entity_cnn, sub_sampler, sub_conditional, sub_calibration, obj_sampler, obj_conditional, obj_calibration, subj_cls_ix, obj_cls_ix, sleeptime=0.0):
    for img_fname in img_fnames:
        viz.set_image(img_fname)
        img_w, img_h = viz.get_image_shape()
        viz.draw()

        sub_supports = np.zeros(3)
        obj_supports = np.zeros(3)

        best_bboxes = [None, None] #0: subj, 1: object
        
        for i in range(0,50):
            # sample sub & obj priors
            sbox = get_prior_sample(sub_sampler, img_w, img_h)#, box_name='sub')
            obox = get_prior_sample(obj_sampler, img_w, img_h)#, box_name='obj')
            print('sub: {:3d},{:3d},{:3d},{:3d} --- obj: {:3d},{:3d},{:3d},{:3d}'.format(sbox[0], sbox[1], sbox[2], sbox[3], obox[0], obox[1], obox[2], obox[3]))

            # draw and score
            viz.set_subject_bbox(sbox)
            scores, crop = entity_cnn.detect(img_fname, sbox)
            subj_score = scores[subj_cls_ix]
            viz.set_subj_patch(crop)
            
            viz.set_object_bbox(obox)
            scores, crop = entity_cnn.detect(img_fname, obox)
            obj_score = scores[obj_cls_ix]
            viz.set_obj_patch(crop)

            # update scores
            # calc subj external support
            
            # calc total support
            viz.set_subject_supports(subj_score, 0.0, 0.0)
            # calc obj external support
            # calc total support
            viz.set_object_supports(obj_score, 0.0, 0.0)
            
            viz.draw()
            time.sleep(sleeptime)

            # update best bboxes if possible
            if best_bboxes[0] is None or subj_score > best_bboxes[0][0]:
                best_bboxes[0] = (subj_score, sbox)
                viz.set_best_subject_bbox(sbox)
                viz.set_subject_best_score(subj_score)
            if best_bboxes[1] is None or obj_score > best_bboxes[1][0]:
                best_bboxes[1] = (obj_score, obox)
                viz.set_best_object_bbox(obox)
                viz.set_object_best_score(obj_score)

