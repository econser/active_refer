import sys
sys.path.append('/home/econser/usr/py-faster-rcnn/lib')
import os
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import caffe, cv2

import numpy as np
import json as j
import ar_utils as au
from scipy.stats import multivariate_normal as mvn_dist
from scipy.stats import norm as normal_dist
from sklearn.utils import shuffle
import sklearn.linear_model as lm

EPSILON = 1e-5

#===============================================================================
# A set of models useful in sampling and testing bounding boxes from a
#   subject-relationship-object expression.
#
# obj_detector - CNN for detecting the contents of a bounding box
# relationship - joint and conditional distributions for the relationship
# subject_priors - distributions for sampling bounding box locations, areas, and shapes
#
class ModelSet(object):
    def __init__(self, json_cfg):
        # read in the cfg params here
        classifier_def = json_cfg['object_classifier']['definition']
        classifier_def = classifier_def.encode('ascii', 'ignore')
        classifier_type = json_cfg['object_classifier']['type']
        classifier_weights = json_cfg['object_classifier']['weights']
        classifier_weights = classifier_weights.encode('ascii', 'ignore')
        class_labels = json_cfg['object_classifier']['class_labels']
        roi_def = json_cfg['roi_generator']['definition']
        roi_def = roi_def.encode('ascii', 'ignore')
        roi_weights = json_cfg['roi_generator']['weights']
        roi_weights = roi_weights.encode('ascii', 'ignore')

        self.roi_generator = FRCNN(roi_def, roi_weights, class_labels, gpu_mode=True)
        
        if classifier_type == 'RCNN':
            self.obj_detector = RCNN(classifier_def, classifier_weights, class_labels, gpu_mode=True) # IN: Image, bbox - OUT: classifications
        elif classifier_type == 'CNN':
            self.obj_detector = CNN(classifier_def, classifier_weights, class_labels, gpu_mode=True)
        self.obj_regressor = None # IN: bbox coords & class - OUT: bbox coords
        
        self.atr_detector = None # IN: image, bbox - OUT: classifications
        self.atr_regressor = None # IN: bbox coords - OUT: bbox coords

        # distributions for 
        self.relationship = Relationship()

        # these priors are used when no box is active
        self.subject_priors = Priors()
        self.object_priors = Priors()
        
        #self.external_support = Nonelinear scale factor definition geometry
        #self.total_support = None # IN: internal, external ; OUT: total
        #self.situation_score = None



#===============================================================================
#python detect.py --gpu --crop_mode 'list' '/home/econser/Pictures/bboxes.csv' '/home/econser/Pictures/out.csv'
#
# window list format is:
# [(<fq_fname>, array([[<bbox1>], [<bbox2>], ..., [<bboxN>]]))]
# [('/home/econser/Pictures/324885996_fbda2fc462_o.jpg', array([[ 191,  766,  436, 1044],[  25,  432,  960, 1148]]))]
class BaseCNN(object):
    def __init__(self, model_def, model_weights, class_labels, image_mean=None, gpu_mode=True):
        self.raw_scale = 255.0
        self.input_scale = None
        self.channel_swap = [2,1,0]
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.class_labels = {}
        if class_labels.endswith('.txt'):
            with open(class_labels, 'rb') as f:
                for line in f:
                    key, value = line.split(': ')
                    key = key.replace('{','')
                    
                    value = value.replace('}','')[1:-3]
                    value = list(value.split(', '))
                    
                    self.class_labels[int(key)] = value
        elif class_labels.endswith('.json'):
            with open(class_labels, 'rb') as f:
                jlabels = j.load(f)
            for ix, label in enumerate(jlabels):
                self.class_labels[ix+1] = label # ix 0 is background

    def detect(self, img, bbox):
        pass

class RCNN(BaseCNN):
    def __init__(self, model_def, model_weights, class_labels, image_mean=None, gpu_mode=True):
        super(RCNN, self).__init__(model_def, model_weights, class_labels, image_mean=image_mean, gpu_mode=gpu_mode)
        # TODO: send in optional parameters to caffe Detector
        context_pad = 5
        self.detector = caffe.Detector(model_def, model_weights, mean=image_mean, input_scale=self.input_scale, raw_scale=self.raw_scale, channel_swap=self.channel_swap, context_pad=context_pad)

    def detect(self, img, bbox):
        # using the caffe Detector is a bit silly and heavy-duty, detect_windows is intended for batch ops on several images, we don't need that here, but this tests the pipeline
        # TODO: streamline the detect_windows call so that we're not constantly opening the image and unpacking/packing lists and tuples.
        bbox_det = np.copy(bbox[np.newaxis,:])
        bbox_det[:,2:] += bbox_det[:,0:1]
        bbox_tup = (img, bbox_det)
        bbox_in = [bbox_tup]
        dets = self.detector.detect_windows(bbox_in)
        scores = dets[0]['prediction']
        scores = zip(self.class_labels, scores)
        scores = np.array(scores, dtype=np.object)
        return scores

class CNN(BaseCNN):
    def __init__(self, model_def, model_weights, class_labels, image_mean=None, gpu_mode=True):
        super(CNN, self).__init__(model_def, model_weights, class_labels, image_mean=image_mean, gpu_mode=gpu_mode)
        # TODO: send in optional parameters to caffe Detector
        self.image_dims = (256, 256)
        self.multi_crop = False
        self.classifier = caffe.Classifier(model_def, model_weights, image_dims = self.image_dims, mean = image_mean, input_scale = self.input_scale, raw_scale = self.raw_scale, channel_swap = self.channel_swap)

    def detect(self, img, bbox):
        img_data = cv2.imread(img)
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]
        crop = img_data[y:y+h,x:x+w]
        mdl_in = cv2.resize(crop, self.image_dims)
        b,g,r = cv2.split(crop)
        crop = cv2.merge([r,g,b])
        scores = self.classifier.predict([mdl_in], self.multi_crop)
        return scores[0], crop



#===============================================================================
# Region of Interest generators
#
# Perhaps generalize this class with different ROI generators.  For now, I just
#  need the one (Faster-RCNN) kind.
class FRCNN(object):
    def __init__(self, model_def, model_weights, class_labels, gpu_mode=True):
        cfg.TEST.HAS_RPN = True
        self.generator = caffe.Net(model_def, model_weights, caffe.TEST)

    def get_rois(self, img_fname, cls_ixs, nms_threshold=None):
        img = cv2.imread(img_fname)
        scores, bboxes = im_detect(self.generator, img)
        detections = {}
        for cls_ix in cls_ixs:
            cls_bboxes = bboxes[:, 4*cls_ix : 4*(cls_ix + 1)].astype(np.int)
            cls_scores = scores[:, cls_ix].astype(np.float)
            cls_dets = np.hstack((cls_bboxes, cls_scores[:,np.newaxis]))
            
            if nms_threshold is not None:
                detections[cls_ix] = nms(cls_dets, float(nms_threshold))
            else:
                detections[cls_ix] = cls_dets
            
        return detections



#===============================================================================
# Priors for sampling bbox location, area, and aspect ratio (shape)
#
class Priors(object):
    def __init__(self):
        self.area = None
        self.area_mu = None
        self.area_std = None
        
        self.shape = None
        self.shape_mu = None
        self.shape_std = None
        
        self.position = None
        self.xy_mean = None
        self.xy_cov = None

    def train(self, dataset):
        # dataset: x0, x1, y0, y1, im_w, im_h (x&y are normalized)

        # relative area (bbox to image)
        bbox_w = np.abs(dataset[:,1] - dataset[:,0])
        bbox_h = np.abs(dataset[:,3] - dataset[:,2])
        bbox_area = np.prod((bbox_w, bbox_h), axis=0)
        rel_areas = bbox_area # x&y are normalized, so img area = 1.0
        rel_areas = np.log(rel_areas)
        self.area_mu = np.mean(rel_areas)
        self.area_std = np.std(rel_areas)
        self.area = normal_dist(loc=self.area_mu, scale=self.area_std)
        
        # shape prior (aspect ratio)
        aspect_ratios = bbox_w / bbox_h
        aspect_ratios = np.log(aspect_ratios)
        self.shape_mu = np.mean(aspect_ratios)
        self.shape_std = np.std(aspect_ratios)
        self.shape = normal_dist(loc=self.shape_mu, scale=self.shape_std)

        # position_prior (normalized xy)
        xy_locs = dataset[:,[True,False,True,False,False,False]]
        self.xy_mu = np.mean(xy_locs, axis=0)
        self.xy_cov = np.cov(xy_locs.T)
        # ensure that the covar mtx is PSD
        if np.linalg.cond(self.xy_cov) > 1./sys.float_info.epsilon:
            self.xy_cov, fixed = au.fix_cov(self.xy_cov)
            
        try:
            L = np.linalg.cholesky(self.xy_cov)
        except np.linalg.LinAlgError as e:
            self.xy_cov, fixed = au.fix_cov(self.xy_cov)
        
        self.position = mvn_dist(mean=self.xy_mu, cov=self.xy_cov)

    def sample(self, size=1):
        areas = np.exp(self.area.rvs(size))
        shapes = np.exp(self.shape.rvs(size))
        positions = self.position.rvs(size)
        
        if size > 1:
            return np.hstack((positions, shapes[:,np.newaxis], areas[:,np.newaxis]))
        else:
            return np.hstack((positions, shapes, areas))



#===============================================================================
# Joint and Conditional model for a subject-relationship-object configuration
#
class Relationship(object):
    def __init__(self):
        self.joint_relationship = None
        self.object_conditional = None
        self.object_calibration = None
        self.subject_conditional = None
        self.subject_calibration = None

    def train(self, dataset):
        n_rows = len(dataset)
        # each training sample is:
        #     0,   1,   2,   3,     4,     5,   6,   7,   8,   9,    10,    11
        #   sx0, sx1, sy0, sy1, img_w, img_h, ox0, ox1, oy0, oy1, img_w, img_h (x&y normalized)
        # convert to x0_s, y0_s, log_aspect_s, log_area_s, x0_o, y0_o, log_aspect_o, log_area_o
        sub_x0y0s = np.hstack((dataset[:,0,None], dataset[:,2,None]))
        
        sub_widths = dataset[:,1,None] - dataset[:,0, None]
        sub_widths = np.abs(sub_widths)
        
        sub_heights = dataset[:,3,None] - dataset[:,2,None]
        sub_heights = np.abs(sub_heights)

        sub_areas = sub_widths * sub_heights
        sub_areas = np.log(sub_areas)
        
        sub_aspects = sub_widths / sub_heights
        sub_aspects = np.log(sub_aspects)
        
        # convert object data
        obj_x0y0s = np.hstack((dataset[:,6,None], dataset[:,8,None]))
        
        obj_widths = dataset[:,7,None] - dataset[:,6,None]
        obj_widths = np.abs(obj_widths)

        obj_heights = dataset[:,9,None] - dataset[:,8,None]
        obj_heights = np.abs(obj_heights)
        
        obj_areas = obj_widths * obj_heights
        obj_areas = np.log(obj_areas)
        
        obj_aspects = obj_widths / obj_heights
        obj_aspects = np.log(obj_aspects)

        varset = (sub_x0y0s, sub_aspects, sub_areas, obj_x0y0s, obj_aspects, obj_areas)
        varset = np.hstack(varset)
        
        # get joint mean and covar mtx
        mean = np.average(varset, axis=0)
        cov = np.cov(varset.T)
        #cov = np.cov(varset, rowvar=False)
        # check that the covar mtx is positive definite
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as e:
            cov, fixed = au.fix_cov(cov)

        # generate the joint MVN
        self.joint_relationship = mvn_dist(mean=mean, cov=cov)
        
        # generate the conditionals
        # find mu_sub & mu_obj
        # split sigma into 4 parts:
        #            | S_ss  S_os |
        #  S_joint = |            |
        #            | S_so  S_oo |
        #
        # [subject conditional] --- P(s|o=b):
        # mu_s|o = mu_sub + S_so * S_oo^-1 * (b - mu_obj)
        # S_s|o  = S_ss - S_so * S_oo^-1 * S_os
        #
        # [object conditional] --- P(o|s=b):
        # mu_o|s = mu_obj + S_os * S_ss^-1 * (b - mu_sub)
        # S_o|s  = S_oo - S_os * S_ss^-1 * S_so

        #-----------------------------------------------------------------------
        # generate conditional distributions
        sub_mask = np.array([True,  True,  True,  True,  False, False, False, False])
        sxy_mask = np.array([True,  True,  False, False, False, False, False, False])
        ss_mask  = np.array([False, False, True,  False, False, False, False, False])
        sa_mask  = np.array([False, False, False,  True, False, False, False, False])
        
        obj_mask = np.array([False, False, False, False, True,  True,  True,  True ])
        oxy_mask = np.array([False, False, False, False, True,  True,  False, False])
        os_mask  = np.array([False, False, False, False, False, False, True,  False])
        oa_mask  = np.array([False, False, False, False, False, False, False, True ])

        self.subject_conditional = self._condition_mvn(varset, sub_mask, obj_mask) # p(sub|obj)
        self.subject_calibration, self.subject_cal_params = self._calibrate(self.subject_conditional(varset[:,obj_mask][0]))
        self.sxy_conditional = self._condition_mvn(varset, sxy_mask, obj_mask)
        self.ss_conditional  = self._condition_mvn(varset, ss_mask, obj_mask)
        self.sa_conditional = self._condition_mvn(varset, sa_mask, obj_mask)
        
        self.object_conditional = self._condition_mvn(varset, obj_mask, sub_mask) # p(obj|sub)
        self.object_calibration, self.object_cal_params = self._calibrate(self.object_conditional(varset[:,sub_mask][0]))
        self.oxy_conditional = self._condition_mvn(varset, oxy_mask, sub_mask)
        self.os_conditional  = self._condition_mvn(varset, os_mask, sub_mask)
        self.oa_conditional = self._condition_mvn(varset, oa_mask,sub_mask)

        # verify that the calibrated probabilities for the original data look reasonable
        #import diagnostics as diag
        #diag.plot_scaling_performance(varset, sub_mask, obj_mask, self.subject_conditional, self.subject_calibration, self.subject_cal_params)
        #diag.plot_scaling_performance(varset, obj_mask, sub_mask, self.object_conditional, self.object_calibration, self.object_cal_params)
        #import pdb;pdb.set_trace();pass

    def _condition_mvn(self, dataset, want_mask, have_mask):
        varset = np.hstack((dataset[:,want_mask], dataset[:,have_mask]))
        n_wants = np.sum(want_mask)
        n_haves = np.sum(have_mask)
        
        cov = np.cov(varset.T)
        if np.linalg.cond(cov) > 1./sys.float_info.epsilon:
            cov, fixed = au.fix_cov(cov)
        cov_ss = cov[0:n_wants, 0:n_wants]
        cov_os = cov[n_wants:,  0:n_wants]
        cov_so = cov[0:n_wants, n_wants:]
        cov_oo = cov[n_wants:,  n_wants:]
        
        inv_cov = np.linalg.inv(cov)
        inv_cov_oo = np.linalg.inv(cov_oo)
        inv_cov_ss = np.linalg.inv(cov_ss)
        
        mu_sub = np.average(dataset[:, want_mask], axis=0).T
        mu_obj = np.average(dataset[:, have_mask], axis=0).T
        
        mu_cond = lambda b : mu_sub + cov_so.dot(inv_cov_oo).dot((b - mu_obj).T)
        cov_cond = cov_ss - cov_so.dot(inv_cov_oo).dot(cov_os)
        # ensure that the covar mtx is PSD
        try:
            L = np.linalg.cholesky(cov_cond)
        except np.linalg.LinAlgError as e:
            cov_cond, fixed = au.fix_cov(cov_cond)
        # p(sub | obj)
        conditional = lambda bbox : mvn_dist(mean=mu_cond(bbox), cov=cov_cond)
        print('==========\nmu:\n{}'.format(mu_sub))
        print('==========\ncov:\n{}'.format(cov_cond))
        return conditional

    def _calibrate(self, mvn):
        n_samples = 1000000
        samples = mvn.rvs(n_samples)
        n_setsize = int(n_samples / 1000.)
        
        densities = mvn.pdf(samples)[:,np.newaxis]
        cal_data = np.hstack((samples, densities))
        cal_data = cal_data[cal_data[:,4].argsort()][::-1]
        
        # partition the top-n and bottom-n
        pos_vals = cal_data[:n_setsize, 4][:,np.newaxis]
        pos_labels = np.ones(n_setsize)[:,np.newaxis]
        #pos_labels *= ((n_setsize + 1.) / (n_setsize + 2.))
        
        neg_vals = cal_data[-n_setsize:, 4][:,np.newaxis]
        neg_labels = np.zeros(n_setsize)[:,np.newaxis]
        #neg_labels *= (1. / (n_setsize + 2.))
        
        # build the dataset
        pos_set = np.hstack((pos_vals, pos_labels))
        neg_set = np.hstack((neg_vals, neg_labels))
        cal_set = np.vstack((pos_set, neg_set))
        vals, labels = shuffle(cal_set[:,0], cal_set[:,1])

        # train the logistic regression
        cal_model = lm.LogisticRegression(penalty='l1')
        cal_model.fit(vals[:,np.newaxis], labels)

        # here we return the function to calculate the probability that the argument would yield a '1' label: high density having a 1 label and low density having a 0 label
        cal_fn = lambda d:cal_model.predict_proba(d)[:,1]
        cal_params = (cal_model.intercept_[0], cal_model.coef_[0][0])
        return cal_fn, cal_params
        
    def sample_joint(self, n_samples):
        return self.joint_relationship.rvs(n_samples) # TODO: convert to sub_xyxy, obj_xyxy?

    def joint_pdf(self, subject_bbox, object_bbox):
        # TODO : convert to varset format
        return self.join_conditional.pdf(varset)

    def sample_object_conditional(self, subject_bbox, n_samples):
        return self.object_conditional(bbox=subject_bbox).rvs(n_samples)

    def object_conditional_density(self, object_bbox, subject_bbox):
        return self.object_conditional(bbox=subject_bbox).pdf(object_bbox)

    def sample_subject_conditional(self, object_bbox, n_samples):
        return self.subject_conditional(bbox=object_bbox).rvs(n_samples)

    def subject_conditional_density(self, subject_bbox, object_bbox):
        return self.subject_conditional(bbox=object_bbox).pdf(subject_bbox)
