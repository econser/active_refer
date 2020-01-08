import math
import numpy as np
import json as j

# Utility class for parsing VRD annotations
class RRQuery(object):
    def __init__(self, json_cfg, cls_dict, predicate_dict):
        # assume query string is: "<subject> <predicate> <object>"
        # subject and predicate are single tokens, predicate may have 0 or more whsp,
        q = json_cfg['query']
        q_tokens = q.split(' ')
        
        self.query_str = q
        
        self.subject_str = q_tokens[0]
        self.object_str = q_tokens[-1]
        self.predicate_str = q_tokens[1:-1]
        self.predicate_str = ' '.join(self.predicate_str)
        
        self.subject_ix = cls_dict.get(self.subject_str)
        self.object_ix = cls_dict.get(self.object_str)
        self.predicate_ix = predicate_dict.get(self.predicate_str)

# Intersection over Union calc
"""
calc_iou
expects bbox coords in xywh format
"""
def calc_iou_ixywh(bbox_a, bbox_b):
    # convert to x1y1x2y2
    xyxy_a = np.copy(bbox_a)
    xyxy_a[2] += xyxy_a[0]
    xyxy_a[3] += xyxy_a[1]
    
    xyxy_b = np.copy(bbox_b)
    xyxy_b[2] += xyxy_b[0]
    xyxy_b[3] += xyxy_b[1]
    
    # store area
    a_area = xyxy_a[2] * xyxy_a[3]
    b_area = xyxy_b[2] * xyxy_b[3]
    
    # find intersection
    inter_x1 = np.maximum(xyxy_a[0], xyxy_b[0])
    inter_y1 = np.maximum(xyxy_a[1], xyxy_b[1])
    inter_x2 = np.minimum(xyxy_a[2], xyxy_b[2])
    inter_y2 = np.minimum(xyxy_a[3], xyxy_b[3])
    
    inter_w = np.maximum(0.0, inter_x2 - inter_x1 + 1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h
    
    union = a_area + b_area - inter_area
    iou = inter_area / union
    
    return iou

# check if a bbox is within constraints
# expects bbox as x0y0x1y1
def valid_bbox_nxyxy(bbox, x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5):
    if bbox[0] < x_min: return False
    if bbox[1] < y_min: return False
    if bbox[2] > x_max: return False
    if bbox[3] > y_max: return False
    return True

# expects bbox as xywh
def valid_bbox_nxywh(bbox, x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5):
    if bbox[0] < x_min: return False
    if bbox[1] < y_min: return False
    if bbox[2] - bbox[0] > x_max: return False
    if bbox[3] - bbox[1] > y_max: return False
    return True

# expects bbox as [x, y, shape, area] (shape & area not log-space)
def valid_bbox_nxysa(bbox, x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5):
    if bbox[0] < x_min: return False
    if bbox[1] < y_min: return False
    
    w = math.sqrt(bbox[2] * bbox[3])
    if bbox[0] + w > x_max: return False
    
    h = bbox[3] / w
    if bbox[1] + h > y_max: return False
    
    return True

# bbox in : img  y0,y1,x0,x1
# bbox out: norm x0,x1,y0,y1
def iyyxx_to_nxxyy(bboxes, img_ws, img_hs):
    bbox_ws = bboxes[:,1] - bboxes[:,0]
    bbox_hs = bboxes[:,3] - bboxes[:,2]
    centers = np.array((bbox_ws, bbox_ws, bbox_hs, bbox_hs)).T / 2.
    shifts = np.array((img_ws, img_ws, img_hs, img_hs)).T / 2.
    scales = 1. / np.array((img_ws, img_ws, img_hs, img_hs)).T

    out = np.copy(bboxes)# + centers
    out[:,0] = bboxes[:,2]
    out[:,1] = bboxes[:,3]
    out[:,2] = bboxes[:,0]
    out[:,3] = bboxes[:,1]
    out -= shifts
    out *= scales
    return out

# bbox in : img  x, y, width, height
# bbox out: norm x, y, shape, area (shape & area are not log-space)
def ixywh_to_nxysa(bboxes, img_ws, img_hs):
    out = np.copy(bboxes).astype(np.float)
    # convert to xyxy
    out[:,2:4] += out[:,0:2]
    shifts = np.array((img_ws, img_hs, img_ws, img_hs)).T / 2.
    scales = 1. / np.array((img_ws, img_hs, img_ws, img_hs)).T
    out -= shifts
    out *= scales
    widths = np.abs(out[:,0] - out[:,2])
    heights = np.abs(out[:,3] - out[:,1])
    out[:,2] = widths / heights
    out[:,3] = widths * heights
    return out

# bbox in : img  x0, y0, x1, y1
# bbox out: norm x, y, shape, area (shape & area are not log-space)
def ixyxy_to_nxysa(bboxes, img_ws, img_hs):
    out = np.copy(bboxes).astype(np.float)
    # now xform from image space to norm space
    shifts = np.array((img_ws, img_hs, img_ws, img_hs)).T / 2.
    scales = 1. / np.array((img_ws, img_hs, img_ws, img_hs)).T
    out -= shifts
    out *= scales
    widths = np.abs(out[:,0] - out[:,2])
    heights = np.abs(out[:,3] - out[:,1])
    out[:,2] = widths / heights
    out[:,3] = widths * heights
    return out
    
# bbox in : norm x, y, shape, area (shape & area not log-space)
# bbox out: img  x, y, width, height
def nxysa_to_ixywh(bbox, img_w, img_h):
    x0 = bbox[0]
    y0 = bbox[1]
    aspect = bbox[2]
    area = bbox[3]
    
    width = math.sqrt(aspect * area)
    height = area / width
    
    # x0, y0, w, h ---> x0, y0, x1, y1
    x1 = x0 + width
    y1 = y0 + height
    normed_bbox = np.array([x0, y0, x1, y1])
    
    # scale
    scale_factor = np.array((img_w, img_h, img_w, img_h)).T
    normed_bbox *= scale_factor
    
    # shift
    normed_bbox += np.array((img_w, img_h, img_w, img_h)).T / 2.
    bbox_w = normed_bbox[2] - normed_bbox[0]
    bbox_h = normed_bbox[3] - normed_bbox[1]
    normed_bbox = np.ceil((normed_bbox[0], normed_bbox[1], bbox_w, bbox_h)).astype(np.int)

    return normed_bbox

def fix_cov(cov):
    n_size = cov.shape[0]
    cov = (cov + cov.T) * 0.5
    cov = cov + np.eye(n_size) * (np.min(np.diag(cov)) * 0.001)
    
    try:
        L = np.linalg.cholesky(cov)
        fixed = True
    except np.linalg.LinAlgError as e:
        cov = np.eye(n_size) * 0.001
        fixed = False

    return (cov, fixed)
