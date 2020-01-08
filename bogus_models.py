import numpy as np

def softmax(mtx):
    temp = 1.0
    exp_mtx = np.exp(temp * mtx)
    exp_mtx /= np.sum(exp_mtx, axis=1, keepdims=True)
    return exp_mtx

    

def bogus_cnn(image, n_bboxes=300, n_scores=100):
    """ p(class | image)
    """
    MIN_BBOX_WIDTH = 10
    MIN_BBOX_HEIGHT = 10

    image_width = image.shape[1]
    image_height = image.shape[0]
    
    max_x0 = int(image_width * 0.9)
    max_y0 = int(image_height * 0.9)

    bboxes = np.zeros((n_bboxes, 4)) #x, y, w, h
    
    # generate fake bboxes
    bboxes[:, 0] = np.random.randint(1, max_x0, n_bboxes)
    bboxes[:, 1] = np.random.randint(1, max_y0, n_bboxes)
    
    for bbox in bboxes:
        available_width = image_width - bbox[0]
        bbox[2] = np.random.randint(MIN_BBOX_WIDTH, available_width)
        
        available_height = image_height - bbox[1]
        bbox[3] = np.random.randint(MIN_BBOX_HEIGHT, available_height)
        
    # generate fake scores
    scores = np.random.random((n_bboxes, n_scores))
    scores = softmax(scores)

    return bboxes, scores

def bogus_bbox_regression(bbox):
    """ perform regression on a bbox, given the initial bbox and the class
    """
    return bbox



def bogus_rel(bbox_pair):
    """ p(x0, xc, x2, y0, yc, y1, log(w), log(h), log(w/h), log(w*h))
    """
    return np.random.beta(1, 3)




def bogus_size_prior(relative_size):
    """ p(relative_bbox_sizes | dataset)
    """
    return np.random.beta(1, 3)

def bogus_shape_prior(relative_shape):
    """ p(relative_bbox_areas | dataset)
    """
    return np.random.beta(1, 3)

def bogus_position_prior(relative_size):
    """ p(relative_bbox_positions | dataset)
    """
    return np.random.beta(1, 3)



def bogus_internal_support(bbox, score):
    """ Internal support - how well does the bbox express the model
    """
    return score # TODO: just returning the score for now

def bogus_external_support(bbox):
    """ run this bbox through the conditioned joint
    """
    return np.random.beta(1, 3)

def bogus_total_support(internal, external):
    """ calculate total support from internal and external supports
    """
    return 0.5 * internal + 0.5 * external



def bogus_situation_score(total_support):
    """ calculate the situation score from interal support
    """
    return np.exp(total_support)
