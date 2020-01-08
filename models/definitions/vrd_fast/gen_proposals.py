from __future__ import print_function
# generate proposals for the VRD dataset to be used for training a fast-rcnn model
#
# scipy.io.savemat on a dict like:
# {'boxes': array([[ array([[   0,    0,  352,  499],
#                           [   0,   43,  352,  499],
#                           [   0,    0,  352,  448],
#                           ..., 
#                           [ 229,   72,  246,   99],
#                           [ 217,  200,  231,  209],
#                           [   0,    0,   54,   41]], dtype=uint16),
#                            ...,
#                            array([<bbox_list>], dtype=uint16)]], dtype=object)
#   }
#
# selective search output is x0,y0,w,h
# fast rcnn expects ?,?,?,?
import sys
sys.path.append('/home/econser/research/utils/selectivesearch')

import os.path
import numpy as np
import skimage.data
import scipy.io
import json as j
import selectivesearch

def demo():
    img = skimage.data.astronaut()

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        #x, y, w, h = r['rect']
        #if w / h > 1.2 or h / w > 1.2:
        #    continue
        candidates.add(r['rect'])

    return candidates

#g.merge_mats('/home/econser/research/irsg_psu_pdx/data/VRD/annotations_train.json', '/home/econser/research/irsg_psu_pdx/data/VRD/ss_proposals/train', '/home/econser/research/irsg_psu_pdx/data/VRD/ss_proposals', 'train.mat')
def merge_mats(anno_file, src_dir, out_dir, out_fname):
    with open(anno_file, 'rb') as f:
        j_anno = j.load(f)
    
    anno_keys = j_anno.keys()
    all_boxes = []
    all_imgnames = []
    for k in anno_keys:
        k_matfile = '{}.mat'.format(k.split('.')[0])
        k_fullpath = os.path.join(src_dir, k_matfile)
        mat = scipy.io.loadmat(k_fullpath)['boxes']
        # TODO : do this part in the regular main
        mat = np.array(mat, dtype=np.uint16)
        mat[:,2] += mat[:,0]
        mat[:,3] += mat[:,1]
        # end
        all_boxes.append(mat)
        all_imgnames.append(k.split('.')[0])
    
    merge_dict = {}
    merge_dict['boxes'] = np.array(all_boxes)
    merge_dict['images'] = np.array(all_imgnames)
    
    out_fullpath = os.path.join(out_dir, out_fname)
    scipy.io.savemat(out_fullpath, merge_dict)



if __name__ == "__main__":
    # run training set
    with open('/home/econser/research/irsg_psu_pdx/data/VRD/train_image_metadata.json', 'rb') as f:
        j_imageset = j.load(f)
        
    imageset_dir = '/home/econser/research/irsg_psu_pdx/data/sg_dataset/sg_train_images'
    imageset_fnames = j_imageset.keys()
    n_images = len(imageset_fnames)
    
    outdir = '/home/econser/research/irsg_psu_pdx/data/VRD/ss_proposals/train'
    for ix, image_fname in enumerate(imageset_fnames):
        print('{:04d}/{:04d}: Processing {} --- '.format(ix+1, n_images, image_fname), end='')
        fq_fname = os.path.join(imageset_dir, image_fname)
        img = skimage.data.load(fq_fname)
        img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
        
        proposals = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in proposals:
                continue
            # excluding regions smaller than 2000 pixels
            if r['size'] < 2000:
                continue
            proposals.add(r['rect'])
        print('generated {} proposals'.format(len(proposals)))

        proposal_dict = {}
        proposal_dict['boxes'] = np.array(list(proposals), dtype=np.float) # TODO: find expected format and convert if necessary
        
        fbasename = image_fname.split('.')[0]
        out_fname = '{}.mat'.format(fbasename)
        fq_outfile = os.path.join(outdir, out_fname)
        scipy.io.savemat(fq_outfile, proposal_dict)
    
    # run test set
    with open('/home/econser/research/irsg_psu_pdx/data/VRD/test_image_metadata.json', 'rb') as f:
        j_imageset = j.load(f)
        
    imageset_dir = '/home/econser/research/irsg_psu_pdx/data/sg_dataset/sg_test_images'
    imageset_fnames = j_imageset.keys()
    n_images = len(imageset_fnames)
    
    outdir = '/home/econser/research/irsg_psu_pdx/data/VRD/ss_proposals/test'
    for ix, image_fname in enumerate(imageset_fnames):
        print('{:04d}/{:04d}: Processing {} --- '.format(ix+1, n_images, image_fname), end='')
        fq_fname = os.path.join(imageset_dir, image_fname)
        img = skimage.data.load(fq_fname)
        img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
        
        proposals = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in proposals:
                continue
            # excluding regions smaller than 2000 pixels
            if r['size'] < 2000:
                continue
            proposals.add(r['rect'])
        print('generated {} proposals'.format(len(proposals)))

        proposal_dict = {}
        proposal_dict['boxes'] = np.array(list(proposals), dtype=np.float) # TODO: find expected format and convert if necessary
        
        fbasename = image_fname.split('.')[0]
        out_fname = '{}.mat'.format(fbasename)
        fq_outfile = os.path.join(outdir, out_fname)
        scipy.io.savemat(fq_outfile, proposal_dict)
