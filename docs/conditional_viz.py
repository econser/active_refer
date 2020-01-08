import cv2
import numpy as np
import matplotlib.image as image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import utils as u

N_SAMPLES = 300
DENSITY_THRESHOLD = 1e-10

class ConditionalViz(object):
    def __init__(self, subject_sampler, subject_conditional_density, object_sampler, object_conditional_density, subject_str, predicate_str, object_str, image_fname):
        self.subject_sampler = subject_sampler
        self.subject_conditional_density = subject_conditional_density
        self.object_sampler = object_sampler
        self.object_conditional_density = object_conditional_density

        self.subject_str = subject_str
        self.object_str = object_str
        self.predicate_str = predicate_str

        figsize = plt.rcParams["figure.figsize"]
        figsize[0] = 14
        plt.rcParams["figure.figsize"] = figsize
        self.fig = plt.figure(0)

        # load the image for future use
        img = image.imread(image_fname)
        self.img_w = img.shape[1]
        self.img_h = img.shape[0]

        # set up subplots
        self.sub_density_ax = plt.subplot2grid((1,3), (0,0))
        self.image_ax = plt.subplot2grid((1,3), (0,1))
        self.obj_density_ax = plt.subplot2grid((1,3), (0,2))
        
        # p(subj_bbox|obj_bbox)
        self.sub_density_vals = np.zeros((N_SAMPLES, N_SAMPLES))#np.random.random((N_SAMPLES, N_SAMPLES))
        self.sub_density_img = cv2.resize(self.sub_density_vals, (self.img_w, self.img_h))
        self.sub_density_ax.imshow(self.sub_density_img, cmap='gray')
        self.sub_density_ax.title.set_text('p({} | {})'.format(subject_str, object_str))
        self.sub_density_ax.title.set_color('b')

        # the image and subject/object bboxes
        self.image_ax.imshow(img)
        self.image_ax.axis('off')

        self.sub_bbox_patch = patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='r', facecolor='none')
        self.image_ax.add_patch(self.sub_bbox_patch)
        
        self.obj_bbox_patch = patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='b', facecolor='none')
        self.image_ax.add_patch(self.obj_bbox_patch)

        # p(obj_bbox|subj_bbox)
        self.obj_density_vals = np.random.random((N_SAMPLES, N_SAMPLES))
        self.obj_density_img = cv2.resize(self.obj_density_vals, (self.img_w, self.img_h))
        self.obj_density_ax.imshow(self.obj_density_img)
        self.obj_density_ax.title.set_text('p({} | {})'.format(object_str, subject_str))
        self.obj_density_ax.title.set_color('r')

        # start drawing
        plt.tight_layout()
        plt.ion()
        plt.show()

    def draw(self):
        plt.draw()
        plt.pause(0.001)

    def step(self):
        self._conditional_step(self.obj_density_img, self.subject_sampler, self.sub_bbox_patch, self.object_sampler, self.object_conditional_density, self.obj_density_ax, n_samples=1000)
        self._conditional_step(self.sub_density_img, self.object_sampler, self.obj_bbox_patch, self.subject_sampler, self.subject_conditional_density, self.sub_density_ax, n_samples=1000)
        self.draw()

    def _conditional_step(self, density_img, cond_sampler, cond_patch, model_sampler, conditioned_mvn, density_ax, n_samples=5000):
        density_img *= 0.
        # sample a subject bbox
        sub_bbox_coords_norm = cond_sampler.sample()
        while not u.valid_bbox_aa(sub_bbox_coords_norm):
            sub_bbox_coords_norm = cond_sampler.sample()
        sub_bbox_coords_img = u.norm_to_img(sub_bbox_coords_norm, self.img_w, self.img_h)
        self._update_bbox_patch(sub_bbox_coords_img, cond_patch)
        
        # generate p(obj|sub) densities
        # sample several object bboxes
        obj_densities = np.array([])
        obj_samples_norm_aa = np.array([])
        while len(obj_samples_norm_aa) < n_samples:
            samples = model_sampler.sample(n_samples)
            densities = conditioned_mvn(samples, sub_bbox_coords_norm)
            samples_img = u.norm_to_img_v(samples, self.img_w, self.img_h)
            samples[:,0] = np.clip(samples_img[:,0], 0, self.img_w)
            samples[:,1] = np.clip(samples_img[:,1], 0, self.img_h)
            samples[:,2] = np.clip(samples_img[:,0] + samples_img[:,2], 0, self.img_w)
            samples[:,2] -= samples[:,0]
            samples[:,3] = np.clip(samples_img[:,1] + samples_img[:,3], 0, self.img_h)
            samples[:,3] -= samples[:,1]
            
            # discard bad bboxes
            nonzeros = np.where(densities > DENSITY_THRESHOLD)[0]
            samples = samples[nonzeros]
            densities = densities[nonzeros]
            
            in_bounds = np.where(samples[:,2] > 0)
            samples = samples[in_bounds]
            densities = densities[in_bounds]
            
            in_bounds = np.where(samples[:,3] > 0)
            samples = samples[in_bounds]
            densities = densities[in_bounds]
            
            obj_densities = np.hstack((obj_densities, densities))
            if len(obj_samples_norm_aa) == 0:
                obj_samples_norm_aa = np.array(samples)
            else:
                obj_samples_norm_aa = np.vstack((obj_samples_norm_aa, samples))

        obj_samples_norm_aa = np.array(obj_samples_norm_aa[:n_samples], dtype=np.int)
        obj_densities = obj_densities[:n_samples]
        
        # update stored density plot like:
        #  stored_density[patch] = max(stored density, new density)
        for ix, bbox in enumerate(obj_samples_norm_aa):
            x0 = bbox[0]
            x1 = x0 + bbox[2]
            y0 = bbox[1]
            y1 = y0 + bbox[3]
            density_img[y0:y1, x0:x1] = np.maximum(obj_densities[ix], density_img[y0:y1, x0:x1])
        density_img = gaussian_filter(density_img, sigma=10)
        density_ax.imshow(density_img, cmap='gray')
        
    def _conditional_step___(self, density_img, cond_sampler, cond_patch, model_sampler, conditioned_mvn, density_ax, n_samples=5000):
        density_img *= 0.
        # sample a subject bbox
        sub_bbox_coords_norm = cond_sampler.sample()
        while not u.valid_bbox_aa(sub_bbox_coords_norm):
            sub_bbox_coords_norm = cond_sampler.sample()
        sub_bbox_coords_img = u.norm_to_img(sub_bbox_coords_norm, self.img_w, self.img_h)
        self._update_bbox_patch(sub_bbox_coords_img, cond_patch)
        
        # generate p(obj|sub) densities
        # sample several object bboxes
        obj_samples_norm_aa = model_sampler.sample(n_samples)
        density_obj_cond_sub = conditioned_mvn(obj_samples_norm_aa, sub_bbox_coords_norm)
        obj_samples_img_wh = u.norm_to_img_v(obj_samples_norm_aa, self.img_w, self.img_h)
        # clip img-space bboxes to image bounds
        clipped_obj_bboxes = np.copy(obj_samples_img_wh)
        clipped_obj_bboxes[:,0] = np.clip(obj_samples_img_wh[:,0], 0, self.img_w)
        clipped_obj_bboxes[:,1] = np.clip(obj_samples_img_wh[:,1], 0, self.img_h)
        clipped_obj_bboxes[:,2] = np.clip(obj_samples_img_wh[:,0] + obj_samples_img_wh[:,2], 0, self.img_w)
        clipped_obj_bboxes[:,2] -= clipped_obj_bboxes[:,0]
        clipped_obj_bboxes[:,3] = np.clip(obj_samples_img_wh[:,1] + obj_samples_img_wh[:,3], 0, self.img_h)
        clipped_obj_bboxes[:,3] -= clipped_obj_bboxes[:,1]
        # discard bad bboxes
        keep_mask = np.ones(len(clipped_obj_bboxes), dtype=np.bool)

        for ix, bbox in enumerate(clipped_obj_bboxes):
            if bbox[2] <= 0 or bbox[3] <= 0 or density_obj_cond_sub[ix] < DENSITY_THRESHOLD:
                keep_mask[ix] = False

        valid_bboxes = clipped_obj_bboxes[keep_mask]
        print('{} valid bboxes'.format(len(valid_bboxes)))
        # update stored density plot like:
        #  stored_density[patch] = max(stored density, new density)
        self.obj_density_vals = np.zeros((N_SAMPLES, N_SAMPLES))
        for ix, bbox in enumerate(valid_bboxes):
            x0 = bbox[0]
            x1 = x0 + bbox[2]
            y0 = bbox[1]
            y1 = y0 + bbox[3]
            density_img[y0:y1, x0:x1] = np.maximum(density_obj_cond_sub[ix], density_img[y0:y1, x0:x1])
        density_img = gaussian_filter(density_img, sigma=10)
        density_ax.imshow(density_img, cmap='gray')
    
    def _update_bbox_patch(self, bbox, patch):
        patch.set_x(bbox[0])     
        patch.set_y(bbox[1])     
        patch.set_width(bbox[2]) 
        patch.set_height(bbox[3])
