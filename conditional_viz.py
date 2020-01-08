import cv2
import numpy as np
import matplotlib.image as image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import ar_utils as u

N_SAMPLES = 300
DENSITY_THRESHOLD = 1e-10

class ConditionalViz(object):
    def __init__(self, subject_sampler, object_sampler, conditionals, subject_str, predicate_str, object_str, image_fname):
        self.subject_sampler = subject_sampler
        self.subject_conditional_density = conditionals.subject_conditional_density
        self.sxy_conditional = conditionals.sxy_conditional
        self.s_shape_conditional = conditionals.ss_conditional
        self.s_area_conditional = conditionals.sa_conditional
        
        self.object_sampler = object_sampler
        self.object_conditional_density = conditionals.object_conditional_density
        self.oxy_conditional = conditionals.oxy_conditional
        self.o_shape_conditional = conditionals.os_conditional
        self.o_area_conditional = conditionals.oa_conditional

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

        sub_mu = np.array([self.subject_sampler.xy_mu[0], self.subject_sampler.xy_mu[1], np.exp(self.subject_sampler.shape_mu), np.exp(self.subject_sampler.area_mu)])
        sub_xywh = u.nxysa_to_ixywh(sub_mu, self.img_w, self.img_h)
        sub_debug = '{} mean: [{: 0.2f}, {: 0.2f}, {: 0.2f}, {: 0.2f}] [{:>4d}, {:>4d}, {:>4d}, {:>4d}]'.format(self.subject_str, sub_mu[0], sub_mu[1], sub_mu[2], sub_mu[3], sub_xywh[0], sub_xywh[1], sub_xywh[2], sub_xywh[3])
        obj_mu = np.array([self.object_sampler.xy_mu[0], self.object_sampler.xy_mu[1], np.exp(self.object_sampler.shape_mu), np.exp(self.object_sampler.area_mu)])
        obj_xywh = u.nxysa_to_ixywh(obj_mu, self.img_w, self.img_h)
        obj_debug = '{} mean: [{: 0.2f}, {: 0.2f}, {: 0.2f}, {: 0.2f}] [{:>4d}, {:>4d}, {:>4d}, {:>4d}]'.format(self.object_str, obj_mu[0], obj_mu[1], obj_mu[2], obj_mu[3], obj_xywh[0], obj_xywh[1], obj_xywh[2], obj_xywh[3])
        print('{} --- {}'.format(sub_debug, obj_debug))
        
        # set up subplots
        self.s_p_xy_ax    = plt.subplot2grid((4,8), (0,0))
        self.s_p_shape_ax = plt.subplot2grid((4,8), (1,0))
        self.s_p_area_ax  = plt.subplot2grid((4,8), (2,0))
        self.empty1_ax    = plt.subplot2grid((4,8), (3,0))
        self.s_c_xy_ax    = plt.subplot2grid((4,8), (0,1))
        self.s_c_shape_ax = plt.subplot2grid((4,8), (1,1))
        self.s_c_area_ax  = plt.subplot2grid((4,8), (2,1))
        self.s_density_ax = plt.subplot2grid((4,8), (3,1))
        self.image_ax     = plt.subplot2grid((4,8), (0,2), colspan=4, rowspan=4)
        self.o_p_xy_ax    = plt.subplot2grid((4,8), (0,6))
        self.o_p_shape_ax = plt.subplot2grid((4,8), (1,6))
        self.o_p_area_ax  = plt.subplot2grid((4,8), (2,6))
        self.o_density_ax = plt.subplot2grid((4,8), (3,6))
        self.o_c_xy_ax    = plt.subplot2grid((4,8), (0,7))
        self.o_c_shape_ax = plt.subplot2grid((4,8), (1,7))
        self.o_c_area_ax  = plt.subplot2grid((4,8), (2,7))
        self.empty2_ax    = plt.subplot2grid((4,8), (3,7))

        # empty cells
        self.empty1_ax.axis('off')
        self.empty2_ax.axis('off')
        
        # subject priors
        self.s_p_xy_ax.set_title('{} prior - xy'.format(subject_str), fontdict={'fontsize':10})
        x_pos = np.linspace(-0.5, 0.5, self.img_w)
        y_pos = np.linspace(-0.5, 0.5, self.img_h)
        X, Y = np.meshgrid(x_pos, y_pos)
        XY = np.dstack((X,Y))
        Z_s = self.subject_sampler.position.pdf(XY)
        Z_s = cv2.resize(Z_s, (self.img_w, self.img_h))
        self.s_p_xy_ax.imshow(Z_s, cmap='bone')
        
        self.s_p_shape_ax.set_title('{} prior - shape'.format(subject_str), fontdict={'fontsize':10})
        xs0 = self.subject_sampler.shape_mu - 2. * self.subject_sampler.shape_std
        xs1 = self.subject_sampler.shape_mu + 2. * self.subject_sampler.shape_std
        self.xs_shape = np.linspace(np.exp(xs0), np.exp(xs1), 100)
        ys_shape = self.subject_sampler.shape.pdf(np.log(self.xs_shape))
        self.s_p_shape_ax.plot(self.xs_shape, np.exp(ys_shape))

        self.s_p_area_ax.set_title('{} prior - area'.format(subject_str), fontdict={'fontsize':10})
        xa0 = self.subject_sampler.area_mu - 2. * self.subject_sampler.area_std
        xa1 = self.subject_sampler.area_mu + 2. * self.subject_sampler.area_std
        self.xs_area = np.linspace(xa0, xa1, 100)
        ys_area = self.subject_sampler.area.pdf(self.xs_area)
        self.s_p_area_ax.plot(self.xs_area, ys_area)

        # subject conditionals
        self.s_c_xy_ax.set_title('{} conditional - xy'.format(subject_str), fontdict={'fontsize':10})
        
        self.s_c_shape_ax.set_title('{} conditional - shape'.format(subject_str), fontdict={'fontsize':10})
        self.s_c_shape_plot, = self.s_c_shape_ax.plot(self.xs_shape, np.linspace(0, 1, 100))
        self.s_c_shape_ax.set_xlim(self.s_p_shape_ax.get_xlim())
        self.s_c_shape_ax.set_ylim(self.s_p_shape_ax.get_ylim())
        
        self.s_c_area_ax.set_title('{} conditional - area'.format(subject_str), fontdict={'fontsize':10})
        self.s_c_area_plot, = self.s_c_area_ax.plot(self.xs_area, np.linspace(0,1,100))
        self.s_c_area_ax.set_xlim(self.s_p_area_ax.get_xlim())
        self.s_c_area_ax.set_ylim(self.s_p_area_ax.get_ylim())
        
        # p(subj_bbox|obj_bbox)
        self.sub_density_vals = np.zeros((N_SAMPLES, N_SAMPLES))
        self.sub_density_img = cv2.resize(self.sub_density_vals, (self.img_w, self.img_h))
        self.s_density_ax.imshow(self.sub_density_img, cmap='bone')
        self.s_density_ax.title.set_text('p({} | {})'.format(subject_str, object_str))
        self.s_density_ax.title.set_color('b')

        # the image and subject/object bboxes
        self.image_ax.imshow(img)
        self.image_ax.axis('off')

        self.sub_bbox_norm = np.array([0.,0.,1.,1.])
        self.sub_bbox_patch = patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='r', facecolor='none')
        self.image_ax.add_patch(self.sub_bbox_patch)
        
        self.obj_bbox_norm = np.array([0.,0.,1.,1.])
        self.obj_bbox_patch = patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='b', facecolor='none')
        self.image_ax.add_patch(self.obj_bbox_patch)

        # object priors
        self.o_p_xy_ax.set_title('{} prior - xy'.format(object_str), fontdict={'fontsize':10})
        Z_o = self.object_sampler.position.pdf(XY)
        Z_o = cv2.resize(Z_o, (self.img_w, self.img_h))
        self.o_p_xy_ax.imshow(Z_o, cmap='bone')
        
        self.o_p_shape_ax.set_title('{} prior - shape'.format(object_str), fontdict={'fontsize':10})
        o_xs0 = self.object_sampler.shape_mu - 2. * self.object_sampler.shape_std
        o_xs1 = self.object_sampler.shape_mu + 2. * self.object_sampler.shape_std
        self.xo_shape = np.linspace(np.exp(o_xs0), np.exp(o_xs1), 100)
        yo_shape = self.object_sampler.shape.pdf(np.log(self.xo_shape))
        self.o_p_shape_ax.plot(self.xo_shape, np.exp(yo_shape))
        
        self.o_p_area_ax.set_title('{} prior - area'.format(object_str), fontdict={'fontsize':10})
        o_xa0 = self.object_sampler.area_mu - 2. * self.object_sampler.area_std
        o_xa1 = self.object_sampler.area_mu + 2. * self.object_sampler.area_std
        self.xo_area = np.linspace(np.exp(o_xa0), np.exp(o_xa1), 100)
        yo_area = self.object_sampler.area.pdf(self.xo_area)
        self.o_p_area_ax.plot(self.xo_area, yo_area)

        # object conditionals
        self.o_c_xy_ax.set_title('{} conditional - xy'.format(object_str), fontdict={'fontsize':10})
        
        self.o_c_shape_ax.set_title('{} conditional - shape'.format(object_str), fontdict={'fontsize':10})
        self.o_c_shape_plot, = self.o_c_shape_ax.plot(self.xo_shape, np.linspace(0,1,100))
        self.o_c_shape_ax.set_xlim(self.o_p_shape_ax.get_xlim())
        self.o_c_shape_ax.set_ylim(self.o_p_shape_ax.get_ylim())

        self.o_c_area_ax.set_title('{} conditional - area'.format(object_str), fontdict={'fontsize':10})
        self.o_c_area_plot, = self.o_c_area_ax.plot(self.xo_area, np.linspace(0,1,100))
        self.o_c_area_ax.set_xlim(self.o_p_area_ax.get_xlim())
        self.o_c_area_ax.set_ylim(self.o_p_area_ax.get_ylim())

        # p(obj_bbox|subj_bbox)
        self.obj_density_vals = np.zeros((N_SAMPLES, N_SAMPLES))
        self.obj_density_img = cv2.resize(self.obj_density_vals, (self.img_w, self.img_h))
        self.o_density_ax.imshow(self.obj_density_img, cmap='bone')
        self.o_density_ax.title.set_text('p({} | {})'.format(object_str, subject_str))
        self.o_density_ax.title.set_color('r')

        # start drawing
        plt.tight_layout()
        plt.ion()
        plt.show()
        plt.draw()

    def draw(self):
        plt.draw()
        plt.pause(0.001)

    def step(self):
        self._conditional_step(self.obj_density_img, self.subject_sampler, self.sub_bbox_patch, self.sub_bbox_norm, self.object_sampler, self.object_conditional_density, self.o_density_ax, n_samples=20)
        # test out denstiy plotting for subject xy conditional -----------------
        x = np.linspace(-0.5, 0.5, self.img_w)
        y = np.linspace(-0.5, 0.5, self.img_h)
        X, Y = np.meshgrid(x, y)
        XY = np.dstack((X,Y))
        Zs = self.sxy_conditional(self.sub_bbox_norm).pdf(XY)
        #Zs = cv2.resize(Zs, (self.img, self.img_h))
        self.s_c_xy_ax.imshow(Zs, cmap='bone')
        
        y_shape = self.s_shape_conditional(self.sub_bbox_norm).pdf(np.log(self.xs_shape))
        self.s_c_shape_plot.set_ydata(y_shape)

        y_area = self.s_area_conditional(self.sub_bbox_norm).pdf(self.xs_area)
        self.s_c_area_plot.set_ydata(y_area)
        #-----------------------------------------------------------------------
        self._conditional_step(self.sub_density_img, self.object_sampler, self.obj_bbox_patch, self.obj_bbox_norm, self.subject_sampler, self.subject_conditional_density, self.s_density_ax, n_samples=20)
        Zo = self.oxy_conditional(self.obj_bbox_norm).pdf(XY)
        #Zo = cv2.resize(Zo, (self.img_w, self.img_h))
        self.o_c_xy_ax.imshow(Zo, cmap='bone')
        
        y_shape = self.o_shape_conditional(self.obj_bbox_norm).pdf(np.log(self.xo_shape))
        self.o_c_shape_plot.set_ydata(y_shape)
        
        y_area = self.o_area_conditional(self.obj_bbox_norm).pdf(self.xo_area)
        self.o_c_area_plot.set_ydata(y_area)

        sub_xywh = u.nxysa_to_ixywh(self.sub_bbox_norm, self.img_w, self.img_h)
        sub_debug = '{} cond: [{: 0.2f}, {: 0.2f}, {: 0.2f}, {: 0.2f}] [{:>4d}, {:>4d}, {:>4d}, {:>4d}]'.format(self.subject_str, self.sub_bbox_norm[0], self.sub_bbox_norm[1], self.sub_bbox_norm[2], self.sub_bbox_norm[3], sub_xywh[0], sub_xywh[1], sub_xywh[2], sub_xywh[3])
        obj_xywh = u.nxysa_to_ixywh(self.obj_bbox_norm, self.img_w, self.img_h)
        obj_debug = '{} cond: [{: 0.2f}, {: 0.2f}, {: 0.2f}, {: 0.2f}] [{:>4d}, {:>4d}, {:>4d}, {:>4d}]'.format(self.object_str, self.obj_bbox_norm[0], self.obj_bbox_norm[1], self.obj_bbox_norm[2], self.obj_bbox_norm[3], obj_xywh[0], obj_xywh[1], obj_xywh[2], obj_xywh[3])
        print('{} --- {}'.format(sub_debug, obj_debug))
        self.draw()
 
    def _conditional_step(self, density_img, sampler, patch, bbox_nxysa, model_sampler, conditioned_mvn, density_ax, n_samples=5000):
        density_img *= 0.
        # sample a subject bbox
        bbox_sample = sampler.sample() # nxysa (regular-space)
        while not u.valid_bbox_nxysa(bbox_sample):
            bbox_sample = sampler.sample()
        bbox_nxysa = np.copy(bbox_sample)
        bbox_ixywh = u.nxysa_to_ixywh(bbox_sample, self.img_w, self.img_h)
        self._update_bbox_patch(bbox_ixywh, patch)

        return
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
    
    def _conditional_step_ORIG(self, density_img, cond_sampler, cond_patch, cond_bbox_norm, model_sampler, conditioned_mvn, density_ax, n_samples=5000):
        density_img *= 0.
        # sample a subject bbox
        sub_bbox_coords_norm = cond_sampler.sample()
        while not u.valid_bbox_aa(sub_bbox_coords_norm):
            sub_bbox_coords_norm = cond_sampler.sample()
        cond_bbox_norm[:] = sub_bbox_coords_norm
        sub_bbox_coords_img = u.norm_to_img(sub_bbox_coords_norm, self.img_w, self.img_h)
        self._update_bbox_patch(sub_bbox_coords_img, cond_patch)

        return
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
    
    def _update_bbox_patch(self, bbox, patch):
        patch.set_x(bbox[0])     
        patch.set_y(bbox[1])     
        patch.set_width(bbox[2]) 
        patch.set_height(bbox[3])
