import sys
import matplotlib.image as image
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class Viz(object):
    """
    Assume only 2 objects are needed
    Grid is:
    +-----+-----+-----+
    |workspace  |subj |
    |           |plot |
    +           +-----+
    |           |obj  |
    |           |plot |
    +-----+-----+-----+
    |subj |obj  |sit  |
    |vals |vals |score|
    +-----+-----+-----+
    """
    def __init__(self, subject_str, predicate_str, object_str):
        # store the subject, object, and predicate names
        self.subject_str = subject_str
        self.object_str = object_str
        self.predicate_str = predicate_str

        # track the subject bboxes and support scores
        self.subject_internal_support = 0.0
        self.subject_external_support = 0.0
        self.subject_total_support = 0.0

        self.subject_best_score = 0.0

        self.subject_gt_bbox = None
        self.subject_gt_label = None

        # track the object bboxes and support scores
        self.object_internal_support = 0.0
        self.object_external_support = 0.0
        self.object_total_support = 0.0

        self.object_best_score = 0.0

        self.object_gt_bbox = None
        self.object_gt_label = None
        
        # track the situation score
        self.situation_score = 0.0
        
        # start the plt figure
        self.fig = plt.figure(0)
        self.fig.set_facecolor('#f0f0f0')

        # set up the subplots here (r, c)
        self.workspace_ax      = plt.subplot2grid((4,8), (0, 0), rowspan=3, colspan=4)
        self.best_sub_plot_ax  = plt.subplot2grid((4,8), (0, 4), colspan=2)
        self.best_sub_vals_ax  = plt.subplot2grid((4,8), (1, 4), colspan=2)
        self.best_sub_vals_ax.axis('off')
        self.sub_plot_ax       = plt.subplot2grid((4,8), (2, 4), colspan=2)
        self.sub_vals_ax       = plt.subplot2grid((4,8), (3, 4), colspan=2)
        self.best_obj_plot_ax  = plt.subplot2grid((4,8), (0, 6), colspan=2)
        self.best_obj_vals_ax  = plt.subplot2grid((4,8), (1, 6), colspan=2)
        self.best_obj_vals_ax.axis('off')
        self.obj_plot_ax       = plt.subplot2grid((4,8), (2, 6), colspan=2)
        self.obj_vals_ax       = plt.subplot2grid((4,8), (3, 6), colspan=2)
        self.best_bbox_vals_ax = plt.subplot2grid((4,8), (3, 0), colspan=4)

        # use subplot grid to arrange the plot components
        # workspace_ax is the main workspace that displays the image and bbox proposals
        self.workspace_ax.axis('off')

        # placeholder for subject conditional plot
        self.sub_plot_ax.set_xticklabels([])
        self.sub_plot_ax.set_yticklabels([])
        self.sub_plot_ax.set_xticks([], [])
        self.sub_plot_ax.set_yticks([], [])

        # object conditional plot
        self.obj_plot_ax.set_xticklabels([])
        self.obj_plot_ax.set_yticklabels([])
        self.obj_plot_ax.set_xticks([], [])
        self.obj_plot_ax.set_yticks([], [])

        # display subject support values
        self.sub_vals_ax.axis('off')
        self.sub_values_text = self.sub_vals_ax.text(0.0, 1.0, subject_str, fontsize=15, va='top', fontname='Ubuntu Mono', color='r')
        
        # display object support values
        self.obj_vals_ax.axis('off')
        self.obj_values_text = self.obj_vals_ax.text(0.0, 1.0, object_str, fontsize=15, va='top', fontname='Ubuntu Mono', color='b')

        # best bbox values
        self.best_bbox_vals_ax.axis('off')
        self.best_bbox_vals_text = self.best_bbox_vals_ax.text(0.0, 1.0, object_str, fontsize=15, va='top', fontname='Ubuntu Mono', color='b')

        # set up keyboard intercept
        self.fig.canvas.mpl_connect('key_press_event', self.keypress)
        self.state = 'RUNNING'
        
        # tighten up the padding and show
        plt.tight_layout(pad=0.3, h_pad=0, w_pad=0)
        plt.ion()
        plt.show()

    def keypress(self, event):
        sys.stdout.flush()
        temp = self.state
        if event.key == ' ' and self.state == 'RUNNING':
            self.state = 'PAUSED'
        elif event.key == ' ' and self.state == 'PAUSED':
            self.state = 'RUNNING'
        elif event.key == 'right' and self.state == 'PAUSED':
            self.state = 'STEP'
        elif event.key == 'g':
            self.state = 'RUNNING'
        elif event.key == 'q':
            # terminate
            self.state = 'TERMINATED'
        if __debug__: print('{} + {} => {}'.format(temp, event.key, self.state))

    def get_state(self):
        return self.state

    #---------------------------------------------------------------------------
    def draw(self):
        plt.draw()
        if self.state == 'STEP':
            self.state = 'PAUSED'
        plt.pause(0.001) # this is needed or updates don't have time to flush
        
    def set_image(self, img_fname):
        # clear the current image
        self.workspace_ax.clear()

        # open the new one and show it
        img = image.imread(img_fname)
        self.img_width = img.shape[1]
        self.img_height = img.shape[0]
        self.workspace_ax.imshow(img)
        self.workspace_ax.axis('off')

        # update the title
        self.workspace_ax.set_title(img_fname.split('/')[-1])

        # initiate the subject and object bbox patches
        self.subject_bbox_patch = patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='r', facecolor='none')
        self.workspace_ax.add_patch(self.subject_bbox_patch)

        self.best_subject_bbox_patch = patches.Rectangle((0,0),1,1,linewidth=3,edgecolor='m',facecolor='none')
        self.workspace_ax.add_patch(self.best_subject_bbox_patch)
        self.best_subject_label = self.workspace_ax.text(0,0, self.subject_str, bbox=dict(facecolor='m', alpha=0.5), fontsize=12, color='r')
        
        self.object_bbox_patch = patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='b', facecolor='none')
        self.workspace_ax.add_patch(self.object_bbox_patch)

        self.best_object_bbox_patch = patches.Rectangle((0,0),1,1,linewidth=3,edgecolor='m',facecolor='none')
        self.workspace_ax.add_patch(self.best_object_bbox_patch)
        self.best_object_label = self.workspace_ax.text(0,0, self.object_str, bbox=dict(facecolor='m', alpha=0.5), fontsize=12, color='b')

    def set_subj_patch(self, img_data):
        self.sub_plot_ax.clear()
        self.sub_plot_ax.imshow(img_data)
        self.sub_plot_ax.axis('off')
        self.sub_plot_ax.set_title('current {}'.format(self.subject_str))

    def set_obj_patch(self, img_data):
        self.obj_plot_ax.clear()
        self.obj_plot_ax.imshow(img_data)
        self.obj_plot_ax.axis('off')
        self.obj_plot_ax.set_title('current {}'.format(self.object_str))

    def set_best_subj_patch(self, img_data):
        self.best_sub_plot_ax.clear()
        self.best_sub_plot_ax.imshow(img_data)
        self.best_sub_plot_ax.axis('off')
        self.best_sub_plot_ax.set_title('best {}'.format(self.subject_str))

    def set_best_obj_patch(self, img_data):
        self.best_obj_plot_ax.clear()
        self.best_obj_plot_ax.imshow(img_data)
        self.best_obj_plot_ax.axis('off')
        self.best_obj_plot_ax.set_title('Best {}'.format(self.object_str))
        
    def get_image_shape(self):
        return self.img_width, self.img_height
    
    #---------------------------------------------------------------------------
    def _update_patch(self, bbox, patch, label=None):
        patch.set_x(bbox[0])
        patch.set_y(bbox[1])
        patch.set_width(bbox[2])
        patch.set_height(bbox[3])

        if label is not None:
            label.set_x(bbox[0] + 3)
            label.set_y(bbox[1] - 9)

    # set up GT bboxes
    def set_gt_bboxes(self, gt_bboxes):
        if self.subject_gt_bbox is None:
            self.subject_gt_bbox = patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='r', linestyle=':', facecolor='none')
            self.workspace_ax.add_patch(self.subject_gt_bbox)
            self.subject_gt_label = self.workspace_ax.text(0, 0, self.subject_str, bbox=dict(facecolor='m', alpha=0.5), fontsize=12, color='r')
        self._update_patch(gt_bboxes[0], self.subject_gt_bbox, self.subject_gt_label)

        if self.object_gt_bbox is None:
            self.object_gt_bbox = patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='b', linestyle=':', facecolor='none')
            self.workspace_ax.add_patch(self.object_gt_bbox)
            self.object_gt_label = self.workspace_ax.text(0, 0, self.object_str, bbox=dict(facecolor='m', alpha=0.5), fontsize=12, color='b')
        self._update_patch(gt_bboxes[1], self.object_gt_bbox, self.object_gt_label)
    
    # expects bboxes in ixywh format
    def set_subject_bbox(self, bbox):
        self.subject_bbox = bbox
        self._update_patch(bbox, self.subject_bbox_patch)

    # expects bboxes in ixywh format
    def set_object_bbox(self, bbox):
        self.object_bbox = bbox
        self._update_patch(bbox, self.object_bbox_patch)

    # expects bboxes in ixywh format
    def set_best_subject_bbox(self, bbox):
        self.best_subject_bbox = bbox
        self._update_patch(bbox, self.best_subject_bbox_patch, self.best_subject_label)

    # expects bboxes in ixywh format
    def set_best_object_bbox(self, bbox):
        self.best_object_bbox = bbox
        self._update_patch(bbox, self.best_object_bbox_patch, self.best_object_label)
        
    #---------------------------------------------------------------------------
    def set_situation_score(self, sit_score):
        self.situation_score = sit_score
        # TODO: also do a running line plot of situation score
    
    def set_subject_supports(self, internal, external, total, gt_iou=None):
        self.subject_internal_support = internal
        self.subject_external_support = external
        self.subject_total_support = total
        self.subject_iou = gt_iou

        if gt_iou is not None:
            self.subject_val_str = '{}\nInternal:{: 0.3f}\nExternal:{: 0.3f}\nTotal   :{: 0.3f}\nGT IoU  :{: 0.3f}'.format(self.subject_str, internal, external, total, gt_iou)
        else:
            self.subject_val_str = '{}\nInternal:{: 0.3f}\nExternal:{: 0.3f}\nTotal   :{: 0.3f}\nGT IoU  : -----'.format(self.subject_str, internal, external, total)
        self.sub_values_text.set_text(self.subject_val_str)

    def set_best_subject_score(self, score):
        self.subject_best_score = score
        self.best_score_str = 'Best scores\n{}:{:0.3f}\n{}:{:0.3f}\nSituation:{:0.3f}'.format(self.subject_str, self.subject_best_score, self.object_str, self.object_best_score,self.situation_score)
        self.best_bbox_vals_text.set_text(self.best_score_str)

    def set_best_subject_iou(self, gt_iou):
        pass
        
    def set_object_supports(self, internal, external, total, gt_iou=None):
        self.object_internal_support = internal
        self.object_external_support = external
        self.object_total_support = total
        self.object_gt_iou = gt_iou

        if gt_iou is not None:
            self.object_val_str = '{}\nInternal:{: 0.3f}\nExternal:{: 0.3f}\nTotal   :{: 0.3f}\nGT IoU  :{: 0.3f}'.format(self.object_str, internal, external, total, gt_iou)
        else:
            self.object_val_str = '{}\nInternal:{: 0.3f}\nExternal:{: 0.3f}\nTotal   :{: 0.3f}\nGT IoU  : -----'.format(self.object_str, internal, external, total)
        self.obj_values_text.set_text(self.object_val_str)

    def set_best_object_score(self, score):
        self.object_best_score = score
        self.best_score_str = 'Best scores\n{}:{:0.3f}\n{}:{:0.3f}\nSituation:{:0.3f}'.format(self.subject_str, self.subject_best_score, self.object_str, self.object_best_score,self.situation_score)
        self.best_bbox_vals_text.set_text(self.best_score_str)

    def set_best_object_iou(self, gt_iou):
        pass
