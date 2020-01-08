# generate a dataset for training VGG16 from the VRD dataset annotations and images
import numpy as np
import os.path
import json as j
from PIL import Image

VGG_W = 224
VGG_H = 224

if __name__ == "__main__":
    dataset = 'train'
    vrd_dir = '/home/econser/research/irsg_psu_pdx/data/VRD'
    json_annos = 'annotations_{}.json'.format(dataset)
    
    full_vrd_annos = os.path.join(vrd_dir, json_annos)
    with open(full_vrd_annos, 'rb') as f:
        j_anno = j.load(f)

    src_img_dir = '/home/econser/research/irsg_psu_pdx/data/sg_dataset/sg_{}_images'.format(dataset)
    out_dir = '/home/econser/research/irsg_psu_pdx/models/model_definitions/vrd_vgg/{}'.format(dataset)
    #anno_txt = open(os.path.join(out_dir, '{}.txt'.format(dataset)), 'wb')
    mean_img = np.zeros((1,3))
    n_generated_images = 0
    for anno_key in j_anno:
        anno_key = anno_key.encode('ascii', 'ignore')

        # find all unique objects in this image
        bboxes = []
        for rel in j_anno[anno_key]:
            s = rel['subject']
            sub_ix = int(s['category']) + 1
            b = s['bbox'] # y0, y1, x0, x1
            sub_bbox = [b[2], b[0], b[3], b[1]] # x0, y0, x1, y1
            anno_tup = (sub_ix, sub_bbox)
            if anno_tup not in bboxes:
                bboxes.append(anno_tup)
            
            o = rel['object']
            obj_ix = int(o['category']) + 1
            b = o['bbox']
            obj_bbox = [b[2], b[0], b[3], b[1]]
            anno_tup = (obj_ix, obj_bbox)
            if anno_tup not in bboxes:
                bboxes.append(anno_tup)

        # crop, resize and save each patch
        base_img = Image.open(os.path.join(src_img_dir, anno_key))
        for ix, anno_patch in enumerate(bboxes):
            crop_coords = anno_patch[1]
            cropped_img = base_img.crop(crop_coords) # TODO: try cropping a square then resizing to VGG input to preserve aspect ratio of the data
            #resized_img = cropped_img.resize((VGG_W, VGG_H))
            mean_img += np.mean(cropped_img, axis=(0,1))
            
            class_category = anno_patch[0]
            root_fname = anno_key.split('.')[0]
            extension = anno_key.split('.')[1]
            out_fname = '{}_{:02d}-{:03d}.{}'.format(root_fname, ix+1, class_category, extension)
            full_out_fname = os.path.join(out_dir, out_fname)
            #cropped_img.save(full_out_fname)
            n_generated_images += 1
            anno_line = '{} {}\n'.format(out_fname, class_category)
            #anno_txt.write(anno_line)
    #anno_txt.close()
    mean_img /= float(n_generated_images)
    mean_img *= 255
    print(mean_img.astype(np.uint8))
