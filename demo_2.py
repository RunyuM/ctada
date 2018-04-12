#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import torch

CLASSES = ('__background__',
           'food')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',),'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where((dets[:, -1] >= thresh))[0]
    if len(inds) == 0:
        return
    else:
        return 1


def demo(net, image_name, path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, path, image_name)
    im = cv2.imread(im_file)
    #print(im_file)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))
    # Visualize detections for each class
    CONF_THRESH = 0.57
    NMS_THRESH = 0

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]
        s=0
        s=vis_detections(im, cls, dets, thresh=CONF_THRESH)
        #print(s)
    if s==1:
        return 1
    else:
        return 0
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = 'vgg16'
    dataset = 'pascal_voc'
    saved_model = os.path.join('output', 'default', DATASETS[dataset][0],'default',
                             NETS[demonet][0] %(70000 if dataset == 'pascal_voc' else 110000))

    print (saved_model)

    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(2,
                          tag='default', anchor_scales=[8, 16, 32])

    net.load_state_dict(torch.load(saved_model))

    net.eval()
    net.cuda()

    print('Loaded network {:s}'.format(saved_model))

    #im_names = ['green-apple_1285.jpg']
    os.chdir('/home/shao112/new_food_image_improved/')
    current=os.getcwd()
    sum=0
    #file=open("scorerecord.txt","w")
    for food_tag in os.listdir(current):
        if os.path.isdir(current+'/'+food_tag+'/'):
            #print(1)
            for site in os.listdir(current+'/'+food_tag+'/'):
                if os.path.isdir(current+'/'+food_tag+'/'+site+'/'):
                    #print(2)
                    path=current+'/'+food_tag+'/'+site
                    #newpath='/home/mao111/pytorch-faster-rcnn/data/demo/newfood/'+food_tag+'/'+site+'/'
                    #os.system('mkdir '+newpath+'/'+food_tag+'/'+site+'/')
                    subsum=0;
                    for im_name in os.listdir(current+'/'+food_tag+'/'+site+'/'):
                        if im_name.endswith(".jpg") or im_name.endswith(".JPG"):
                            #print(im_name)
                            #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                            #print('Demo for data/demo/{}'.format(im_name))
                            print(path)
                            print(im_name)
                            try:
                                demo(net, im_name, path)
                                if demo(net, im_name, path)==0:
                                    #plt.savefig(im_name+'.jpg')
                                    print(1)
                                    os.system('rm '+path+'/'+im_name)
                                    sum=sum+1
                                    subsum=subsum+1
                            except AttributeError:
                                pass
                            except RuntimeError:
                                pass
    print (sum)
    #file.close()
