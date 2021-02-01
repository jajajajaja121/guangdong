_base_ = './cascade_rcnn_r50_fpn_20e_coco.py'
load_from = '/home/xiongpan/code/mmdetection-master/pretrain-model/cascade_rcnn_r101_fpn_20e_coco_bbox_mAP-0.425_20200504_231812-5057dcc5.pth'
model = dict(pretrained=None, backbone=dict(depth=101))
