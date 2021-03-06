import os
import json
import os.path as osp
from tqdm import tqdm
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
import argparse
import warnings
from ensemble_boxes import *
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config1', help='test config file path')
    parser.add_argument('config2', help='test config file path')
    parser.add_argument('config3', help='test config file path')
    parser.add_argument('checkpoint1', help='checkpoint file')
    parser.add_argument('checkpoint2', help='checkpoint file')
    parser.add_argument('checkpoint3', help='checkpoint file')
    parser.add_argument('json_name', help='save json_name')
    parser.add_argument('flag', help='which part to generate')
    # parser.add_argument('task', help='which part to generate')

    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def visulize_result(image_path,result_path,save_path):
    results = json.load(open(result_path))
    im_bbox = {}
    for res in results:
        name = res['name']
        bbox = res['bbox']
        category = res['category']
        if not name in im_bbox.keys():
            im_bbox[name] = [bbox,category]
    for im_name in tqdm(im_bbox.keys()):
        img_path = osp.join(image_path,im_name)
        image = cv2.imread(img_path)
        for ann in im_bbox[im_name]:
            bbox = ann[0]
            cat = ann[1]
            image = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),3)
            image = cv2.puttext(image,str(cat),(bbox[0],bbox[1]),cv2.FONT_HERSHEY_SIMPLEX,10,(0,0,255),3)
        img_save = osp.join(save_path,im_name)
        cv2.imwrite(img_save,image)

def fuse_single(results,save_path):
    # results = json.load(open(json_file))
    new_result = []
    for res in tqdm(results):
        name = res['name']
        bbox = res['bbox']
        if bbox[0]==bbox[2] or bbox[1]==bbox[3]:
            continue
        m_ind = name.find('M')
        str_after_m = name[m_ind+3:-4]
        h_index,w_index = str_after_m.split('_', 1)
        h_index = int(h_index)
        w_index = int(w_index)
        fname,ext = osp.splitext(name)
        # import pdb
        # pdb.set_trace()
        father_name = fname[:m_ind+2]
        new_name = father_name+ext
        bbox = [bbox[0]+w_index*500,bbox[1]+h_index*500,bbox[2]+w_index*500,bbox[3]+h_index*500]
        res['name'] = new_name
        res['bbox'] = bbox
        new_result.append(res)
    with open(save_path, 'w') as f:
        json.dump(new_result,f,indent=6)
    return new_result


def py_cpu_nms(dets, thresh):
    # import pdb
    # pdb.set_trace()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep


def fuse_single_new(results,save_path):
    # results = json.load(open(json_file))
    xmin = xmax = ymin = ymax = 0
    # if cam=='CAM1':
    #     xmin = 1366
    #     xmax = 6830
    #     ymin = 400
    #     ymax = 5600
    # elif cam == 'CAM2':
    #     xmin = 1366
    #     xmax = 6830
    #     ymin = 400
    #     ymax = 5600
    # elif cam == 'CMA3':
    #     xmin = 682
    #     xmax = 3413
    #     ymin = 700
    #     ymax = 3300
    final_result = []
    new_result = {}
    for res in tqdm(results):
        name = res['name']
        bbox = res['bbox']
        category = res['category']
        score = res['score']
        if bbox[0]==bbox[2] or bbox[1]==bbox[3]:
            continue
        m_ind = name.find('M')
        str_after_m = name[m_ind+3:-4]
        x,y = str_after_m.split('_', 1)
        x = int(x)
        y = int(y)
        fname,ext = osp.splitext(name)
        father_name = fname[:m_ind+2]
        new_name = father_name+ext
        if not new_name in new_result.keys():
            new_result[new_name] = []
        bbox = [bbox[0]+x,bbox[1]+y,bbox[2]+x,bbox[3]+y]
        # if score<0.05:
        #     continue
        # if bbox[0]<xmin or bbox[1]<ymin or bbox[2]>xmax or bbox[3]>ymax or score<0.05:
        #     continue
        bbox.append(score)
        new_result[new_name].append([bbox,category])

    for img_name in new_result.keys():
        res = new_result[img_name]
        bboxs = np.array([re[0] for re in res])
        if bboxs.shape[0]==0:
            new_result[img_name] = [[], [],[]]
            continue
        # keep = py_cpu_nms(bboxs,0.5)
        # bbox = bboxs[keep]
        # bbox = bboxs
        scores = bboxs[:,4].tolist()
        bboxs = bboxs[:,:4].tolist()
        categories= np.array([re[1] for re in res]).tolist()
        new_result[img_name] = [categories,scores,bboxs]

    for img_name in new_result.keys():
        res = new_result[img_name]
        for cat,score,bbox in zip(res[0],res[1],res[2]):
            fin_res = {}
            fin_res['name'] = img_name
            fin_res['category'] = cat
            fin_res['bbox'] = bbox
            fin_res['score'] = score
            final_result.append(fin_res)

    with open(save_path, 'w') as f:
        json.dump(final_result,f,indent=6)
    return final_result

def process_result(result):
    result_dict = {}
    pre_name_dict = {}
    for res in result:
        name = res['name']
        pre_name_dict[name[:8]] = name
        if not res['name'] in result_dict.keys():
            result_dict[name] = {}
            result_dict[name]['bboxs'] = []
            result_dict[name]['scores'] = []
            result_dict[name]['labels'] = []
        box = res['bbox']
        box = [1.0 * box[0] / 8192, 1.0 * box[1] / 6000, 1.0 * box[2] / 8192, 1.0 * box[3] / 6000]
        result_dict[name]['bboxs'].append(box)
        result_dict[name]['scores'].append(res['score'])
        result_dict[name]['labels'].append(res['category'])
    return result_dict,pre_name_dict

def process_result_cam3(result):
    result_dict = {}
    pre_name_dict = {}

    for res in result:
        name = res['name']
        pre_name_dict[name[:8]] = name
        if not res['name'] in result_dict.keys():
            result_dict[name] = {}
            result_dict[name]['bboxs'] = []
            result_dict[name]['scores'] = []
            result_dict[name]['labels'] = []
        box = res['bbox']
        box = [1.0 * box[0] / 4096, 1.0 * (box[1]-500) / 3000 if box[1]>500 else 0, 1.0 * box[2] / 4096, 1.0 * (box[3]-500) / 3000 if box[3]>500 else 0]
        result_dict[name]['bboxs'].append(box)
        result_dict[name]['scores'].append(res['score'])
        result_dict[name]['labels'].append(res['category'])
    return result_dict,pre_name_dict

def split_result(res_path):
    result = json.load(open(res_path))
    res1 = []
    res2 = []
    res3 = []
    for ann in result:
        if 'CAM1' in ann['name']:
            res1.append(ann)
        if 'CAM2' in ann['name']:
            res2.append(ann)
        if 'CAM3' in ann['name']:
            res3.append(ann)
    return res1,res2,res3

def box_map_cam12(box):
    box = [round(box[0]*8192,2),round(box[1]*6000,2),round(box[2]*8192,2),round(box[3]*6000,2)]
    return box

def box_map_cam3(box):
    box = [round(box[0]*4096,2),round(box[1]*3000+500,2),round(box[2]*4096,2),round(box[3]*3000+500,2)]
    return box


def fuse_result(res1,res2,res3,save_path):
    # import pdb
    # pdb.set_trace()
    # res1_dict,pre_name_dict1 = process_result(res1)
    # res2_dict,pre_name_dict2 = process_result(res2)
    # res3_dict,pre_name_dict3 = process_result(res3)
    # img1_pre = pre_name_dict1.keys()
    # img2_pre = pre_name_dict2.keys()
    # img3_pre = pre_name_dict3.keys()
    # common_img = set(img1_pre) & set(img2_pre) & set(img3_pre)
    # fused_result = {}
    # for img_name in tqdm(common_img):
    #     anno_1 = res1_dict[pre_name_dict1[img_name]]
    #     anno_2 = res2_dict[pre_name_dict2[img_name]]
    #     anno_3 = res3_dict[pre_name_dict3[img_name]]
    #     bbox_res1 = anno_1['bboxs']
    #     score_res1 = anno_1['scores']
    #     cat_res1 = anno_1['labels']
    #     bbox_res2 = anno_2['bboxs']
    #     score_res2 = anno_2['scores']
    #     cat_res2 = anno_2['labels']
    #     bbox_res3 = anno_3['bboxs']
    #     score_res3 = anno_3['scores']
    #     cat_res3 = anno_3['labels']
    #     bbox_list = [bbox_res1,bbox_res2,bbox_res3]
    #     score_list = [score_res1,score_res2,score_res3]
    #     cat_list = [cat_res1,cat_res2,cat_res3]
    #     weights = [2,1,1]
    #     iou_thr = 0.5
    #     skip_box_thr = 0.001
    #     boxes, scores, labels = weighted_boxes_fusion(bbox_list, score_list, cat_list, weights=weights,
    #                                                   iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    #     fused_result[img_name] = {}
    #     fused_result[img_name]['bboxes'] = boxes
    #     fused_result[img_name]['scores'] = scores
    #     fused_result[img_name]['labels'] = labels
    # for i,ann in enumerate(res1):
    #     if ann['name'][:8] in common_img:
    #         res1.pop(i)
    # for i, ann in enumerate(res2):
    #     if ann['name'][:8] in common_img:
    #         res2.pop(i)
    # for i, ann in enumerate(res3):
    #     if ann['name'][:8] in common_img:
    #         res3.pop(i)
    #
    # for pre_name in tqdm(common_img):
    #     anno = fused_result[pre_name]
    #     bboxs = anno['bboxes']
    #     scores = anno['scores']
    #     labels = anno['labels']
    #     for i in range(len(bboxs)):
    #         res1.append({
    #             'name':pre_name_dict1[pre_name],
    #             'categoty':int(labels[i]),
    #             'bbox':box_map_cam12(bboxs[i]),
    #             'score':scores[i]
    #         })
    #         res2.append({
    #             'name': pre_name_dict2[pre_name],
    #             'categoty': int(labels[i]),
    #             'bbox': box_map_cam12(bboxs[i]),
    #             'score': scores[i]
    #         })
    #         res3.append({
    #             'name': pre_name_dict3[pre_name],
    #             'categoty': int(labels[i]),
    #             'bbox': box_map_cam3(bboxs[i]),
    #             'score': scores[i]
    #         })
    # import pdb
    # pdb.set_trace()
    result = res1 + res2 + res3
    # import pdb
    # pdb.set_trace()
    with open(save_path,'w') as f:
        json.dump(result,f,indent=6)

def all_cat_plus_1(json_path):
    all_data = json.load(open(json_path))
    for anno in tqdm(all_data):
        anno["category"] = anno["category"]+2
    with open(json_path,'w') as f:
        json.dump(all_data,f,indent=6)

def process_output(outputs,save_path):
    # import pdb
    # pdb.set_trace()
    result = []
    bboxs = outputs[0]
    names = outputs[1]
    for i,bbox in enumerate(bboxs): #batch size
        for cat,boxs in enumerate(bbox):
            for box in boxs:
                ann = {}
                ann['name'] = names[i]
                ann['category'] = cat+1
                ann['bbox'] = [int(b) for b in box[:4]]
                ann['score'] = float(box[4])
                result.append(ann)
    with open(save_path,'w') as f:
        json.dump(result,f,indent=6)
    return result

def genetate_result_single(config, checkpoint, save_path,show=False, show_dir = None,
                           show_score_thr = 0.3,):
    # import pdb
    # pdb.set_trace()
    cfg = Config.fromfile(config)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=8,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, show, show_dir,
                              show_score_thr,output_guangdong=True)
    result = process_output(outputs,save_path)
    return result

def generate_result(config1,checkpoint1,config2,checkpoint2,config3,checkpoint3,save_path,json_name,flag):
    # import pdb
    # pdb.set_trace()
    save_path1 = "/home/xiongpan/code/mmdetection-master/guangdong/workspace/result/cam1_"+json_name
    save_path2 = "/home/xiongpan/code/mmdetection-master/guangdong/workspace/result/cam2_"+json_name
    save_path3 = "/home/xiongpan/code/mmdetection-master/guangdong/workspace/result/cam3_"+json_name
    save_path1_crop = "/home/xiongpan/code/mmdetection-master/guangdong/workspace/result/cam1_crop_"+json_name
    save_path2_crop = "/home/xiongpan/code/mmdetection-master/guangdong/workspace/result/cam2_crop_"+json_name
    save_path3_crop = "/home/xiongpan/code/mmdetection-master/guangdong/workspace/result/cam3_crop_"+json_name
    if flag == 'all':
        result1 = fuse_single_new(genetate_result_single(config1,checkpoint1,save_path1_crop),save_path1)
        result2 = fuse_single_new(genetate_result_single(config2,checkpoint2,save_path2_crop),save_path2)
        result3 = fuse_single_new(genetate_result_single(config3,checkpoint3,save_path3_crop),save_path3)
        fuse_result(result1,result2,result3,save_path)
    elif flag == '1':
        result1 = fuse_single_new(genetate_result_single(config1, checkpoint1,save_path1_crop),save_path1)
    elif flag == '2' :
        result2 = fuse_single_new(genetate_result_single(config2, checkpoint2,save_path2_crop),save_path2)
    elif flag == '3':
        result3 = fuse_single_new(genetate_result_single(config3, checkpoint3,save_path3_crop),save_path3)

def fuse_result_by_file(json1,json2,json3,save_path):
    res1 = json.load(open(json1))
    res2 = json.load(open(json2))
    res3 = json.load(open(json3))
    res = res1+res2+res3
    with open(save_path,'w') as f:
        json.dump(res,f,indent=6)

def visulization(img_path,ann_path,save_img):
    if not osp.exists(save_img):
        os.makedirs(save_img)
    result = json.load(open(ann_path))
    im_bbox = {}
    for res in result:
        name = res['name']
        bbox = [int(r) for r in res['bbox']]
        cate = res['category']
        score = res['score']
        if not name in im_bbox.keys():
            im_bbox[name] = []
        im_bbox[name].append([cate,score,bbox])
    for im_name in tqdm(im_bbox.keys()):
        path = osp.join(img_path,im_name)
        image = cv2.imread(path)
        annos = im_bbox[im_name]
        for anno in annos:
            cate = anno[0]
            score = round(anno[1],2)
            # if score<0.1:
            #     continue
            bbox = anno[2]
            image = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),thickness=1)
            image =cv2.putText(image,str(cate),(bbox[0],bbox[1]),cv2.FONT_HERSHEY_COMPLEX,5,(255,0,0),thickness=3)
            image = cv2.putText(image,str(score),(bbox[2],bbox[1]),cv2.FONT_HERSHEY_COMPLEX,5,(255,0,0),thickness=3)
        save_path = osp.join(save_img,im_name)
        cv2.imwrite(save_path,image)

def process_coco_result(path):
    coco = json.load(open(path))
    annotations = coco['annotations']
    images = coco['images']
    result = []
    id_name = {}
    for img in images:
        id = img['id']
        name = img['file_name']
        id_name[id] = name
    for anno in annotations:
        res = {}
        id = anno['image_id']
        bbox = anno['bbox']
        category = anno['category_id']
        bbox = [int(bbox[0]),int(bbox[1]),int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])]
        res['name'] = id_name[id]
        res['bbox'] = bbox
        res['category'] = category
        result.append(res)
    return result

def visulization_val(img_path,ann_path,result_path,save_img):
    if not osp.exists(save_img):
        os.makedirs(save_img)
    result = json.load(open(result_path))
    annotation = process_coco_result(ann_path)
    # import pdb
    # pdb.set_trace()
    im_bbox_ann = {}
    im_bbox = {}
    for res in result:
        name = res['name']
        bbox = [int(r) for r in res['bbox']]
        cate = res['category']
        score = res['score']
        if not name in im_bbox.keys():
            im_bbox[name] = []
        im_bbox[name].append([cate,score,bbox])
    for res in annotation:
        name = res['name']
        bbox = [int(r) for r in res['bbox']]
        cate = res['category']
        if not name in im_bbox_ann.keys():
            im_bbox_ann[name] = []
        im_bbox_ann[name].append([cate,bbox])
    # import pdb
    # pdb.set_trace()
    for im_name in tqdm(im_bbox.keys()):
        path = osp.join(img_path,im_name)
        image = cv2.imread(path)
        annos = im_bbox[im_name]
        for anno in annos:
            cate = anno[0]
            score = round(anno[1],2)
            if score<0.1:
                continue
            bbox = anno[2]
            image = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),thickness=1)
            # image =cv2.putText(image,str(cate),(bbox[0],bbox[1]),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),thickness=1)
            # image = cv2.putText(image,str(score),(bbox[2],bbox[1]),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),thickness=1)
        if im_name in im_bbox_ann.keys():
            annos_annotation = im_bbox_ann[im_name]
            for anno in annos_annotation:
                cate = anno[0]
                bbox = anno[1]
                image = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),thickness=1)
                # image =cv2.putText(image,str(cate),(bbox[0],bbox[1]),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),thickness=1)
        save_path = osp.join(save_img,im_name)
        cv2.imwrite(save_path,image)

def fuse_annotations(json1,json2,json3,save):
    coco1 = json.load(open(json1))
    coco2 = json.load(open(json2))
    coco3 = json.load(open(json3))
    coco = {}
    images = []
    annotations = []
    coco['categories'] = coco1['categories']
    images1 = coco1['images']
    images2 = coco2['images']
    images3 = coco3['images']
    annotations1 = coco1['annotations']
    annotations2 = coco2['annotations']
    annotations3 = coco3['annotations']
    img_len1 = len(images1)
    img_len2 = len(images2)
    img_len3 = len(images3)
    ann_len1 = len(annotations1)
    ann_len2 = len(annotations2)
    ann_len3 = len(annotations3)
    for img in tqdm(images1):
        images.append(img)
    for img in tqdm(images2):
        img['id'] = img['id'] + img_len1
        images.append(img)
    for img in tqdm(images3):
        img['id'] = img['id'] + img_len1 + img_len2
        images.append(img)
    for ann in tqdm(annotations1):
        annotations.append(ann)
    for ann in tqdm(annotations2):
        ann['id'] = ann['id'] + ann_len1
        ann['image_id'] = ann['image_id'] + img_len1
        annotations.append(ann)
    for ann in tqdm(annotations3):
        ann['id'] = ann['id'] + ann_len1 + ann_len2
        ann['image_id'] = ann['image_id'] + img_len1 + img_len2
        annotations.append(ann)
    coco['images'] = images
    coco['annotations'] =annotations
    with open(save,'w') as f:
        json.dump(coco,f,indent=6)

def select_result(result,save,thr):
    new_result = []
    for res in tqdm(result):
        name = res['name']
        bbox = res['bbox']
        category = res['category']
        score = res['score']
        if bbox[0] == bbox[2] or bbox[1] == bbox[3] or score<thr:
            continue
        new_result.append(res)
    with open(save,'w') as f:
        json.dump(new_result,f,indent=6)




if __name__=="__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # json_file3 = 'cam3_result_20210113171134.json'
    # json_file2 = 'cam2_result_20210113171130.json'
    # json_file1 = 'cam1_result_20210113171054.json'
    #
    # save_path = 'final_result.json'
    # fuse_result(json_file1,json_file2,json_file3,save_path)
    # args = parse_args()
    # save_path = "/home/xiongpan/code/mmdetection-master/guangdong/workspace/result/"+args.json_name
    # generate_result(args.config1,args.checkpoint1,args.config2,args.checkpoint2,args.config3,args.checkpoint3,save_path,args.json_name,args.flag)
    # img_path = '/tmp/defect_det/tile_round1_testB_20210128/testB_imgs/'
    # result_path = '/home/xiongpan/code/mmdetection-master/guangdong/workspace/result/origin_diff_cam_anchor10_thr0.003.json'
    # save_path = '/home/xiongpan/dataset/flaw-detection/visulization/testB_anchor10'
    # visulization(img_path, result_path, save_path)
    # json1 = "/tmp/tile_round1_train_20201231/all_cam/test1.json"
    # json2 = "/tmp/tile_round1_train_20201231/all_cam/test2.json"
    # json3 = "/tmp/tile_round1_train_20201231/all_cam/test3.json"
    # save = "/tmp/tile_round1_train_20201231/all_cam/test.json"
    # fuse_annotations(json1,json2,json3,save)
    # json1 =  '/home/xiongpan/code/mmdetection-master/guangdong/workspace/result/origin_diff_cam_anchor10.json'
    # save1 =  '/home/xiongpan/code/mmdetection-master/guangdong/workspace/result/origin_diff_cam_anchor10_thr0.003.json'
    # res = json.load(open(json1))
    # select_result(res,save1,0.003)
    res1 =  '/home/xiongpan/code/mmdetection-master/guangdong/workspace/result/cam1_origin_diff_cam.json'
    res2 =  '/home/xiongpan/code/mmdetection-master/guangdong/workspace/result/cam2_origin_diff_cam_anchor10.json'
    res3 =  '/home/xiongpan/code/mmdetection-master/guangdong/workspace/result/cam3_origin_diff_cam_anchor10.json'
    save = '/home/xiongpan/code/mmdetection-master/guangdong/workspace/result/result.json'
    fuse_result_by_file(res1,res2,res3,save)
