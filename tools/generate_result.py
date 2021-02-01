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


def genetate_result_single(config,checkpoint,show, show_dir,
                              show_score_thr):
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
        samples_per_gpu=samples_per_gpu,
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
                              show_score_thr)
    # else:
    #     model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False)
    #     outputs = multi_gpu_test(model, data_loader, args.tmpdir,
    #                              args.gpu_collect)




def fuse_single(json_file):
    results = json.load(open(json_file))
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
    return new_result

def fuse_result(json1,json2,json3,save_path):
    res1 = fuse_single(json1)
    res2 = fuse_single(json2)
    res3 = fuse_single(json3)
    result = res1 + res2 +res3
    with open(save_path,'w') as f:
        json.dump(result,f,indent=6)



if __name__=="__main__":
    config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    json_file3 = 'cam3_result_20210113171134.json'
    json_file2 = 'cam2_result_20210113171130.json'
    json_file1 = 'cam1_result_20210113171054.json'

    save_path = 'final_result.json'
    fuse_result(json_file1,json_file2,json_file3,save_path)

