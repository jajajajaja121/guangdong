from .generate_result import *
from .process import *

if __name__=='__main__':
    args = parse_args()
    json_path = './tile_round1_train_20201231/coco_anno.json'
    train_path = './tile_round1_train_20201231/train_imgs'
    test_path = '/home/xiongpan/dataset/flaw-detection/tile_round1_testA_20201231/testA_imgs'
    process_data = ProcessData(json_path, test_path, train_path)
    if args.task == 'generate_result':
        generate_result(args.config1, args.checkpoint1, args.config2, args.checkpoint2, args.config3, args.checkpoint3,
                        save_path, args.json_name, args.flag)
    elif args.task == 'visulization_val':
        img_path = '/tmp/tile_round1_train_20201231/cam3/val'
        result_path = '/home/xiongpan/code/mmdetection-master/guangdong/workspace/result/cam1_crop_new_data_val.json'
        ann_path = '/tmp/tile_round1_train_20201231/cam3/val.json'
        sava_path = '/home/xiongpan/dataset/flaw-detection/visulization/cam1_crop_new_data_val'
        visulization_val(img_path,ann_path,result_path,save_path)
