export CUDA_VISIBLE_DEVICES=0
python ./guangdong/workspace/tools/generate_result.py \
	./guangdong/workspace/configs/r50/cascade_rcnn_r50_fpn_20e_coco_cam1.py \
	./guangdong/workspace/configs/r50/cascade_rcnn_r50_fpn_20e_coco_cam2.py \
	./guangdong/workspace/configs/r50/cascade_rcnn_r50_fpn_20e_coco_cam3.py \
	./work_dirs/cascade_rcnn_r50_fpn_20e_coco_cam1_2000/epoch_5.pth \
	./work_dirs/cascade_rcnn_r50_fpn_20e_coco_cam2_2000/epoch_5.pth \
	./work_dirs/cascade_rcnn_r50_fpn_20e_coco_cam3_2000/epoch_5.pth \
	origin_diff_cam.json \
	all
	
