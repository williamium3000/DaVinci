now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_coco_vg_c4_10e_semi_filter/nlvr2
cfg=configs/NLVR.yaml
ckpt=work_dirs/pretrain_coco_vg_c4_10e_semi_filter/checkpoint.pth
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29508 --use_env NLVR.py --config $cfg \
--output_dir $save_path \
--checkpoint $ckpt 2>&1 | tee $save_path/$now.txt
