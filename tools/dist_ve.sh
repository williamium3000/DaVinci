now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_coco_vg/ve_davinci_cfg
cfg=configs/VE.yaml
ckpt=/root/william/project/explore-vlp/work_dirs/davinci/pretrain_coco_vg_10e/checkpoint_09.pth
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29506 --use_env VE.py --config $cfg \
--output_dir $save_path \
--checkpoint $ckpt 2>&1 | tee $save_path/$now.txt
