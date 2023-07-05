now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_coco_vg/ve_davinci_cfg
cfg=configs/VE.yaml
ckpt=work_dirs/davinci/pretrain_coco_vg/checkpoint_39.pth
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29509 --use_env VE.py --config $cfg \
--output_dir $save_path --amp \
--checkpoint $ckpt 2>&1 | tee $save_path/$now.txt
