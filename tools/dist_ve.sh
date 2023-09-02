now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/Pretrain_10e_semi_acc2_4gpu_coco_only/ve_davinci_cfg_5e
cfg=configs/VE.yaml
ckpt=work_dirs/Pretrain_10e_semi_acc2_4gpu_coco_only/checkpoint.pth
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29506 --use_env VE.py --config $cfg \
--output_dir $save_path \
--checkpoint $ckpt 2>&1 | tee $save_path/$now.txt
