now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_coco_vg/nlvr2
cfg=configs/NLVR.yaml
ckpt=work_dirs/pretrain_coco_vg/checkpoint_39.pth
mkdir -p $save_path

 python -m torch.distributed.launch --nproc_per_node=8 --master_port 29506 --use_env NLVR.py --config $cfg \
--output_dir $save_path \
--checkpoint $ckpt 2>&1 | tee $save_path/$now.txt
