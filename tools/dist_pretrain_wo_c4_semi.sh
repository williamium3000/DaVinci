now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_coco_vg_10e_semi 
mkdir -p $save_path
python3 -m torch.distributed.launch --nproc_per_node=8  \
    --master_port=39587 \
    --use_env Pretrain_wo_c4_semi.py \
    --config ./configs/Pretrain_10e.yaml \
    --amp \
    --resume True --checkpoint work_dirs/pretrain_coco_vg_10e_semi/checkpoint_07.pth \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt