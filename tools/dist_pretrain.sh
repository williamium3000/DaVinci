now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_coco_vg_c4
mkdir -p $save_path
python3 -m torch.distributed.launch --nproc_per_node=1  \
    --master_port=39587 \
    --use_env Pretrain.py \
    --config ./configs/Pretrain.yaml \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt