now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_coco_vg_c4_10e_semi_filter
mkdir -p $save_path

python -m torch.distributed.launch --nproc_per_node=8  \
    --master_port=39587 \
    --use_env Pretrain_semi_filter.py \
    --config configs/Pretrain_10e_semi.yaml \
    --amp \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt