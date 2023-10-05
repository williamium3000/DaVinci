now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_10e_semi_acc2_4gpu_coco_only
mkdir -p $save_path

python -m torch.distributed.launch --nproc_per_node=8  \
    --master_port=39587 \
    --use_env Pretrain_wo_c4_semi_filter.py \
    --config configs/Pretrain_10e_semi_acc2_4gpu_coco_only.yaml \
    --amp  \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt