now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_coco_vg_10e_semi_sampling
mkdir -p $save_path

python -m torch.distributed.launch --nproc_per_node=4  \
    --master_port=39587 \
    --use_env Pretrain_wo_c4_semi.py \
    --config configs/Pretrain_10e_semi.yaml \
    --amp \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt