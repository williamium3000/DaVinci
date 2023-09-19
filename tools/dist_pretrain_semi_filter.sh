now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_coco_vg_c4_10e_semi_filter_contrastive
mkdir -p $save_path

python -m torch.distributed.launch --nproc_per_node=2  \
    --master_port=39587 \
    --use_env Pretrain_semi_filter_contrastive_sampling.py \
    --config configs/Pretrain_10e_semi_acc2_8gpu_con-sampling-topk4-alpha0.6.yaml \
    --amp \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt