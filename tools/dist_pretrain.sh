now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_coco_vg_c4_5e_acc2_4gpu
mkdir -p $save_path
python3 -m torch.distributed.launch --nproc_per_node=4  \
    --master_port=39587 \
    --use_env Pretrain.py \
    --config configs/Pretrain_5e_semi_acc2_4gpu.yaml \
    --amp \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt