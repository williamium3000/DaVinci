now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_coco_vg_c4_10e_small
mkdir -p $save_path
python3 -m torch.distributed.launch --nproc_per_node=4  \
    --master_port=39587 \
    --use_env Pretrain_small.py \
    --config configs/Pretrain_10e_small_acc2_8gpu.yaml \
    --amp \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt