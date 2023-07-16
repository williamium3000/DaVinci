now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_cc3m 
mkdir -p $save_path
python3 -m torch.distributed.launch --nproc_per_node=8  \
    --master_port=39587 \
    --use_env Pretrain_wo_c4.py \
    --config ./configs/Pretrain.yaml \
    --checkpoint work_dirs/pretrain_cc3m/checkpoint_29.pth \
    --resume true \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt