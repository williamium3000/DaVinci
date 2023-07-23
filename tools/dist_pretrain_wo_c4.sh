now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/test 
mkdir -p $save_path
python3 -m torch.distributed.launch --nproc_per_node=8  \
    --master_port=39587 \
    --use_env Pretrain_wo_c4.py \
    --config ./configs/Pretrain.yaml \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt