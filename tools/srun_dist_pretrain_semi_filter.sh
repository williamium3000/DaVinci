now=$(date +"%Y%m%d_%H%M%S")
config=configs/Pretrain_10e_semi_acc2_4gpu.yaml
save_path=work_dirs/pretrain_coco_vg_10e_semi_filter_acc2_4gpu
mkdir -p $save_path


srun --partition ica100 \
    --gres=gpu:4 \
    --ntasks-per-node=1 \
    --cpus-per-task=5 \
    --nodes=1 \
    --job-name=pretrain \
    --mem=128G  \
    --time 72:00:00 \
    -A ayuille1_gpu    \
    --kill-on-bad-exit=1 \
    python -m torch.distributed.launch --nproc_per_node=4  \
    --master_port=39587 \
    --use_env Pretrain_semi_filter.py \
    --config $config \
    --amp  \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt