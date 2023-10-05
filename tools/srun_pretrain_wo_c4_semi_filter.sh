now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_coco_vg_10e_semi_filter
mkdir -p $save_path


srun --partition a100 \
    --gres=gpu:4 \
    --ntasks-per-node=4 \
    --cpus-per-task=2 \
    --nodes=2 \
    --job-name=pretrain \
    --mem=128G  \
    --time 36:00:00 \
    -A ayuille1_gpu    \
    --kill-on-bad-exit=1 \
    python Pretrain_semi_filter.py \
    --config configs/Pretrain_10e_semi.yaml \
    --amp  \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt