now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/Pretrain_10e_wo_c4_semi_acc2_8gpu_coco_only
mkdir -p $save_path


srun --partition gpuA40x4 \
    --gres=gpu:4 \
    --ntasks-per-node=4 \
    --cpus-per-task=5 \
    --nodes=2 \
    --job-name=pretrain \
    --mem-per-cpu=10GB  \
    --time 48:00:00 \
    -A bbrt-delta-gpu    \
    --kill-on-bad-exit=1 \
    python Pretrain_wo_c4_semi_filter.py \
    --config configs/Pretrain_10e_semi_acc2_8gpu_coco_only.yaml \
    --amp \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt &