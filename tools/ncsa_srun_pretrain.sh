now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/Pretrain_wo_c4_10e_acc2_8gpu_coco_only
mkdir -p $save_path

srun --partition gpuA100x8 \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    --cpus-per-task=5 \
    --nodes=1 \
    --job-name=pretrain \
    --mem-per-cpu=16GB  \
    --time 48:00:00 \
    -A bbrt-delta-gpu    \
    --kill-on-bad-exit=1 \
    python Pretrain_wo_c4.py \
    --config configs/Pretrain_10e_semi_acc2_8gpu_coco_only.yaml \
    --amp  \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt &