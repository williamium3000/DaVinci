now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_coco_vg_c4_10e_small
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
    python Pretrain_small.py \
    --config configs/Pretrain_10e_small.yaml \
    --amp  --resume True --checkpoint work_dirs/pretrain_coco_vg_c4_10e_small/checkpoint_08.pth \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt