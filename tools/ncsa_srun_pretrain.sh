now=$(date +"%Y%m%d_%H%M%S")
config=configs/Pretrain_10e_semi_acc2_8gpu_vg_only.yaml
save_path=work_dirs/Pretrain_10e_wo__acc2_8gpu_vg_only_blip_imagenet
mkdir -p $save_path

srun --partition gpuA40x4 \
    --gres=gpu:4 \
    --ntasks-per-node=4 \
    --cpus-per-task=3 \
    --nodes=2 \
    --job-name=pretrain \
    --mem-per-cpu=12GB  \
    --time 48:00:00 \
    -A bbrt-delta-gpu    \
    --kill-on-bad-exit=1 \
    python Pretrain_with_blip.py \
    --config $config \
    --amp  \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt