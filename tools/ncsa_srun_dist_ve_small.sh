now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_coco_vg_c4_10e_small/ve_davinci_cfg_5e
cfg=configs/VE_small.yaml
ckpt=work_dirs/pretrain_coco_vg_c4_10e_small/checkpoint_09.pth
mkdir -p $save_path


srun --partition gpuA100x4 \
    --gres=gpu:4 \
    --ntasks-per-node=1 \
    --cpus-per-task=5 \
    --nodes=1 \
    --job-name=pretrain \
    --mem-per-cpu=20GB  \
    --time 48:00:00 \
    -A bbrt-delta-gpu    \
    --kill-on-bad-exit=1 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port 29506 --use_env VE_small.py --config $cfg \
    --output_dir $save_path \
    --checkpoint $ckpt 2>&1 | tee $save_path/$now.txt &