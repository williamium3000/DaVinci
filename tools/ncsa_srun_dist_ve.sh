now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/Pretrain_10e_wo_c4_semi_acc2_8gpu_coco_only/ve_davinci_cfg_5e
cfg=configs/VE.yaml
ckpt=work_dirs/Pretrain_10e_wo_c4_semi_acc2_8gpu_coco_only/checkpoint.pth
mkdir -p $save_path


srun --partition gpuA40x4 \
    --gres=gpu:4 \
    --ntasks-per-node=1 \
    --cpus-per-task=5 \
    --nodes=1 \
    --job-name=pretrain \
    --mem-per-cpu=20GB  \
    --time 48:00:00 \
    -A bbrt-delta-gpu    \
    --kill-on-bad-exit=1 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port 29506 --use_env VE.py --config $cfg \
    --output_dir $save_path --amp \
    --checkpoint $ckpt 2>&1 | tee $save_path/$now.txt &