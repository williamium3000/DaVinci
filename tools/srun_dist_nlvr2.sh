now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/Pretrain_10e_semi_acc2_8gpu_sampling-topk50-topp0.95-t0.7_coco_vg_semi_filter/nlvr2
cfg=configs/NLVR.yaml
ckpt=work_dirs/Pretrain_10e_semi_acc2_8gpu_sampling-topk50-topp0.95-t0.7_coco_vg_semi_filter/checkpoint.pth
mkdir -p $save_path

srun --partition ica100 \
    --gres=gpu:4 \
    --ntasks-per-node=1 \
    --cpus-per-task=5 \
    --nodes=1 \
    --job-name=pretrain \
    --mem=200G  \
    --time 72:00:00 \
    -A ayuille1_gpu    \
    --kill-on-bad-exit=1 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port 29508 --use_env NLVR.py --config $cfg \
    --output_dir $save_path \
    --checkpoint $ckpt 2>&1 | tee $save_path/$now.txt