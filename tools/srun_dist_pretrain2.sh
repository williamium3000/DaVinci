now=$(date +"%Y%m%d_%H%M%S")
config=configs/Pretrain_10e_semi_acc2_4gpu_coco_only.yaml
save_path=work_dirs/Pretrain_10e_wo_c4_acc2_4gpu_coco_only_blip_imagenet
mkdir -p $save_path


srun --partition ica100 \
    --gres=gpu:4 \
    --ntasks-per-node=1 \
    --cpus-per-gpu=3 \
    --nodes=1 \
    --job-name=pretrain \
    --mem=200G  \
    --time 48:00:00 \
    -A ayuille1_gpu    \
    --kill-on-bad-exit=1 \
    python -m torch.distributed.launch --nproc_per_node=4  \
    --master_port=39587 \
    --use_env Pretrain_with_blip.py \
    --config $config \
    --amp  \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt