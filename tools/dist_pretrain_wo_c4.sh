now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/pretrain_vg_coco_wo_c4_10e_acc2_4gpu 
mkdir -p $save_path
python3 -m torch.distributed.launch --nproc_per_node=4  \
    --master_port=39587 \
    --use_env Pretrain_wo_c4.py \
    --config configs/Pretrain_10e_semi_acc2_4gpu.yaml \
    --output_dir $save_path 2>&1 | tee $save_path/$now.txt