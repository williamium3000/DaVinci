now=$(date +"%Y%m%d_%H%M%S")
save_path=work_dirs/image_linprobe/$1 
mkdir -p $save_path

python image_linprobe.py \
  --pretrained work_dirs/pretrain_coco_vg/checkpoint_39.pth \
    --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
    --output_dir $save_path \
    --config configs/image_linprobe/$1.yaml 2>&1 | tee $save_path/$now.txt