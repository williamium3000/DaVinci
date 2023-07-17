
python image_linprobe.py \
  --pretrained work_dirs/pretrain_coco_vg/checkpoint_39.pth \
    --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
    --override_cfg "dataset:imagenet;optimizer: {opt: adamW, lr: 1e-4, weight_decay: 0.01}" \
    --config configs/image_linprobe/$1