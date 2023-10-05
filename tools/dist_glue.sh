now=$(date +"%Y%m%d_%H%M%S")
task=$1
save_path=work_dirs/pretrain_coco_vg/glue/$task
ckpt=work_dirs/pretrain_coco_vg/checkpoint_39.pth
mkdir -p $save_path

accelerate launch --main_process_port $2 \
    glue.py \
  --model_name_or_path $ckpt \
  --task_name $task \
  --max_length 128 \
  --per_device_train_batch_size 128 \
  --learning_rate 2e-5 \
  --num_warmup_steps 50\
  --num_train_epochs 8 \
  --output_dir $save_path 2>&1 | tee $save_path/$now.txt