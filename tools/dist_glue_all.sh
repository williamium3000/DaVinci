CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_glue.sh mrpc 39500 &
CUDA_VISIBLE_DEVICES=2,3 bash tools/dist_glue.sh mnli 39501 &
CUDA_VISIBLE_DEVICES=4,5 bash tools/dist_glue.sh cola 39502 &
CUDA_VISIBLE_DEVICES=6,7 bash tools/dist_glue.sh qqp 39503 &
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_glue.sh sst2 39504 &
CUDA_VISIBLE_DEVICES=2,3 bash tools/dist_glue.sh qnli 39505 &
CUDA_VISIBLE_DEVICES=4,5 bash tools/dist_glue.sh rte 39506 &
CUDA_VISIBLE_DEVICES=6,7 bash tools/dist_glue.sh stsb 39507 &