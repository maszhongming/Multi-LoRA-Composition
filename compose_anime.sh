export CUDA_VISIBLE_DEVICES=0

python compose_lora.py \
    --method composite \
    --compos_num 2 \
    --save_path output \
    --lora_scale 0.8 \
    --image_style anime \
    --denoise_steps 200 \
    --cfg_scale 10 \
