export CUDA_VISIBLE_DEVICES=3

python compose_lora.py \
    --method merge \
    --compos_num 2 \
    --save_path output \
    --lora_scale 0.8 \
    --image_style anime \
    --denoise_steps 200 \
    --cfg_scale 10 \
