# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/huggingface/diffusers/tree/v0.15.0/examples/text_to_image
# ------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)

# 사용 가능한 GPU 개수 자동으로 감지
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

MODEL_NAME="CompVis/stable-diffusion-v1-4"
TRAIN_DATA_DIR="./data/laion_aes/latent_212k" # 절대 경로로 설정
EXTRA_TEXT_DIR="./data/laion400m-meta"

UNET_CONFIG_PATH="./src/unet_config_channel_small_4"
UNET_NAME="original" # option: ["bk_base", "bk_small", "bk_tiny"]

OUTPUT_DIR="./results/ablation_text_size_ch/10M"
MODEL_ID="nota-ai/bk-sdm-${UNET_NAME#bk_}"

BATCH_SIZE=64  # GPU당 batch size
TOTAL_BATCH_SIZE=256  # BATCH_SIZE * GRAD_ACCUMULATION = 256이 되도록 설정
GRAD_ACCUMULATION=$((TOTAL_BATCH_SIZE / (BATCH_SIZE * NUM_GPUS)))  # 동적으로 GRAD_ACCUMULATION 계산

StartTime=$(date +%s)

# 공통 파라미터 설정
COMMON_ARGS="
  --pretrained_model_name_or_path $MODEL_NAME \
  --train_data_dir $TRAIN_DATA_DIR\
  --extra_text_dir $EXTRA_TEXT_DIR\
  --use_ema \
  --resolution 512 --center_crop --random_flip \
  --train_batch_size $BATCH_SIZE \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --learning_rate 5e-05 \
  --max_grad_norm 1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --report_to="wandb" \
  --seed 1234 \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --checkpointing_steps 25000 \
  --valid_steps 10000 \
  --lambda_sd 1.0 --lambda_kd_output 1.0 --lambda_kd_feat 1.0 \
  --unet_config_path $UNET_CONFIG_PATH --unet_config_name $UNET_NAME \
  --output_dir $OUTPUT_DIR \
  --max_train_steps 400000 \
  --model_id $MODEL_ID \
  --drop_text \
  --random_conditioning \
  --random_conditioning_lambda 5 \
  --dataloader_num_workers 2 \
  --channel_mapping \
  --max_extra_text_samples 10000000
"
#--drop_text

# 멀티 GPU 또는 싱글 GPU 실행 조건
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --num_processes ${NUM_GPUS} src/kd_train_text_to_image.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/kd_train_text_to_image.py $COMMON_ARGS
fi

EndTime=$(date +%s)
echo "** KD training takes $(($EndTime - $StartTime)) seconds."
