# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/huggingface/diffusers/tree/v0.15.0/examples/text_to_image
# ------------------------------------------------------------------------------------
MODEL_NAME="CompVis/stable-diffusion-v1-4" #"CompVis/stable-diffusion-v1-4"
#TRAIN_DATA_DIR="./data/laion_aes/pt_cache_212k" # please adjust it if needed
TRAIN_DATA_DIR="./data/laion_aes/latent_212k" # 절대 경로로 설정]
EXTRA_TEXT_DIR="./data/laion400m-meta"

UNET_CONFIG_PATH="./src/unet_config"
UNET_NAME="bk_base" # option: ["bk_base", "bk_small", "bk_tiny"]

OUTPUT_DIR="./results/unseen/ours_400M_CVPR_"$UNET_NAME # please adjust it if needed
MODEL_ID="nota-ai/bk-sdm-${UNET_NAME#bk_}"

BATCH_SIZE=64
GRAD_ACCUMULATION=1
NUM_GPUS=4
StartTime=$(date +%s)

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes ${NUM_GPUS} src/kd_train_text_to_image.py \
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
  --use_unseen_setting
EndTime=$(date +%s)
echo "** KD training takes $(($EndTime - $StartTime)) seconds."