# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/huggingface/diffusers/tree/v0.15.0/examples/text_to_image
# ------------------------------------------------------------------------------------

# 사용 가능한 GPU 목록을 쉼표로 구분된 형식으로 가져옴
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)

# 사용 가능한 GPU 개수 자동으로 감지
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# 여러 개의 unet_path를 쉼표로 구분하여 전달 (예: "path1,path2,path3")
unet_paths="./results/abl_RC/BKSDM_init_R_Generated/checkpoint-50000, \
./results/abl_RC/BKSDM_init_R_Generated/checkpoint-75000, \
./results/abl_RC/BKSDM_init_R_Generated/checkpoint-100000, \
./results/abl_RC/BKSDM_init_R_Generated/checkpoint-125000, \
./results/abl_RC/BKSDM_init_R_Generated/checkpoint-150000, \
./results/abl_RC/BKSDM_init_R_Generated/checkpoint-175000, \
./results/abl_RC/BKSDM_init_R_Generated/checkpoint-200000
"

# 쉼표로 구분된 문자열을 배열로 변환
IFS=',' read -r -a unet_path_array <<< "$unet_paths"

StartTime=$(date +%s)

# 배열에 있는 각 unet_path에 대해 명령 실행
for unet_path in "${unet_path_array[@]}"; do
  echo "Running with UNet path: ${unet_path}"
  
  if [ ${NUM_GPUS} -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --num_processes ${NUM_GPUS} src/eval.py \
      --unet_path ${unet_path}
  else
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/eval.py \
      --unet_path ${unet_path}
  fi
done

EndTime=$(date +%s)
echo "** KD training takes $(($EndTime - $StartTime)) seconds."
