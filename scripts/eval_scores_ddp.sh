GPU_NUM=0  # 평가 작업은 메인 GPU만 사용

# MODEL_ID 설정
MODEL_ID=generated_images

# 경로 설정
IMG_PATH=./results/$MODEL_ID/im256
DATA_LIST=../T2I_distillation/data/mscoco_val2014_30k/metadata.csv

# === Inception Score (IS) ===
IS_TXT=./results/$MODEL_ID/im256_is.txt

if [ $RANK -eq 0 ]; then
    echo "=== Inception Score (IS) ==="
    fidelity --gpu $GPU_NUM --isc --input1 $IMG_PATH | tee $IS_TXT
    echo "============"
fi

# === Fréchet Inception Distance (FID) ===
FID_TXT=./results/$MODEL_ID/im256_fid.txt
NPZ_NAME_gen=./results/$MODEL_ID/im256_fid.npz
NPZ_NAME_real=../T2I_distillation/data/mscoco_val2014_41k_full/real_im256.npz

if [ $RANK -eq 0 ]; then
    echo "=== Fréchet Inception Distance (FID) ==="
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid --save-stats $IMG_PATH $NPZ_NAME_gen 2> /dev/null
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid $NPZ_NAME_real $NPZ_NAME_gen 2> /dev/null | tee $FID_TXT
    echo "============"
fi

# # === CLIP Score ===
# CLIP_TXT=./results/$MODEL_ID/im256_clip.txt
# echo "=== CLIP Score ==="
# CUDA_VISIBLE_DEVICES=$DDP_GPU_NUM python3 src/eval_clip_score_ddp.py --img_dir $IMG_PATH --save_txt $CLIP_TXT --data_list $DATA_LIST 2> /dev/null
# echo "============"
