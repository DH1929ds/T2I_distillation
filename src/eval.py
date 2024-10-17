import os
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_TIMEOUT'] = '3600'
os.environ['NCCL_TIMEOUT_MS'] = '3600000'  # 개별 NCCL 작업의 타임아웃을 20분으로 설정

import torch
import argparse
import subprocess
import sys
import shutil

from accelerate import Accelerator
from eval_clip_score_ddp import evaluate_clip_score
from generate_ddp2 import sample_images_30k
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # arguments for evaluation
    parser.add_argument("--model_id", type=str, default="nota-ai/bk-sdm-base", help="Path to the pretrained model or checkpoint directory.")
    parser.add_argument("--unet_path", type=str, default="/home/work/StableDiffusion/T2I_distill1_GPU4/results/toy_ddp_bk_base/checkpoint-25000", help="Model checkpoint for evaluate")
    parser.add_argument("--img_sz", type=int, default=512)
    parser.add_argument("--img_resz", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--batch_sz", type=int, default=25)    


    parser.add_argument("--save_txt", type=str, default="./results/generated_images/im256_clip.txt")
    parser.add_argument("--data_list", type=str, default="../T2I_distillation/data/mscoco_val2014_30k/metadata.csv")
    parser.add_argument("--img_dir", type=str, default="./results/generated_images/im256")
    parser.add_argument("--save_dir", type=str, default="./results/generated_images")
    parser.add_argument('--clip_seed', type=int, default=1234, help='Random seed for reproducibility')
    parser.add_argument('--clip_batch_size', type=int, default=50, help='Batch size for processing images')
    args = parser.parse_args()
    return args
        
def main():
    args = parse_args()
    
    accelerator = Accelerator()
                  
    ################################################# Evaluate IS, FID, CLIP SCORE #################################################
    sample_images_30k(args, accelerator, args.unet_path) 
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        try:
            subprocess.run(["sh", "./scripts/eval_scores_ddp.sh"], check=True, stdout=sys.stdout, stderr=sys.stderr )
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running script: {e}")

    # Wait for all ranks to complete the evaluation
    accelerator.wait_for_everyone()
    evaluate_clip_score(args, accelerator)
    accelerator.wait_for_everyone()
    ################################################# Evaluate IS, FID, CLIP SCORE #################################################

    # Read and log evaluation scores to WandB (main process only)
    if accelerator.is_main_process:
        is_txt_path = os.path.join(args.save_dir, "im256_is.txt")
        fid_txt_path = os.path.join(args.save_dir, "im256_fid.txt")
        clip_txt_path = os.path.join(args.save_dir, "im256_clip.txt")

        score_dict = {}

        # Read Inception Score (IS)
        try:
            with open(is_txt_path, "r") as f:
                lines = f.readlines()
                is_score = float(lines[-2].strip().split()[-1])
                score_dict["IS"] = is_score
        except FileNotFoundError:
            print(f"Warning: IS score file {is_txt_path} not found.")
            score_dict["IS"] = None

        # Read Fréchet Inception Distance (FID)
        try:
            with open(fid_txt_path, "r") as f:
                lines = f.readlines()
                fid_score = float(lines[0].strip().split()[-1])
                score_dict["FID"] = fid_score
        except FileNotFoundError:
            print(f"Warning: FID score file {fid_txt_path} not found.")
            score_dict["FID"] = None

        # Read CLIP Score
        try:
            with open(clip_txt_path, "r") as f:
                lines = f.readlines()
                clip_score = float(lines[0].strip().split()[-1])
                score_dict["CLIP"] = clip_score
        except FileNotFoundError:
            print(f"Warning: CLIP score file {clip_txt_path} not found.")
            score_dict["CLIP"] = None

        # Print scores for verification
        print(f"Inception Score (IS): {score_dict['IS']}")
        print(f"Fréchet Inception Distance (FID): {score_dict['FID']}")
        print(f"CLIP Score: {score_dict['CLIP']}")

        try:
            shutil.rmtree(args.save_dir)
            print(f"All folders in {args.save_dir} have been deleted.")
        except Exception as e:
            print(f"Error occurred while deleting folders in {args.save_dir}: {e}")
    torch.cuda.empty_cache()

    accelerator.end_training()

if __name__ == "__main__":
    main()