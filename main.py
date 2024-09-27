import argparse
import os, sys
import warnings

import torch.multiprocessing as mp

import torch

from transformers import CLIPTextModel, CLIPTokenizer


from diffusers import StableDiffusionPipeline
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

def get_parser():
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()

    return args

def pre_caching(args):

    device = torch.device('cuda:0')
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    # Define teacher and student
    T_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision, cache_dir="/workspace/huggingface"
    )
    
        
    
def distillation(rank, world_size, args):

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    # Define teacher and student
    T_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    config_student = UNet2DConditionModel.load_config(args.unet_config_path, subfolder=args.unet_config_name)
    S_unet = UNet2DConditionModel.from_config(config_student, revision=args.non_ema_revision)
    
    
    
    
def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    parser = get_parser()
    distill_args = parser.parse_args(argv[1:])  # argv[1:]로 수정하여 인자 전달
    #seed_everything(distill_args.seed)
    
    if distill_args.pre_caching:
        pre_caching(distill_args)
        
    else:
        os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
        os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
        os.environ['NCCL_TIMEOUT'] = '1800'
        os.environ['NCCL_TIMEOUT_MS'] = '1200000'  # 개별 NCCL 작업의 타임아웃을 20분으로 설정

        # world_size 설정
        os.environ['MASTER_ADDR'] = distill_args.MASTER_ADDR
        os.environ['MASTER_PORT'] = distill_args.MASTER_PORT

        if distill_args.world_size is None:
            # 선택하지 않은 경우, torch.cuda.device_count()를 사용
            distill_args.world_size = torch.cuda.device_count()
            
        # world_size는 지정된 world_size와 device_count 중 더 작은 값으로 설정
        world_size = min(torch.cuda.device_count(), distill_args.world_size)
        print('world_size(gpu num): ', world_size)
        
        # Ensure we have multiple GPUs available
        if world_size < 1:
            print("No GPUs available for DDP. Exiting...")
            sys.exit(1)

        # Spawn processes for DDP
        mp.spawn(
            distillation,
            args=(world_size, distill_args),
            nprocs=world_size,
            join=True
        )

if __name__ == '__main__':
    main(sys.argv)  # main()을 직접 호출