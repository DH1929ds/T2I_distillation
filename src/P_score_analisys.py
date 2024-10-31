# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/huggingface/diffusers/blob/v0.15.0/examples/text_to_image/train_text_to_image.py
# ------------------------------------------------------------------------------------
#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import os
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_TIMEOUT'] = '3600'
os.environ['NCCL_TIMEOUT_MS'] = '3600000'  # 개별 NCCL 작업의 타임아웃을 20분으로 설정
import argparse
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler

import logging
import math
import random
from pathlib import Path
from typing import Optional
import subprocess
import sys
import shutil
import uuid
from datetime import timedelta
from PIL import Image

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Subset
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from eval_clip_score_ddp import evaluate_clip_score, evaluate_clip_score_unseen_setting
from generate_ddp2 import sample_images_30k, sample_images_41k
from eval_score_wandb_log import log_eval_scores, log_eval_scores_unseen_setting
from torchvision.utils import save_image, make_grid


import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

import csv
import time
import copy

from P_score_analisys_dataset import x0_dataset, collate_fn
from funcs import MultiConv1x1, get_layer_output_channels, count_parameters

import warnings
warnings.filterwarnings("ignore")

# try to import wandb
# try:
#     import wandb
#     has_wandb = True
# except:
#     has_wandb = False

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0")

logger = get_logger(__name__, log_level="INFO")

def get_activation(mem, name):
    def get_output_hook(module, input, output):
        mem[name] = output

    return get_output_hook

def add_hook(net, mem, mapping_layers):
    for n, m in net.named_modules():
        if n in mapping_layers:
            m.register_forward_hook(get_activation(mem, n))

########################################################################## MIIL ##############################################################################################3
def get_input_activation(mem, name):
    def get_input_hook(module, input):
        if isinstance(input, tuple):
            print(input)
        # input은 튜플일 수 있으므로 첫 번째 요소를 저장
        mem[name] = input[0] if isinstance(input, tuple) else input
        
        print(f"Hooking input for {name}: type(input) = {type(input)}")
        print(mem[name])
    return get_input_hook

def add_pre_hook(net, mem, mapping_layers):
    for n, m in net.named_modules():
        if n in mapping_layers:
            # forward_pre_hook을 사용하여 입력값을 후킹
            m.register_forward_pre_hook(get_input_activation(mem, n))
########################################################################## MIIL ##############################################################################################3


def copy_weight_from_teacher(unet_stu, unet_tea, student_type):

    connect_info = {} # connect_info['TO-student'] = 'FROM-teacher'
    if student_type in ["bk_base", "bk_small"]:
        connect_info['up_blocks.0.resnets.1.'] = 'up_blocks.0.resnets.2.'
        connect_info['up_blocks.1.resnets.1.'] = 'up_blocks.1.resnets.2.'
        connect_info['up_blocks.1.attentions.1.'] = 'up_blocks.1.attentions.2.'
        connect_info['up_blocks.2.resnets.1.'] = 'up_blocks.2.resnets.2.'
        connect_info['up_blocks.2.attentions.1.'] = 'up_blocks.2.attentions.2.'
        connect_info['up_blocks.3.resnets.1.'] = 'up_blocks.3.resnets.2.'
        connect_info['up_blocks.3.attentions.1.'] = 'up_blocks.3.attentions.2.'
    elif student_type in ["bk_tiny"]:
        connect_info['up_blocks.0.resnets.0.'] = 'up_blocks.1.resnets.0.'
        connect_info['up_blocks.0.attentions.0.'] = 'up_blocks.1.attentions.0.'
        connect_info['up_blocks.0.resnets.1.'] = 'up_blocks.1.resnets.2.'
        connect_info['up_blocks.0.attentions.1.'] = 'up_blocks.1.attentions.2.'
        connect_info['up_blocks.0.upsamplers.'] = 'up_blocks.1.upsamplers.'
        connect_info['up_blocks.1.resnets.0.'] = 'up_blocks.2.resnets.0.'
        connect_info['up_blocks.1.attentions.0.'] = 'up_blocks.2.attentions.0.'
        connect_info['up_blocks.1.resnets.1.'] = 'up_blocks.2.resnets.2.'
        connect_info['up_blocks.1.attentions.1.'] = 'up_blocks.2.attentions.2.'
        connect_info['up_blocks.1.upsamplers.'] = 'up_blocks.2.upsamplers.'
        connect_info['up_blocks.2.resnets.0.'] = 'up_blocks.3.resnets.0.'
        connect_info['up_blocks.2.attentions.0.'] = 'up_blocks.3.attentions.0.'
        connect_info['up_blocks.2.resnets.1.'] = 'up_blocks.3.resnets.2.'
        connect_info['up_blocks.2.attentions.1.'] = 'up_blocks.3.attentions.2.'       
    else:
        raise NotImplementedError


    for k in unet_stu.state_dict().keys():
        flag = 0
        k_orig = k
        for prefix_key in connect_info.keys():
            if k.startswith(prefix_key):
                flag = 1
                k_orig = k_orig.replace(prefix_key, connect_info[prefix_key])            
                break

        if flag == 1:
            print(f"** forced COPY {k_orig} -> {k}")
        else:
            print(f"normal COPY {k_orig} -> {k}")
        unet_stu.state_dict()[k].copy_(unet_tea.state_dict()[k_orig])

    return unet_stu

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--extra_text_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the extra text data for random conditioning."
        ),
    )    
    
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="score",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    
    parser.add_argument("--random_conditioning", action='store_true', help='perform condition sharing')
    parser.add_argument("--random_conditioning_lambda", type=float, default=5, help="condition share lambda")

    parser.add_argument("--use_unseen_setting", action='store_true', help='use unseen setting(train_data, eval_data)')

    
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=50000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument("--unet_config_path", type=str, default="./src/unet_config")     
    parser.add_argument("--unet_config_name", type=str, default="bk_small", choices=["bk_base", "bk_small", "bk_tiny","original"])   
    parser.add_argument("--lambda_sd", type=float, default=1.0, help="weighting for the denoising task loss")  
    parser.add_argument("--lambda_kd_output", type=float, default=1.0, help="weighting for output KD loss")  
    parser.add_argument("--lambda_kd_feat", type=float, default=1.0, help="weighting for feature KD loss")  
    parser.add_argument("--valid_steps", type=int, default=10000)
    parser.add_argument("--num_valid_images", type=int, default=2)
    parser.add_argument("--use_copy_weight_from_teacher", action="store_true", help="Whether to initialize unet student with teacher's weights",)
    parser.add_argument("--valid_prompt", type=str, default="a golden vase with different flowers")

    parser.add_argument("--valid_prompts", type=str, nargs='+', default=[
        "A cat crawling into a white toilet seat.",
        "a small elephant walks in a lush forest",
        "A couple of elephants walking across a river.",
        "A white horse looking through the window of a tall brick building.",
        "A dog with goggles is in a motorcycle side car.",
        "A large elephant walking next to a fallen tree.",
        "A striped cat curled up above a comforter.",
        "A cat laying bed in a small room.",
        "A horse with a blanket on it eating hay.",
        "Dog in bag with mesh and items around",
        "A young sexy woman holding a tennis racquet on a tennis court.",
        "The torso of a man who is holding a knife.",
        "a person that is standing up in a tennis court",
        "A man swinging a tennis racquet on a tennis court.",
        "a young man in a grey shirt is going to cut his hair",
        "a red fire hydrant near a dirt road with trees in the background",
        "a tennis player attempting to reach a tennis ball",
        "A bathroom with a tiled floor and a sink.",
        "Bride and grooms arms cutting the wedding cake with fruit on top.",
        "A bathroom with a brown shower curtain and white toilet"
    ])
    
    
    parser.add_argument("--channel_mapping", action="store_true", help="channel mapping",)
    
    
    parser.add_argument("--drop_text", action="store_true", help="distillation null text",)
    parser.add_argument("--drop_text_p", type=float, default=0.1, help="null text ratio",)
    
    # arguments for evaluation
    parser.add_argument("--model_id", type=str, default="nota-ai/bk-sdm-base", help="Path to the pretrained model or checkpoint directory.")
    parser.add_argument("--unet_path", type=str, default="/home/work/StableDiffusion/T2I_distill1_GPU4/results/toy_ddp_bk_base/checkpoint-25000", help="Model checkpoint for evaluate")
    parser.add_argument("--img_sz", type=int, default=512)
    parser.add_argument("--img_resz", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--batch_sz", type=int, default=10)    


    parser.add_argument("--data_list", type=str, default="./data/mscoco_val2014_30k/metadata.csv")
    parser.add_argument("--valid41k_dir", type=str, default="./data/mscoco_val2014_41k")
    

    parser.add_argument('--clip_device', type=str, default='cuda', help='Device to use, cuda or cpu')
    parser.add_argument('--clip_seed', type=int, default=1234, help='Random seed for reproducibility')
    parser.add_argument('--clip_batch_size', type=int, default=50, help='Batch size for processing images')

    parser.add_argument("--use_sd_loss", action="store_true", help="Whether to calculate sd_loss (denoising task loss).")
        
    args = parser.parse_args()
    
    args.save_dir = os.path.join(args.output_dir, "generated_images")
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
        
    run_id = uuid.uuid4().hex
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir, f"run_{run_id}")
    os.makedirs(logging_dir, exist_ok=True)
    
    wandb_dir = os.path.join(args.output_dir, "wandb_logs", f"run_{run_id}")
    os.makedirs(wandb_dir, exist_ok=True)
    
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    
    ipg_handler = InitProcessGroupKwargs(
            timeout=timedelta(seconds=5400)
            )
    
    accelerator = Accelerator(
        kwargs_handlers=[ipg_handler],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    
    world_size = accelerator.num_processes  
    local_rank = accelerator.local_process_index
    
    # Add custom csv logger and validation image folder
    val_img_dir = os.path.join(args.output_dir, 'val_img')
    os.makedirs(val_img_dir, exist_ok=True)


    csv_log_path = os.path.join(args.output_dir, 'log_loss.csv')
    print(csv_log_path)
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'step', 'global_step',
                                'loss_total', 'loss_sd', 'loss_kd_output', 'loss_kd_feat',
                                'lr', 'lamb_sd', 'lamb_kd_output', 'lamb_kd_feat'])

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed+accelerator.local_process_index)

    # Handle the repository creation
    if accelerator.is_main_process and (args.output_dir is not None):
        os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    ddpm_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler = DDIMScheduler.from_config(ddpm_scheduler.config)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    # Define teacher and student
    unet_teacher = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    # Freeze student's vae and text_encoder and teacher's unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_teacher.requires_grad_(False)


    train_dataset = x0_dataset(data_dir=args.train_data_dir, extra_text_dir=args.extra_text_dir,n_T=1000, 
                               random_conditioning=args.random_conditioning, random_conditioning_lambda=args.random_conditioning_lambda, 
                               world_size=world_size, rank=local_rank, drop_text=args.drop_text, drop_text_p=args.drop_text_p, 
                               use_unseen_setting=args.use_unseen_setting)

    with accelerator.main_process_first():
            if args.max_train_samples is not None:
                print("all:", len(train_dataset))
                indices = random.sample(range(len(train_dataset)), args.max_train_samples)
                train_dataset = Subset(train_dataset, indices)
                print("Subset:", len(train_dataset))

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn(tokenizer),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move student's text_encode and vae and teacher's unet to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet_teacher.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            "text2image-fine-tune",
            config=vars(args)
        )
        
    # Train!
    global_step = 0
    first_epoch = 0

    vae_scale_factor = getattr(vae.config, "scaling_factor", 0.18215)
    progress_bar = tqdm(
    range(global_step, args.max_train_steps), 
    initial=global_step,  # 초기값을 현재 global_step으로 설정
    total=args.max_train_steps,  # 전체 스텝을 max_train_steps로 설정
    disable=not accelerator.is_local_main_process,
    dynamic_ncols=True
)
    progress_bar.set_description("Steps")
    
    timesteps = noise_scheduler.timesteps
    
    epoch_losses = []

    for epoch in range(0, 25):
        # Initialize a list to store step losses for this epoch
        per_epoch_loss = []
        
        if args.seed is not None:
            set_seed(args.seed+accelerator.local_process_index)
        for step, batch in enumerate(train_dataloader):
            # Convert images to latents
            latents = batch["latents"].to(weight_dtype).to(accelerator.device)
            # latents = latents * vae_scale_factor  # Not needed since we're working with latents directly

            # Text encoding
            with torch.no_grad():
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]
                changed_hidden_states = text_encoder(batch["changed_ids"].to(accelerator.device))[0]

            # Generate initial noise
            noise = torch.randn_like(latents)

            # Ensure timesteps are set and on the correct device
            noise_scheduler.set_timesteps(args.num_inference_steps)
            timesteps = noise_scheduler.timesteps.to(accelerator.device)
            # print(timesteps)

            # Add initial noise to the latents at the current epoch's timestep
            current_timestep = timesteps[epoch]
            latents_noisy = noise_scheduler.add_noise(latents, noise, current_timestep)

            # Unconditional text encoding (empty text)
            with torch.no_grad():
                uncond_input = tokenizer(
                    [""] * latents.size(0),
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt"
                )
                uncond_embeddings = text_encoder(uncond_input.input_ids.to(accelerator.device))[0]

            # Combine unconditional and conditional embeddings
            prompt_embeds = torch.cat([uncond_embeddings, changed_hidden_states], dim=0)

            # Denoising loop starting from the current epoch's timestep
            latents_denoised = latents_noisy
            for i, t in enumerate(tqdm(timesteps[epoch:], disable=not accelerator.is_local_main_process)):
                # Expand latents for classifier-free guidance
                latents_input = torch.cat([latents_denoised] * 2)

                # Scale model input according to scheduler requirements
                latents_input = noise_scheduler.scale_model_input(latents_input, t)

                # Predict noise residual
                with torch.no_grad():
                    noise_pred = unet_teacher(latents_input, t, encoder_hidden_states=prompt_embeds).sample

                # Perform classifier-free guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                guidance_scale = 7.5
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute previous noisy sample x_t -> x_t-1
                latents_denoised = noise_scheduler.step(noise_pred, t, latents_denoised, eta=1.0).prev_sample

            # After denoising, compute score_loss

            # Initialize a variable to accumulate losses for this step
            step_loss = 0.0
            for i, t in enumerate(tqdm(timesteps, disable=not accelerator.is_local_main_process)):
                noise_score = torch.randn_like(latents_denoised)
                latents_input = noise_scheduler.add_noise(latents_denoised, noise_score, t)
                latents_input = noise_scheduler.scale_model_input(latents_input, t)
                with torch.no_grad():
                    noise_pred = unet_teacher(latents_input, t, encoder_hidden_states=encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred.float(), noise_score.float(), reduction="mean")
                # Accumulate the loss
                step_loss += loss.item()

            # Average the step loss over the number of timesteps
            avg_step_loss = step_loss / len(timesteps)
            per_epoch_loss.append(avg_step_loss)

            progress_bar.update(1)
            # latents_denoised = latents_denoised / vae_scale_factor
            # # Decode latents to images
            # with torch.no_grad():
            #     images = vae.decode(latents_denoised.to(weight_dtype)).sample

            # # Save images
            # save_image_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
            # os.makedirs(save_image_dir, exist_ok=True)
            
            # if accelerator.is_main_process:
            #     for idx, image in enumerate(images[0:2]):
            #         image = (image / 2 + 0.5).clamp(0, 1)
            #         image = image.cpu().permute(1, 2, 0).numpy()
            #         image = (image * 255).astype(np.uint8)
            #         image = Image.fromarray(image)
            #         image.save(os.path.join(save_image_dir, f"image_{step * args.train_batch_size + idx}.png"))
        # At the end of the epoch, compute the average loss over all steps
        avg_epoch_loss = sum(per_epoch_loss) / len(per_epoch_loss)
        epoch_losses.append(avg_epoch_loss)

        # Optionally, print the average loss for this epoch
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1}, Average Loss: {avg_epoch_loss:.6f}")

    # After all epochs, print or save the epoch losses
    if accelerator.is_main_process:
        print("Epoch losses:", epoch_losses)
        # Save the losses to a file
        import json
        with open(os.path.join(args.output_dir, 'epoch_losses:C.json'), 'w') as f:
            json.dump(epoch_losses, f, indent=4)

if __name__ == "__main__":
    main()