import os
import argparse
import time
import torch
from accelerate import Accelerator
from tqdm import tqdm
from utils.inference_pipeline import InferencePipeline
from utils.misc import get_file_list_from_csv, change_img_size
import contextlib
import sys

def sample_images_30k(args, accelerator, save_path=None):
    device = accelerator.device

    pipeline = InferencePipeline(weight_folder=args.model_id, seed=args.clip_seed, device=device)
    pipeline.set_pipe_and_generator()
    
    pipeline.pipe.set_progress_bar_config(disable=True)

    if save_path is not None:  # use a separate trained unet for generation        
        from diffusers import UNet2DConditionModel 
        unet = UNet2DConditionModel.from_pretrained(save_path, subfolder='unet')
        pipeline.pipe.unet = unet.half().to(device)
        accelerator.print(f"** load unet from {save_path}")        

    save_dir_src = os.path.join(args.save_dir, f'im{args.img_sz}')  # for model's raw output images
    save_dir_tgt = os.path.join(args.save_dir, f'im{args.img_resz}')  # for resized images for benchmark

    # Create directories only on the main process
    if accelerator.is_main_process:
        os.makedirs(save_dir_src, exist_ok=True)
        os.makedirs(save_dir_tgt, exist_ok=True)

    accelerator.wait_for_everyone()

    file_list = get_file_list_from_csv(args.data_list)
    total_files = len(file_list)
    num_processes = accelerator.num_processes
    rank = accelerator.process_index

    # Distribute files evenly among ranks without leaving any unprocessed files
    files_per_process = total_files // num_processes
    remainder = total_files % num_processes

    if rank < remainder:
        start_index = rank * (files_per_process + 1)
        end_index = start_index + files_per_process + 1
    else:
        start_index = remainder * (files_per_process + 1) + (rank - remainder) * files_per_process
        end_index = start_index + files_per_process

    # Get the list of files to process for this rank
    process_file_list = file_list[start_index:end_index]

    # tqdm progress bar setup for rank 0 only
    if accelerator.is_main_process:
        progress_bar = tqdm(total=len(process_file_list), desc="Generating Images", disable=not accelerator.is_main_process)

    # Process the assigned files in batches
    for batch_start in range(0, len(process_file_list), args.batch_sz):
        batch_end = min(batch_start + args.batch_sz, len(process_file_list))

        img_names = [file_info[0] for file_info in process_file_list[batch_start:batch_end]]
        val_prompts = [file_info[1] for file_info in process_file_list[batch_start:batch_end]]

        # Suppress output of pipeline.generate
        with contextlib.redirect_stdout(sys.stderr), contextlib.redirect_stderr(sys.stderr):
            imgs = pipeline.generate(prompt=val_prompts,
                                     n_steps=args.num_inference_steps,
                                     img_sz=args.img_sz)

        for img, img_name in zip(imgs, img_names):
            img.save(os.path.join(save_dir_src, img_name))
            img.close()

        # Update progress bar only on rank 0
        if accelerator.is_main_process:
            progress_bar.update(1)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        change_img_size(save_dir_src, save_dir_tgt, args.img_resz)
        progress_bar.close()
        accelerator.print(f"Image generation completed and resized images saved.")

    pipeline.clear()
    torch.cuda.empty_cache()
