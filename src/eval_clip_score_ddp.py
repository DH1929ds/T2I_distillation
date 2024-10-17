import os
import argparse
import torch
import open_clip
from PIL import Image
from utils.misc import get_file_list_from_csv
from accelerate import Accelerator

def evaluate_clip_score(args, accelerator):
    # Set seed for reproducibility
    torch.manual_seed(args.clip_seed)
    
    device = accelerator.device

    # Load model and tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14',
                                                                 pretrained='laion2b_s34b_b88k',
                                                                 device=device)
    tokenizer = open_clip.get_tokenizer('ViT-g-14')

    # Load the list of image paths and corresponding prompts
    file_list = get_file_list_from_csv(args.data_list)

    # Distribute file_list among processes manually
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    num_files = len(file_list)
    files_per_rank = num_files // world_size
    start_idx = rank * files_per_rank
    end_idx = start_idx + files_per_rank if rank != world_size - 1 else num_files
    local_file_list = file_list[start_idx:end_idx]

    score_arr = []
    print("#############################################################################")
    print(args.img_dir)
    print("#############################################################################")

    for batch_start in range(0, len(local_file_list), args.clip_batch_size):
        batch_end = min(batch_start + args.clip_batch_size, len(local_file_list))
        batch_files = local_file_list[batch_start:batch_end]

        img_paths = [os.path.join(args.img_dir, file_info[0]) for file_info in batch_files]
        val_prompts = [file_info[1] for file_info in batch_files]
        texts = tokenizer(val_prompts).to(device)

        images = [preprocess(Image.open(img_path)).unsqueeze(0) for img_path in img_paths]
        images = torch.cat(images).to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity score
        probs = (text_features @ image_features.T).diagonal().cpu().numpy()
        score_arr.extend(probs)

        if batch_start % (args.clip_batch_size * 100) == 0:
            accelerator.print(f"Rank {rank} | Processed {batch_start}/{len(local_file_list)} images")
            accelerator.print(f"Rank {rank} | {batch_start}/{len(local_file_list)}| probs {probs[0]}")

    print(f"score_arr: {len(score_arr)}, rank: {device}\n")
    score_arr_tensor = torch.tensor(score_arr, device=device)

    accelerator.wait_for_everyone()  # 모든 프로세스가 작업을 마칠 때까지 대기  
    all_scores = accelerator.gather(score_arr_tensor).tolist()
    accelerator.wait_for_everyone()  # 모든 프로세스가 작업을 마칠 때까지 대기
    print(f"all_scores: {len(all_scores)}, rank: {device}\n")

    # Save results (only on process 0)
    if accelerator.is_main_process:
        final_score = sum(all_scores) / len(all_scores)
        with open(args.save_txt, 'w') as f:
            f.write(f"FINAL clip score {final_score}\n")
            f.write(f"-- sum score {sum(all_scores)}\n")
            f.write(f"-- len {len(all_scores)}\n")