import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", type=str, default="./metadata.parquet")
    parser.add_argument("--output_csv", type=str, default="./random_samples.csv")
    args = parser.parse_args()
    return args

def create_random_prompts_csv(metadata_path, output_csv, num_samples=500):
    # Load metadata from the parquet file
    metadata_df = pd.read_parquet(metadata_path)
    
    # Select only the 'image_name' and 'prompt' columns
    metadata_df = metadata_df[["image_name", "prompt"]]
    
    # Randomly sample `num_samples` rows from the dataset
    random_samples = metadata_df.sample(n=num_samples, random_state=42)
    
    # Save the sampled rows to a CSV file
    random_samples.to_csv(output_csv, index=False, header=True)
    print(f"Random samples saved to {output_csv}")

if __name__ == "__main__":
    args = parse_args()
    create_random_prompts_csv(args.metadata_path, args.output_csv)


# import pandas as pd
# import numpy as np
# import csv
# import argparse

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--metadata_path", type=str, default="./data/mscoco_val2014_41k/metadata_mscoco41k_unseen.csv")
#     parser.add_argument("--output_csv", type=str, default="./cherry.csv")
#     args = parser.parse_args()
#     return args

# def create_random_prompts_csv(metadata_path, output_csv, num_samples=500):
#     # Load metadata from the parquet file
#     metadata_df = pd.read_csv(metadata_path)
    
#     # Randomly sample 500 prompts from the dataset
#     random_prompts = metadata_df.sample(n=num_samples, random_state=42)
    
#     # Save the sampled prompts to a CSV file
#     random_prompts.to_csv(output_csv, index=False, header=True)
#     print(f"Random prompts saved to {output_csv}")

# if __name__ == "__main__":
#     args = parse_args()
#     create_random_prompts_csv(args.metadata_path, args.output_csv)

# import pandas as pd
# import argparse

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--metadata_path", type=str, default="./data/mscoco_val2014_30k/metadata.csv", help="Path to the metadata CSV file")
#     parser.add_argument("--output_csv", type=str, default="./cherry_mscoco.csv", help="Path to save the sampled CSV")
#     parser.add_argument("--num_samples", type=int, default=500, help="Number of random samples to select")
#     args = parser.parse_args()
#     return args

# def create_random_prompts_csv(metadata_path, output_csv, num_samples=500):
#     # Load metadata from the CSV file
#     metadata_df = pd.read_csv(metadata_path)
    
#     # Randomly sample 'num_samples' prompts from the dataset
#     random_samples = metadata_df.sample(n=num_samples, random_state=42)
    
#     # Save the sampled prompts to a CSV file
#     random_samples.to_csv(output_csv, index=False)
#     print(f"Random samples saved to {output_csv}")

# if __name__ == "__main__":
#     args = parse_args()
#     create_random_prompts_csv(args.metadata_path, args.output_csv, args.num_samples)


# import os
# from PIL import Image
# from torchvision.utils import make_grid, save_image
# import torch
# from torchvision import transforms

# # Define the paths
# base_dir = "./qualititive_results_mscoco"
# subfolders = ["teacher", "bksdm-base", "bksdm-tiny", "ch-small4"]
# output_dir = os.path.join(base_dir, "Grid")
# os.makedirs(output_dir, exist_ok=True)

# # Ensure all subfolders exist
# for folder in subfolders:
#     folder_path = os.path.join(base_dir, folder)
#     if not os.path.exists(folder_path):
#         raise FileNotFoundError(f"Subfolder {folder_path} does not exist")

# # Iterate over the image indices
# num_images = 500
# for i in range(num_images):
#     images = []
#     for folder in subfolders:
#         img_path = os.path.join(base_dir, folder, f"{i}.jpg")
#         if not os.path.exists(img_path):
#             raise FileNotFoundError(f"Image {img_path} does not exist")
#         img = Image.open(img_path)
#         images.append(transforms.ToTensor()(img))
#         img.close()
    
#     # Create a grid image from the four images
#     imgs_tensor = torch.stack(images)
#     grid_img = make_grid(imgs_tensor, nrow=4, padding=2, normalize=True)
    
#     # Save the grid image
#     grid_img_path = os.path.join(output_dir, f"grid_{i}.jpg")
#     save_image(grid_img, grid_img_path)
#     print(f"Grid image saved at {grid_img_path}")
