import os
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader, Dataset

# Dataset to handle txt files and their corresponding content
class TxtDataset(Dataset):
    def __init__(self, txt_dir):
        self.txt_dir = txt_dir
        self.txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

    def __len__(self):
        return len(self.txt_files)

    def __getitem__(self, idx):
        txt_file = self.txt_files[idx]
        with open(os.path.join(self.txt_dir, txt_file), 'r') as f:
            text = f.read().strip()
        return text, txt_file

# Function to generate images from the given dataset
def generate_images(pipeline, dataloader, save_dir, device):
    generator = torch.Generator(device=device).manual_seed(1234)
    for batch in dataloader:
        texts, txt_files = batch
        for i, text in enumerate(texts):
            image = pipeline(text, num_inference_steps=25, generator=generator).images[0]
            img_name = txt_files[i].replace('.txt', '.jpg')
            image.save(os.path.join(save_dir, img_name))

# Main function
def main():
    accelerator = Accelerator() 
    device = accelerator.device

    # Define directories
    txt_dir = "./data/laion_aes/train"  # Directory where txt files are located
    save_dir = "./data/laion_aes/x0_cache_212k/train"  # Directory where generated images will be saved
    os.makedirs(save_dir, exist_ok=True)

    # Initialize dataset and dataloader
    dataset = TxtDataset(txt_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load the Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        safety_checker=None,
    )
    pipeline = pipeline.to(device)

    # Use accelerator to prepare dataloader
    pipeline, dataloader = accelerator.prepare(pipeline, dataloader)

    # Generate and save images based on txt file content
    generate_images(pipeline, dataloader, save_dir, device)

if __name__ == "__main__":
    main()
