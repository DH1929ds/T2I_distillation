

import os
import argparse
import time
import csv
from PIL import Image
import torch
from utils.inference_pipeline import InferencePipeline
from torchvision import transforms
from torchvision.utils import make_grid, save_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    #parser.add_argument("--model_id", type=str, default="nota-ai/bk-sdm-tiny")
    
    parser.add_argument("--save_dir", type=str, default="./qualtitive_results_unseen_seen",
                        help="$save_dir/grid is created for saving the grid image")
    parser.add_argument("--unet_path", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_sz", type=int, default=512)
    parser.add_argument("--batch_sz", type=int, default=25)
    parser.add_argument("--unet_config", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default="./random_samples.csv")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    from diffusers import UNet2DConditionModel
    import torch
    from torchvision.utils import make_grid, save_image
    import os
    import time

    args = parse_args()
    
    pipeline = InferencePipeline(weight_folder=args.model_id,
                                 seed=args.seed,
                                 device=args.device)
    pipeline.set_pipe_and_generator()
    
    # Load UNet from a pretrained model if a path is provided
    if args.unet_path is not None:
        unet = UNet2DConditionModel.from_pretrained(args.unet_path, subfolder='unet')
        # Set the UNet in the pipeline
        pipeline.pipe.unet = unet.half().to(args.device)
    
    
    # Read prompts from CSV file
    with open(args.csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        prompts = [row[1] for row in reader]

    # # Read prompts from CSV file
    # with open(args.csv_path, 'r') as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     prompts = [row['text'] for row in reader]
    
    #Ensure we have 500 prompts
    # prompts = prompts[:1000]
    # assert len(prompts) == 1000, "The CSV file must contain at least 500 prompts."
    
    t0 = time.perf_counter()
    img_save_dir = os.path.join(args.save_dir, args.model_name)
    os.makedirs(img_save_dir, exist_ok=True)
    # Generate images in batches and save them
    for batch_start in range(0, len(prompts), args.batch_sz):
        batch_end = batch_start + args.batch_sz
        batch_prompts = prompts[batch_start:batch_end]

        batch_imgs = pipeline.generate(prompt=batch_prompts,
                                       n_steps=args.num_inference_steps,
                                       img_sz=args.img_sz)

        for i, img in enumerate(batch_imgs):
            img_index = batch_start + i
            img_save_path = os.path.join(img_save_dir, f"{img_index}.jpg")
            img.save(img_save_path)
            img.close()
            print(f"Image saved at {img_save_path}")
    
    pipeline.clear()
    print(f"{(time.perf_counter() - t0):.2f} sec elapsed")









# # ------------------------------------------------------------------------------------
# # Copyright 2023–2024 Nota Inc. All Rights Reserved.
# # ------------------------------------------------------------------------------------
# # "A cocker spaniel with its ears blowing in the wind during a car ride" - seed42


# import os
# import argparse
# import time
# from PIL import Image
# import torch
# from utils.inference_pipeline import InferencePipeline
# from torchvision import transforms
# from torchvision.utils import make_grid, save_image

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
#     #parser.add_argument("--model_id", type=str, default="nota-ai/bk-sdm-base")
#     parser.add_argument("--save_dir", type=str, default="./qualititive_results",
#                         help="$save_dir/grid is created for saving the grid image")
#     parser.add_argument("--unet_path", type=str, default=None)
#     parser.add_argument("--num_inference_steps", type=int, default=25)
#     parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--img_sz", type=int, default=512)
#     parser.add_argument("--batch_sz", type=int, default=1)
#     parser.add_argument("--unet_config", type=str, default="./src/unet_config_channel_small_4")
#     parser.add_argument("--model_name", type=str, default=None)

#     # parser.add_argument("--valid_prompts", type=str, nargs='+', default=[
#     #     "Two cakes sitting on a class table near a candle",
#     #     "The fresh fruit is left out on the counter",
#     #     "A couple of buses that are lined up by some buildings",
#     #     "A bicycle with a basket and a blue and pink umbrella",
#     #     "A kitchen with a stove top oven next to a white fridge.",
#     #     "A desk with a laptop, computer monitor, keyboard and mouse",
#     #     "A wooden and metal bench near a over grown bush",
#     #     "Two men in suits and ties next to plant",
#     #     "A woman standing next to the ocean flying a colorful kite",
#     #     "A baseball player wearing a leather glove standing in the dirt",
#     #     "A bird flies over a large body of water",
#     #     "A cat sitting underneath a blue and white umbrella",
#     #     "A dog jumps in order to catch a ring",
#     #     "A horse drawn carriage driving down a small road",
#     #     "A sheep standing on top of a rock",
#     #     "A brown cow walking through a tree filled forest",
#     #     "An elephant is carrying some plants in its tusk",
#     #     "A bear peaks out behind the grass in the woods",
#     #     "A zebra rolling on its back on the ground",
#     #     "A giraffe standing next to a tall tree"
#     # ])
# #     parser.add_argument("--valid_prompts", type=str, nargs='+', default=[
# #     "A small dog running through a lush green field with a ball in its mouth",
# #     "A playful puppy sitting next to a pile of autumn leaves",
# #     "Two dogs playing tug-of-war with a rope in the backyard",
# #     "A dog lying in the sun on a cozy porch",
# #     "A group of dogs gathered at a park, playing fetch with their owners",
# #     "A dog wearing a raincoat on a rainy day, jumping over puddles",
# #     "A fluffy dog curled up next to a fireplace",
# #     "A dog riding in the back of a pickup truck, tongue out, enjoying the breeze",
# #     "A dog dressed up for Halloween as a superhero standing on a doorstep",
# #     "A border collie herding sheep in a field",
# #     "A labrador retriever swimming in a lake retrieving a stick",
# #     "A poodle performing tricks at a dog show",
# #     "A beagle sniffing around in a forest during a hike",
# #     "A dachshund peeking out from under a blanket on a sofa",
# #     "A golden retriever waiting patiently at the crosswalk of a busy street",
# #     "A bulldog skateboarding down a city sidewalk",
# #     "A schnauzer barking at a squirrel in a tree",
# #     "A terrier digging a hole in a sandy beach",
# #     "A cocker spaniel with its ears blowing in the wind during a car ride",
# #     "A chihuahua in a sweater sitting in a cafe"
# # ])
#     parser.add_argument("--valid_prompts", type=str, nargs='+', default=[
#     "A giraffe eating leaves from the top of an acacia tree",
#     "A giraffe family walking across the savanna at sunset",
#     "Two giraffes standing side by side, their necks intertwined",
#     "A baby giraffe standing close to its mother",
#     "A giraffe bending its long neck to drink water from a pond",
#     "A giraffe silhouetted against the orange sky at dusk",
#     "A giraffe walking gracefully across the African plains",
#     "A giraffe eating leaves while birds perch on its back",
#     "A group of giraffes with their heads above the treetops",
#     "A giraffe peeking curiously over a tall fence",
#     "A giraffe's long tongue reaching out to grab leaves",
#     "A giraffe calmly chewing leaves while zebras graze nearby",
#     "A pair of giraffes running through the open savanna",
#     "A giraffe bending awkwardly to reach grass on the ground",
#     "A close-up of a giraffe's face with its large eyes and long eyelashes",
#     "A giraffe resting in the shade of a large tree",
#     "A giraffe walking past a safari jeep with tourists taking photos",
#     "A group of giraffes resting together under the hot sun",
#     "A giraffe looking down curiously at a small animal on the ground",
#     "A giraffe stretching its neck to reach the tallest branches"
#     ])




#     args = parser.parse_args()
#     return args

# if __name__ == "__main__":
#     from diffusers import UNet2DConditionModel
#     import torch
#     from torchvision.utils import make_grid, save_image
#     import os
#     import time

#     args = parse_args()
    
#     pipeline = InferencePipeline(weight_folder=args.model_id,
#                                  seed=args.seed,
#                                  device=args.device)
#     pipeline.set_pipe_and_generator()
    
   
#     # Load UNet from a pretrained model if a path is provided
#     unet = UNet2DConditionModel.from_pretrained(args.unet_path, subfolder='unet')
#     # Set the UNet in the pipeline
#     pipeline.pipe.unet = unet.half().to(args.device)
 
#     # Use the valid prompts from args
#     val_prompts = args.valid_prompts
#     assert len(val_prompts) == 20, "The valid prompts list must contain exactly 20 prompts."
    
#     t0 = time.perf_counter()
    
#     # Generate images in batches
#     imgs = []
#     for batch_start in range(0, len(val_prompts), args.batch_sz):
#         batch_end = batch_start + args.batch_sz
#         batch_prompts = val_prompts[batch_start:batch_end]

#         batch_imgs = pipeline.generate(prompt=batch_prompts,
#                                        n_steps=args.num_inference_steps,
#                                        img_sz=args.img_sz)
#         imgs.extend(batch_imgs)

#         for i, val_prompt in enumerate(batch_prompts):
#             print(f"{batch_start + i}/{len(val_prompts)} | {val_prompt}")

#     # Create a grid image of size 2 rows x 10 columns using make_grid from torchvision
#     assert len(imgs) == 20, "Expected 20 generated images."
#     imgs_tensor = torch.stack([transforms.ToTensor()(img) for img in imgs])
#     grid_img = make_grid(imgs_tensor, nrow=10, padding=2, normalize=True)

#     # Save the grid image
#     if args.model_name is not None:
#         img_save_dir = os.path.join(args.save_dir, args.model_name)
#         #print(img_save_dir)
#         os.makedirs(img_save_dir, exist_ok=True)  # makedir -> makedirs로 수정
#     grid_img_path = os.path.join(img_save_dir,'grid_image.png')
#     save_image(grid_img, grid_img_path)
#     print(f"Grid image saved at {grid_img_path}")

#     pipeline.clear()
#     print(f"{(time.perf_counter() - t0):.2f} sec elapsed")


