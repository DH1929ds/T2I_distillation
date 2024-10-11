import os
import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np

# VAE 모델 로드
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", revision=None)
device = "cuda" if torch.cuda.is_available() else "cpu"
vae.to(device)

# 이미지 전처리 및 디코딩 함수 정의
def load_and_decode(latent_path, vae, device):
    # Latent 파일 불러오기
    latent = torch.load(latent_path).to(device)

    # Latent가 배치 차원을 포함하는지 확인
    if len(latent.shape) == 3:
        latent = latent.unsqueeze(0)  # 배치 차원이 없는 경우 추가

    # VAE 디코딩 (디코딩 시 스케일링 팩터 적용)
    latents = (1 / 0.18215) * latent
    with torch.no_grad():
        decoded = vae.decode(latents).sample

    # 이미지를 [0, 1] 범위로 조정
    decoded = (decoded / 2 + 0.5).clamp(0, 1)

    # 이미지를 numpy 배열로 변환하고 [0, 255] 범위로 조정
    decoded_img = (decoded.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

    # numpy 배열을 PIL 이미지로 변환
    pil_image = Image.fromarray(decoded_img)

    return pil_image

# 예시: latent 파일 경로와 저장 경로 설정
latent_path = "data/laion_aes/pt_cache_212k/000027947_latent.pt"  # 불러올 latent pt 파일 경로
output_image_path = "decoded_image.png"  # 저장할 이미지 파일 경로

# Latent 파일을 디코딩하고 이미지를 저장
decoded_image = load_and_decode(latent_path, vae, device)
decoded_image.save(output_image_path)
print(f"Decoded image saved to {output_image_path}")