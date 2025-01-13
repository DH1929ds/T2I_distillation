import os
import torch
import pandas as pd
from safetensors.torch import save_file

def process_metadata_and_tensors(metadata_path, data_dir, output_dir):
    """
    메타데이터를 읽고, 각 텍스트 항목과 연결된 텐서를 로드 및 처리한 후 저장.

    Args:
        metadata_path (str): 메타데이터 CSV 파일 경로.
        data_dir (str): 텐서 파일들이 저장된 디렉토리 경로.
        output_dir (str): 처리된 텐서를 저장할 디렉토리 경로.
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 메타데이터 읽기
    metadata = pd.read_csv(metadata_path)

    for idx, row in metadata.iterrows():
        file_name = row['file_name']
        text = row['text']

        # 텐서 파일 경로
        latent_file_name = file_name.replace('.jpg', '_latent.pt')
        latent_path = os.path.join(data_dir, latent_file_name)

        if os.path.exists(latent_path):
            try:
                # 텐서 로드
                tensor = torch.load(latent_path, map_location=torch.device('cpu'))

                # 텐서 속성 제거 및 정리
                tensor = tensor.detach().contiguous()

                # Safetensors 형식으로 저장
                output_file = os.path.join(output_dir, latent_file_name.replace('.pt', '.safetensors'))
                save_file({"tensor": tensor}, output_file)

                print(f"Processed and saved: {output_file} (Text: {text})")

            except Exception as e:
                print(f"Failed to process {latent_path}: {e}")
        else:
            print(f"Tensor file not found for: {file_name}")

# 사용 예시
metadata_path = "./data/laion_aes/SDXL_latent_212k/SDXL_base_latents/metadata.csv"  # 메타데이터 파일 경로
data_dir = "./data/laion_aes/SDXL_latent_212k/SDXL_base_latents"  # 텐서 파일 디렉토리 경로
output_dir = "./data/laion_aes/SDXL_latent_212k/SDXL_base_latents_safetensors"  # 출력 디렉토리 경로

process_metadata_and_tensors(metadata_path, data_dir, output_dir)
