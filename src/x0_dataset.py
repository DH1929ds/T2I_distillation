import os
import math
import torch
import pandas as pd
from torch.utils.data import Dataset

class x0_dataset(Dataset):
    def __init__(self, data_dir, n_T=1000, cond_sharing = False, cond_share_lambda=5):
        """
        Args:
            data_dir (str): 데이터가 저장된 폴더의 경로.
        """
        self.data_dir = data_dir
        self.n_T = n_T
        self.cond_sharing = cond_sharing
        self.cond_share_lambda = cond_share_lambda
        
        # 메타데이터 CSV 파일 로드
        metadata_path = os.path.join(data_dir, "metadata.csv")
        self.metadata = pd.read_csv(metadata_path)
        
        # 폴더 내에 존재하는 파일들의 이름을 불러옵니다.
        # 각 인덱스에 맞는 데이터들이 있는지 확인하기 위해 파일 이름의 공통 부분을 추출합니다.
        self.data_indices = self._get_data_indices()

    def _get_data_indices(self):
        """
        데이터 디렉토리에서 인덱스에 맞는 파일을 확인하고, 유효한 인덱스 목록을 반환합니다.
        """
        # 폴더 내의 파일 리스트를 가져오고, 인덱스를 추출
        files = os.listdir(self.data_dir)
        indices = set()

        for file_name in files:
            # 파일 이름에서 인덱스를 추출 ('0_latent.pt', '0.txt' 등의 형식을 가정)
            base_name = file_name.split("_")[0]
            if base_name.isdigit():  # 숫자 인덱스가 존재할 때만 추가
                indices.add(int(base_name))

        return sorted(list(indices))  # 정렬된 인덱스 목록 반환

    def __len__(self):
        # 유효한 인덱스 개수를 반환합니다.
        return len(self.data_indices)

    def __getitem__(self, idx):
        # 인덱스에 해당하는 데이터 파일들을 로드하여 반환합니다.
        index = self.data_indices[idx]

        # 각 파일의 경로를 설정
        latent_path = os.path.join(self.data_dir, f"{index}_latent.pt")
        text_emb_path = os.path.join(self.data_dir, f"{index}_text_emb.pt")
        
        timestep = torch.randint(0, self.n_T, (1,)).long()
        
        if self.cond_sharing:
            t_value = timestep.item()
            p = math.exp(-self.cond_share_lambda * (1 - t_value / self.n_T))
            if torch.rand(1).item() < p:
                rand_index = torch.randint(0, len(self.data_indices), (1,)).item()
                text_emb_path = os.path.join(self.data_dir, f"{rand_index}_text_emb.pt")

        latent_tensor = torch.load(latent_path)
        text_embedding_tensor = torch.load(text_emb_path)

        return latent_tensor, text_embedding_tensor, timestep


def collate_fn(batch):
    latents, text_embs, timesteps = zip(*batch)
    
    latent_tensors = torch.stack(latents)
    text_embedding_tensors = torch.stack(text_embs)
    timesteps = torch.cat(timesteps)  # 타임스텝도 배치로 결합

    return {
        "latents": latent_tensors,
        "text_embs": text_embedding_tensors,
        "timesteps": timesteps   # timesteps 추가
    }