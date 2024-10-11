import os

folder_path = "data/laion_aes/x0_cache_212k/train"
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

if csv_files:
    print("CSV 파일이 존재합니다:", csv_files)
else:
    print("CSV 파일이 존재하지 않습니다.")
