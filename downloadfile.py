import kagglehub
import shutil
import os
# Download latest version
path = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")
print("Path to dataset files:", path)
target_dir = "C:\\Users\\dingr\\source\\repos"
target_path = os.path.join(target_dir, "CIC-IDS- 2017.csv")
os.makedirs(target_dir, exist_ok=True)
shutil.move(path, target_path)
print("Moved to:", target_path)