from tiatoolbox import data
import os

# Directory to save the downloaded WSI
save_dir = '/home/path02/python_envs/ImpartLabs/tmp/input'
os.makedirs(save_dir, exist_ok=True)

# URL to download the sample WSI
url = "https://tiatoolbox.dcs.warwick.ac.uk/models/slide_graph/cell-composition/TCGA-C8-A278-01Z-00-DX1.188B3FE0-7B20-401A-A6B7-8F1798018162.svs"

# Download and save the WSI
wsi_path = os.path.join(save_dir, 'sample.svs')
data.download_data(url, wsi_path)

print(f"WSI saved at: {wsi_path}")
