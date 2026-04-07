from pathlib import Path

import h5py

import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# --- SETTINGS ---
CSV_PATH = Path("./.datasets/grass/train.csv")
IMG_DIR = Path("./.datasets/grass/")
OUTPUT_H5 = Path("./.datasets/grass/Cell200_64x64.h5")
IMG_SIZE = 64
PRIMARY_TARGET = "Dry_Total_g"

g = Path("./.datasets/grass")
print(g.exists())

# 1. Pivot the CSV
print("Pivoting dataset...")
df_long = pd.read_csv(CSV_PATH)
df_wide = df_long.pivot_table(
    index=['image_path'],
    columns='target_name',
    values='target'
).reset_index()

biomass_cols = df_wide.select_dtypes(include=[np.number]).columns
df_wide[biomass_cols] = np.floor(df_wide[biomass_cols])
# df_wide[biomass_cols] = int(df_wide[biomass_cols])

df_wide.to_csv(".datasets/grass/wide.csv")

# 2. Process Images and Labels
imgs_list = []
labels_list = []

print(f"Packing images (Squashing to {IMG_SIZE}x{IMG_SIZE})...")
for _, row in tqdm(df_wide.iterrows(), total=len(df_wide)):
    img_full_path = Path(IMG_DIR / row['image_path'])

    if not img_full_path.exists():
        print(f"warning image not found {img_full_path}")
        continue

    # Direct squash: 2000x1000 -> 64x64
    # This will compress the horizontal axis by 2x more than the vertical
    img = Image.open(img_full_path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

    imgs_list.append(np.array(img))
    labels_list.append(row[PRIMARY_TARGET])

# 3. Convert to Numpy
imgs_array = np.array(imgs_list).reshape(-1, 1, IMG_SIZE, IMG_SIZE).astype(np.uint8)
labels_array = np.array(labels_list).astype(np.float32)

# 4. Save to H5
with h5py.File(OUTPUT_H5, 'w') as f:
    f.create_dataset('IMGs_grey', data=imgs_array)
    f.create_dataset('CellCounts', data=labels_array)

print(f"Done! Created {OUTPUT_H5} with {len(labels_array)} samples.")

print("Grabbing highest and lowest values")
# print(df_wide[PRIMARY_TARGET])
print("max = ", np.max(df_wide[PRIMARY_TARGET]))
print("min = ", np.min(df_wide[PRIMARY_TARGET]))