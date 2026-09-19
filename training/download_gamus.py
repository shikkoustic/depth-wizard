"""
DepthWizard - Real Hugging Face GAMUS Dataset Downloader.
Downloads official paired Optical Satellite RGB (.h5) and LiDAR Heights (.h5)
from https://huggingface.co/datasets/earthflow/GAMUS
"""

import os
import requests
import h5py
import numpy as np
from PIL import Image

HF_BASE_URL = "https://huggingface.co/datasets/earthflow/GAMUS/resolve/main"

# Selected real Washington DC & Philadelphia urban & sparse LiDAR tiles from earthflow/GAMUS
TRAIN_SAMPLES = [
    "DC_01_25", "DC_02_24", "DC_02_25", "DC_02_27", "DC_03_23",
    "DC_03_24", "DC_03_25", "DC_03_27", "DC_03_28", "DC_04_24"
]

TEST_SAMPLES = [
    "DC_03_26", "DC_05_28", "DC_05_30", "DC_07_21", "DC_07_29"
]


def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return True
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    resp = requests.get(url, stream=True)
    if resp.status_code == 200:
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return True
    else:
        print(f"Failed to download {url} (status {resp.status_code})")
        return False


def fetch_gamus_dataset(target_dir: str = "data/gamus", num_train: int = 6, num_test: int = 3):
    print("=" * 70)
    print(" Downloading Official GAMUS Dataset from Hugging Face (earthflow/GAMUS)")
    print("=" * 70)

    # 1. Download Train Samples
    train_dir = os.path.join(target_dir, "train")
    for sample_id in TRAIN_SAMPLES[:num_train]:
        rgb_url = f"{HF_BASE_URL}/images/train/{sample_id}_RGB.h5"
        agl_url = f"{HF_BASE_URL}/heights/train/{sample_id}_AGL.h5"

        rgb_path = os.path.join(train_dir, f"{sample_id}_RGB.h5")
        agl_path = os.path.join(train_dir, f"{sample_id}_AGL.h5")

        print(f"Downloading Train Sample {sample_id}...")
        download_file(rgb_url, rgb_path)
        download_file(agl_url, agl_path)

    # 2. Download Test Samples
    test_dir = os.path.join(target_dir, "test")
    for sample_id in TEST_SAMPLES[:num_test]:
        rgb_url = f"{HF_BASE_URL}/images/test/{sample_id}_RGB.h5"
        agl_url = f"{HF_BASE_URL}/heights/test/{sample_id}_AGL.h5"

        rgb_path = os.path.join(test_dir, f"{sample_id}_RGB.h5")
        agl_path = os.path.join(test_dir, f"{sample_id}_AGL.h5")

        print(f"Downloading Test Sample {sample_id}...")
        download_file(rgb_url, rgb_path)
        download_file(agl_url, agl_path)

        # Convert test sample to GeoTIFF and JPG for immediate 3D visualization
        convert_to_sample_data(rgb_path, agl_path, sample_id)

    print("\n[Download Complete] GAMUS dataset downloaded into:", target_dir)


def convert_to_sample_data(rgb_h5: str, agl_h5: str, sample_id: str):
    """Converts a real GAMUS test pair into sample_data/ for immediate 3D visualization."""
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample_data"))
    os.makedirs(out_dir, exist_ok=True)

    try:
        import tifffile
        with h5py.File(rgb_h5, "r") as f_rgb:
            rgb = np.array(f_rgb["image"])
        with h5py.File(agl_h5, "r") as f_agl:
            agl = np.array(f_agl["image"]).astype(np.float32)

        # Save high-res optical JPEG
        jpg_name = os.path.join(out_dir, f"gamus_{sample_id}_rgb.jpg")
        Image.fromarray(rgb).save(jpg_name, quality=92)

        # Save real LiDAR GeoTIFF
        lidar_name = os.path.join(out_dir, f"gamus_{sample_id}_lidar.tif")
        tifffile.imwrite(lidar_name, agl)

        print(f" -> Exported real GAMUS test tile: {jpg_name}")
    except Exception as e:
        print(f"Error converting {sample_id}: {e}")


if __name__ == "__main__":
    fetch_gamus_dataset()
