import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from features import get_tri_stream_inputs


class FFPPFrameDataset(Dataset):
    def __init__(self, data_root_dir: str | Path, augment: bool = False):
        self.data_root = Path(data_root_dir)
        self.augment = augment
        self.samples: List[Tuple[Path, int]] = []  # (img_path, label)

        real_dir = self.data_root / "Real"
        fake_dir = self.data_root / "Fake"

        def collect_from(parent_dir: Path, label: int) -> int:
            if not parent_dir.exists():
                return 0
            count = 0
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                for img_path in parent_dir.rglob(pattern):
                    self.samples.append((img_path, label))
                    count += 1
            return count

        real_count = collect_from(real_dir, 0)
        fake_count = collect_from(fake_dir, 1)

        print(f"[FFPP] Loaded {real_count} real and {fake_count} fake frames from {self.data_root}")
        print(f"[FFPP] Total frames: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def _low_res_augment(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Low-resolution augmentation:
        downscale then upscale to simulate blur/compression.
        """
        if random.random() < 0.5:
            return img_rgb

        h, w, _ = img_rgb.shape
        scale = random.choice([0.5, 0.6, 0.7])
        new_w = max(8, int(w * scale))
        new_h = max(8, int(h * scale))

        small = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return restored

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            # if a frame is bad, sample another index
            new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.augment:
            rgb = self._low_res_augment(rgb)

        spatial_input, freq_input, noise_input = get_tri_stream_inputs(rgb)
        label_tensor = torch.tensor([float(label)], dtype=torch.float32)

        return spatial_input, freq_input, noise_input, label_tensor


class CelebDFImageDataset(Dataset):
    def __init__(self, data_root_dir: str | Path, augment: bool = False):
        self.data_root = Path(data_root_dir)
        self.augment = augment

        self.image_files: List[Path] = []
        self.labels: List[int] = []

        possible_real_folders = [("Celeb-real", 0), ("Real", 0), ("real", 0)]
        possible_fake_folders = [("Celeb-synthesis", 1), ("Fake", 1), ("fake", 1)]

        def load_images_from_path(folder_name: str, label: int) -> int:
            folder_path = self.data_root / folder_name
            if not folder_path.is_dir():
                return 0

            count = 0
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                for img_path in folder_path.glob(ext):
                    self.image_files.append(img_path)
                    self.labels.append(label)
                    count += 1
            return count

        # real
        real_count_total = 0
        real_found = False
        for folder, label in possible_real_folders:
            c = load_images_from_path(folder, label)
            if c > 0:
                real_found = True
                real_count_total += c
                print(f"[CelebDF] Found real images in: {folder} ({c})")
                break

        # fake
        fake_count_total = 0
        fake_found = False
        for folder, label in possible_fake_folders:
            c = load_images_from_path(folder, label)
            if c > 0:
                fake_found = True
                fake_count_total += c
                print(f"[CelebDF] Found fake images in: {folder} ({c})")
                break

        if not real_found:
            print(f"[CelebDF] Warning: no real images found in {self.data_root}")
        if not fake_found:
            print(f"[CelebDF] Warning: no fake images found in {self.data_root}")

        print(f"[CelebDF] Total images: {len(self.image_files)}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _low_res_augment(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Low-resolution augmentation:
        downscale then upscale to simulate blur/compression.
        """
        if random.random() < 0.5:
            return img_rgb

        h, w, _ = img_rgb.shape
        scale = random.choice([0.5, 0.6, 0.7])
        new_w = max(8, int(w * scale))
        new_h = max(8, int(h * scale))

        small = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return restored

    def __getitem__(self, idx: int):
        image_path = self.image_files[idx]
        label = self.labels[idx]

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            # fallback to a black image if corrupted
            bgr = np.zeros((256, 256, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.augment:
            rgb = self._low_res_augment(rgb)

        spatial_input, freq_input, noise_input = get_tri_stream_inputs(rgb)
        label_tensor = torch.tensor([float(label)], dtype=torch.float32)

        return spatial_input, freq_input, noise_input, label_tensor


def tri_collate_fn(batch):
    spatial_list, freq_list, noise_list, label_list = [], [], [], []

    for spatial, freq, noise, label in batch:
        spatial_list.append(spatial)
        freq_list.append(freq)
        noise_list.append(noise)
        label_list.append(label)

    spatial_batch = torch.stack(spatial_list)
    freq_batch = torch.stack(freq_list)
    noise_batch = torch.stack(noise_list)
    label_batch = torch.stack(label_list).view(-1, 1)

    return (spatial_batch, freq_batch, noise_batch), label_batch
