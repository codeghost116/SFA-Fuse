import random
from typing import Tuple

import cv2
import numpy as np
import torch
from scipy.fftpack import dct
from torchvision import transforms

SPATIAL_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

SINGLE_CHANNEL_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

srm_filter_kernel = (
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 2.0, -1.0, 0.0],
            [0.0, 2.0, -4.0, 2.0, 0.0],
            [0.0, -1.0, 2.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    / 4.0
)


def apply_low_res_augmentations(frame_bgr: np.ndarray) -> np.ndarray:
    frame = frame_bgr

    if random.random() < 0.5:
        h, w, _ = frame.shape
        scale = random.uniform(0.3, 0.7)
        low_res_h, low_res_w = int(h * scale), int(w * scale)
        lr_frame = cv2.resize(frame, (low_res_w, low_res_h), interpolation=cv2.INTER_AREA)
        frame = cv2.resize(lr_frame, (w, h), interpolation=cv2.INTER_LINEAR)

    if random.random() < 0.5:
        quality = random.randint(40, 75)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _result, encimg = cv2.imencode(".jpg", frame, encode_param)
        frame = cv2.imdecode(encimg, 1)

    return frame


def _dct_frequency_map(gray: np.ndarray) -> np.ndarray:
    dct_map = np.zeros_like(gray, dtype=np.float32)

    for i in range(0, gray.shape[0], 8):
        for j in range(0, gray.shape[1], 8):
            block = gray[i : i + 8, j : j + 8]

            if block.shape != (8, 8):
                continue

            dct_block = dct(dct(block.T, norm="ortho").T, norm="ortho")
            dct_map[i : i + 8, j : j + 8] = dct_block

    dct_map = np.log(np.abs(dct_map) + 1e-9)
    dct_map = np.clip(dct_map, -10, 10)

    return dct_map


def _noise_residual_map(gray: np.ndarray) -> np.ndarray:
    noise_map = cv2.filter2D(gray.astype(np.float32), -1, srm_filter_kernel)
    noise_map = np.clip(noise_map, -3, 3)
    return noise_map


def get_tri_stream_inputs(
    frame: np.ndarray,
    input_format: str = "rgb",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_format.lower() == "bgr":
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_rgb = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    spatial_input = SPATIAL_TRANSFORM(frame_rgb)

    dct_map = _dct_frequency_map(frame_gray)
    frequency_input = SINGLE_CHANNEL_TRANSFORM(dct_map)

    noise_map = _noise_residual_map(frame_gray)
    noise_input = SINGLE_CHANNEL_TRANSFORM(noise_map)

    return spatial_input, frequency_input, noise_input
