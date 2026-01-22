import cv2
import numpy as np
import torch

def fft_magnitude(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    return magnitude


def noise_residual(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return img - blur


def build_tri_stream(img):
    rgb = img
    fft = fft_magnitude(img)
    noise = noise_residual(img)
    return rgb, fft, noise
