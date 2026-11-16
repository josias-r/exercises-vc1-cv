import cv2
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

script_dir = Path(__file__).parent
lena_noise = cv2.imread(f"{script_dir}/lena_noise.png")

def filter_sp_rm(pixels: np.array):
    pixels_center = (pixels.shape[0] // 2, pixels.shape[1] // 2)
    centerp = pixels[pixels_center]
    if centerp == 255 or centerp == 0:
        return np.median(np.ravel(pixels))
    return centerp

def apply_sliding_w_filter(img: np.array, filter, window_shape=(3,3)):
    windows = sliding_window_view(img, window_shape=window_shape)

    imgNew = np.zeros(shape=(windows.shape[0], windows.shape[1]), dtype=np.uint8)
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            window = windows[i][j]
            imgNew[i][j] = filter(window)
    return imgNew

b, g, r = cv2.split(lena_noise)

lena_filtered = cv2.merge(
    [apply_sliding_w_filter(b, filter_sp_rm),
    apply_sliding_w_filter(g, filter_sp_rm),
    apply_sliding_w_filter(r, filter_sp_rm)]
)

cv2.imwrite(f"{script_dir}/lena_filtered.png", lena_filtered)
