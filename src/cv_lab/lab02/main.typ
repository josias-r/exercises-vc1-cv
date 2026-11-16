#set page(height: auto)
#set page(width: auto)
#set page(margin: 0.5cm)

= A1

#box(width: 18cm)[
  I applied a simple median filter but ONLY for full black or full white pixels to preserve sharpness of the image, but still remove the salt and pepper noise. The filter is applied to each channel of the image separately.

  Small window size of 3x3 seems to be the smart choice for this task to keep changes minimal locally.
]

#stack(dir: ltr, image("lena_filtered.png"), image("lena_noise.png"))

```python
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
```

= A2

#box(width: 18cm)[
  I played around with different blur sizes etc. i ended up with box blur of 20x20 (images are 330x330) and used that same lp to get hp of earth.

  The effect works good enough for me. When unfocusing my eyes i blur the "high frequencies" of what i see only leaving the moon picture (since the high earth frequencies are only faint then)
]

#grid(
  columns: 2,
  image("a02/earoon.png", width: 8cm),
  block(width: 8cm)[
    #grid(
      columns: 2,
      image("a02/earth.png"), image("a02/moon.png"),
      image("a02/earth_hp.png"), image("a02/moon_lp.png"),
    )],
)

```python
earth = cv2.imread(f"{script_dir}/a02/earth.png")
moon = cv2.imread(f"{script_dir}/a02/moon.png")

kernel_box = np.ones((20, 20))
kernel_box = kernel_box / (kernel_box.shape[0] ** 2)

earth_lp = cv2.filter2D(earth, ddepth=-1, kernel=kernel_box)
earth_hp = np.clip(earth.astype(np.float32) - earth_lp.astype(np.float32), 0, 255).astype(np.uint8)

moon_lp = cv2.filter2D(moon, ddepth=-1, kernel=kernel_box)

cv2.imwrite(f"{script_dir}/a02/earth_hp.png", earth_hp)
cv2.imwrite(f"{script_dir}/a02/moon_lp.png", moon_lp)

earoon = np.clip(earth_hp.astype(np.float32) + moon_lp.astype(np.float32), 0, 255).astype(np.uint8)
cv2.imwrite(f"{script_dir}/a02/earoon.png", earoon)
```
