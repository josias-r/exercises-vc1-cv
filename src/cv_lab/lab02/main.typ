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

  I aligned the images before processing them ofc inside an external application (affnity, da best).
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

= A3
= A3

#box(width: 18cm)[
  I played around with basic gradient filters (Sobel and Laplacian) and compared them to the Canny edge detector. The basic filters use 3x3 kernels to approximate derivatives, finding rapid intensity changes (Sobel X for vertical edges, Sobel Y for horizontal, and Laplacian for all directions at once).

  Canny is definitely the winner for this one. It runs a whole multi-stage algorithm behind the scenes: smoothing out noise with a Gaussian blur, finding gradients, thinning the edges down (non-maximum suppression), and finally linking the strong and weak edges together using double thresholding and hysteresis tracking.

  The basic filters are fast but ended up looking pretty thick and noisy. Canny gives clean, continuous, and thin outlines that actually capture the coin's structure perfectly.
]

#grid(
  columns: 2,
  figure(image("a03/canny.png", width: 8cm), caption: [Canny Edge Detector]),
  block(width: 8cm)[
    #grid(
      columns: 2,
      row-gutter: 0.2cm,
      figure(image("a03/coin.jpeg"), caption: [Original]), figure(image("a03/laplacian.png"), caption: [Laplacian]),
      figure(image("a03/sobel_x.png"), caption: [Sobel X]), figure(image("a03/sobel_y.png"), caption: [Sobel Y]),
    )],
)

```python
coin = cv2.imread(f"{script_dir}/a03/coin.jpeg")
scale = 0.25
coin_sm = cv2.resize(coin, dsize=None, fx=scale, fy=scale)
coin_gray = cv2.cvtColor(coin_sm, cv2.COLOR_BGR2GRAY)
ddepth = cv2.CV_16S

# 1. Basic Gradient Filters (Sobel and Laplacian)
coin_x = cv2.Sobel(coin_gray, ddepth, 1, 0, ksize=3)
coin_y = cv2.Sobel(coin_gray, ddepth, 0, 1, ksize=3)
coin_laplacian = cv2.Laplacian(coin_gray, ddepth, ksize=3)

abs_x = cv2.convertScaleAbs(coin_x)
abs_y = cv2.convertScaleAbs(coin_y)
abs_laplacian = cv2.convertScaleAbs(coin_laplacian)

cv2.imwrite(f"{script_dir}/a03/sobel_x.png", abs_x)
cv2.imwrite(f"{script_dir}/a03/sobel_y.png", abs_y)
cv2.imwrite(f"{script_dir}/a03/laplacian.png", abs_laplacian)

# 2. Canny Edge Detector
coin_blurred = cv2.GaussianBlur(coin_gray, (5, 5), 0)
edges_canny = cv2.Canny(coin_blurred, 100, 200)

cv2.imwrite(f"{script_dir}/a03/canny.png", edges_canny)
```

= Reflection

#box(width: 18cm)[
  These exercises really showed me how filters directly mess with image frequencies. Blurs (low-pass) keep broad shapes but kill sharp details, while gradients (high-pass) strip away the smooth areas to leave only the edges.

  There is always a trade-off between reducing noise and keeping things sharp. Selectively filtering (like my median trick in A1) fixes noise without blurring the whole picture. Otherwise, Canny (A3) handles the trade-off well -> it blurs the noise away first, then thins the lines back down so the structure stays crisp.
]
