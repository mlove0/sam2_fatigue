import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def skeletonizedDataFrame(input_dir: str) -> pd.DataFrame:
    """
    Compute per-image crack metrics from RGBA transparency masks (alpha channel > 0 => mask).

    For each image:
      - Extract alpha channel (expects 4-channel PNG/TIFF)
      - For each x column where mask exists, compute min(y), max(y), median(y)
      - Aggregate to per-image:
          * avg_crack_width_pixels = mean(max_y - min_y) across x
          * crack_length_pixels    = max(x) - min(x)
          * mask_area_pixels       = number of masked pixels

    Returns DataFrame with one row per image:
        ['ImageFileName', 'avg_crack_width_pixels', 'crack_length_pixels', 'mask_area_pixels']
    """
    records = []

    for fname in tqdm(sorted(os.listdir(input_dir))):
        if not fname.lower().endswith((".png", ".tif", ".tiff")):
            continue

        path = os.path.join(input_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim < 3 or img.shape[2] < 4:
            # Not RGBA; skip
            continue

        alpha = img[:, :, 3]
        y_coords, x_coords = np.where(alpha > 0)
        if x_coords.size == 0:
            # Empty mask; skip but keep a record with zeros if desired
            continue

        # Per-x statistics using pandas for clarity
        df_xy = pd.DataFrame({"x": x_coords, "y": y_coords})
        stats = df_xy.groupby("x")["y"].agg(["min", "max", "median"]).reset_index()
        y_diff = (stats["max"] - stats["min"]).astype(float)

        avg_width = float(y_diff.mean())
        crack_len_proj = int(stats["x"].max() - stats["x"].min())
        area = int(y_coords.size)

        records.append(
            {
                "ImageFileName": fname,
                "avg_crack_width_pix": avg_width,
                "crack_length_proj_pix": crack_len_proj,
                "mask_area_pix": area,
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    input_dir = "/path/to/mask/transparencies"
    skel = skeletonizedDataFrame(input_dir)
