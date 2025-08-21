import os
import cv2
import numpy as np
from skimage import exposure, img_as_ubyte
from tqdm import tqdm


def _list_images(d: str) -> list[str]:
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    return sorted([f for f in os.listdir(d) if f.lower().endswith(exts) and os.path.isfile(os.path.join(d, f))])


def normalizeImages(
    input_dir: str,
    output_dir: str = "normalized",
    clip_limit: float = 3.5,
    percentiles: tuple[float, float] = (3.0, 99.8),
) -> None:
    """
    For each image in `input_dir`:
      - Read as GRAYSCALE
      - CLAHE (clipLimit=clip_limit)
      - Contrast stretching to [p_low, p_high] percentiles
      - Save as PNG with same basename in `output_dir` (created if missing)
    """
    os.makedirs(output_dir, exist_ok=True)
    files = _list_images(input_dir)

    for fname in tqdm(files, desc="Normalizing", unit="img"):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.png")

        try:
            image = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"[normalizeImages] Skipping non-image or unreadable file: {fname}")
                continue

            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit)
            clahe_img = clahe.apply(image)

            # Contrast stretching
            p_low, p_high = np.percentile(clahe_img, percentiles)
            img_rescale = exposure.rescale_intensity(clahe_img, in_range=(p_low, p_high))
            img_rescale_u8 = img_as_ubyte(img_rescale)

            cv2.imwrite(out_path, img_rescale_u8)

        except Exception as e:
            print(f"[normalizeImages] Error processing {fname}: {e}")


def negatize(input_dir: str, output_dir: str = "inverted") -> None:
    """
    Create photographic negatives for all images in `input_dir` and write to `output_dir`.
    Works with grayscale or color images.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = _list_images(input_dir)

    for fname in tqdm(files, desc="Inverting", unit="img"):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        try:
            img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[negatize] Skipping non-image or unreadable file: {fname}")
                continue

            img_neg = 255 - img
            cv2.imwrite(out_path, img_neg)

        except Exception as e:
            print(f"[negatize] Error processing {fname}: {e}")


if __name__ == "__main__":
    dir = "/path/to/image/stack"
    normalizeImages(dir)
    negatize("path/to/image/stack/normalized")
