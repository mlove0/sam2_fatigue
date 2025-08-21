import os
import gc
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def applyMaskRGB(image: np.ndarray, mask: np.ndarray, mask_color=(255, 0, 0), alpha: float = 0.6) -> np.ndarray:
    """
    Overlay a semi-transparent color on top of `image` where `mask` is True.
    image: RGB uint8 (H, W, 3)
    mask: bool or 0/1 (H, W)
    """
    mask = mask.astype(bool)
    overlay = image.copy()
    overlay[mask] = (alpha * np.array(mask_color) + (1 - alpha) * image[mask]).astype(np.uint8)
    return overlay


def saveRgbaTransparency(mask: np.ndarray, mask_color, out_path: str, alpha: int = 255) -> None:
    """
    Save an RGBA image where RGB = mask_color for mask==True, alpha = `alpha` (0..255), transparent elsewhere.
    """
    mask = mask.astype(bool)
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3][mask] = np.array(mask_color, dtype=np.uint8)
    rgba[..., 3][mask] = alpha
    Image.fromarray(rgba).save(out_path)


def _extreme_points(image_u8: np.ndarray, mode: str, num_points: int, threshold: int | None) -> np.ndarray:
    """
    Pick `num_points` most extreme pixels by intensity (brightest or darkest) given a threshold.
    Returns Nx2 array of (x, y) coordinates as float32 for SAM2.

    mode: 'bright' or 'dark'
    threshold: for 'bright' keep pixels >= threshold; for 'dark' keep <= threshold. If None, auto-pick.
    """
    if image_u8.ndim != 2:
        raise ValueError("Expected grayscale image for _extreme_points")

    img = image_u8
    H, W = img.shape

    if mode == "bright":
        thr = 250 if threshold is None else threshold
        ys, xs = np.where(img >= thr)
        scores = img[ys, xs]  # higher is better
        order = np.argsort(scores)[::-1]
    elif mode == "dark":
        thr = 50 if threshold is None else threshold
        ys, xs = np.where(img <= thr)
        scores = img[ys, xs]  # lower is better
        order = np.argsort(scores)  # ascending
    else:
        raise ValueError("mode must be 'bright' or 'dark'")

    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Select top-K
    k = min(num_points, order.size)
    sel = order[:k]
    xs_sel = xs[sel].astype(np.float32)
    ys_sel = ys[sel].astype(np.float32)

    # Stack as (x, y) for SAM2
    return np.stack([xs_sel, ys_sel], axis=1)


def processFolder(
    folder: str,
    checkpoint: str,
    cfg: str,
    device: str | None = None,
    num_pos: int = 10,
    num_neg: int = 20,
    bright_thr: int = 250,
    dark_thr: int = 150,
    mask_color=(255, 0, 0),
    alpha: float = 0.6,
) -> None:
    """
    Run SAM2 on all PNG images in `folder`. For each image:
      - Compute positive clicks from BRIGHT pixels (since upstream 'negatize' makes cracks bright)
      - Compute negative clicks from DARK pixels
      - Predict multiple masks, pick the best-IoU mask
      - Save:
          * RGB overlay -> {folder}/masked/<same filename>
          * RGBA transparency -> {folder}/transparencies/<same filename>
    """
    subdir_m = os.path.join(folder, "masked")
    subdir_t = os.path.join(folder, "transparencies")
    os.makedirs(subdir_m, exist_ok=True)
    os.makedirs(subdir_t, exist_ok=True)

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"[img_masking] Using device: {device}")

    model = build_sam2(cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)

    for fname in tqdm(sorted(os.listdir(folder))):
        if not fname.lower().endswith(".png"):
            continue

        try:
            path = os.path.join(folder, fname)

            # Work in grayscale for click selection
            gray = np.array(Image.open(path).convert("L"), dtype=np.uint8)

            # Build prompts: pos = bright, neg = dark (because images were inverted earlier)
            pos_pts = _extreme_points(gray, mode="bright", num_points=num_pos, threshold=bright_thr)
            neg_pts = _extreme_points(gray, mode="dark", num_points=num_neg, threshold=dark_thr)

            # If no points found, skip
            if pos_pts.shape[0] == 0 and neg_pts.shape[0] == 0:
                print(f"[img_masking] No prompts generated for {fname}; skipping.")
                continue

            # Concatenate clicks/labels for SAM2
            point_coords = np.concatenate([pos_pts, neg_pts], axis=0) if neg_pts.size else pos_pts
            point_labels = np.concatenate([
                np.ones((pos_pts.shape[0],), dtype=np.int32),
                np.zeros((neg_pts.shape[0],), dtype=np.int32),
            ]) if neg_pts.size else np.ones((pos_pts.shape[0],), dtype=np.int32)

            # SAM2 expects RGB
            rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            predictor.set_image(rgb_image)

            # Predict multiple masks, keep the best IoU
            masks, iou_scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            best_idx = int(np.argmax(iou_scores))
            best_mask = masks[best_idx].astype(bool)

            # Save overlay and RGBA transparency using the SAME base filename
            base = os.path.splitext(fname)[0]
            out_m = os.path.join(subdir_m, f"{base}.png")
            out_t = os.path.join(subdir_t, f"{base}.png")

            colored = applyMaskRGB(rgb_image, best_mask, mask_color=mask_color, alpha=alpha)
            cv2.imwrite(out_m, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
            saveRgbaTransparency(best_mask, mask_color, out_t, alpha=int(alpha * 255))

            # Cleanup
            del gray, masks
            gc.collect()

        except Exception as e:
            print(f"[img_masking] Error processing {fname}: {e}")


if __name__ == "__main__":
    IMAGE_FOLDER = "/home/mlove/Sandia/Au 3 stills/manual/cropped"
    CHECKPOINT = "/home/mlove/Sandia/sam2_fatigue/sam2_repo/checkpoints/sam2.1_hiera_large.pt"
    CFG = "configs/sam2.1/sam2.1_hiera_l.yaml" # Needs to be relative path; breaks when using absolute

    processFolder(IMAGE_FOLDER, CHECKPOINT, CFG)
