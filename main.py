import os
import pandas as pd

from src import img_filtering as filt
from src import img_normalization as norm
from src import img_masking as im
from src import skeletonize_masks as sm


def main():
    # === USER: set these ===
    metadata_path = "/path/to/metadata.csv"  # CSV with 'ImageFileName' and 'FocusScore' columns
    input_dir = "/path/to/raw/image/stack"   # directory of raw images
    checkpoint = "/path/to/checkpoints/sam2.1_hiera_large.pt"  # SAM2 checkpoint
    cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"                 # must remain this relative path for SAM2

    # === 1) Filter frames by focus-score events, copy to filtered dir ===
    filtered_dir = "original_filtered"
    keepers = filt.filterImages(metadata_path, image_folder=input_dir, output_dir=filtered_dir)

    # Rename filtered files to canonical sequence (0002.png, 0004.png, ...)
    rename_map = filt.renameFiles(filtered_dir)  # DataFrame: old_name, new_name

    # Update keepers' filenames to reflect renames for downstream merge
    keepers_renamed = keepers.merge(
        rename_map, left_on="ImageFileName", right_on="old_name", how="left"
    )
    keepers_renamed["ImageFileName"] = keepers_renamed["new_name"].fillna(keepers_renamed["ImageFileName"])
    keepers_renamed = keepers_renamed.drop(columns=["old_name", "new_name"], errors="ignore")

    # === 2) Normalize -> Invert ===
    ## NOTE: Inversion is optional. Comment out the norm.negatize line if you don't want inverted images.
    ## Just don't forget to change the output directory name if you choose to do that
    normalized_dir = os.path.join(filtered_dir, "normalized")
    norm.normalizeImages(input_dir=filtered_dir, output_dir=normalized_dir)

    inverted_dir = os.path.join(normalized_dir, "inverted")
    norm.negatize(input_dir=normalized_dir, output_dir=inverted_dir)

    # === 3) Segment with SAM2; save best IoU mask per image (same filename) ===
    im.processFolder(inverted_dir, checkpoint, cfg)  # creates 'masked' and 'transparencies' subdirs

    # === 4) Crack metrics from transparency masks ===
    transparencies_dir = os.path.join(inverted_dir, "transparencies")
    metrics_df = sm.skeletonizedDataFrame(transparencies_dir)

    # === 5) Merge with metadata (left-join on standardized ImageFileName) ===
    # If you also want *all* original metadata columns, re-read metadata CSV and merge:
    try:
        meta_df = pd.read_csv(metadata_path)
    except Exception:
        meta_df = pd.DataFrame()

    # Merge metrics with the filtered/renamed subset to carry over any relevant columns
    df = metrics_df.merge(keepers_renamed, on="ImageFileName", how="left", suffixes=("", "_meta"))

    # Optional: also merge original metadata CSV on filename if it still matches naming
    if not meta_df.empty and "ImageFileName" in meta_df.columns:
        df = df.merge(meta_df, on="ImageFileName", how="left", suffixes=("", "_origmeta"))

    # === 6) Derive cycles from filename (assumes first 4 digits), *1000 like original code ===
    # e.g. "0002.png" -> 2*1000 -> 2000
    cycles_extracted = df["ImageFileName"].str.extract(r"(\d{4})")
    df["cycles"] = cycles_extracted[0].astype(float).fillna(0).astype(int) * 1000

    # === 7) Save results ===
    out_csv = os.path.join(inverted_dir, "crack_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"[main] Saved: {out_csv}")


if __name__ == "__main__":
    main()
