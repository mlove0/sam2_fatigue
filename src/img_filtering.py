import os
import shutil
import numpy as np
import pandas as pd


def filterImages(
    metadata_path: str,
    image_folder: str,
    output_dir: str = "autoselected_frames",
    z_thresh: float = 4.0,
    window: int = 25,
    min_separation: int = 10,
) -> pd.DataFrame:
    """
    Select frames whose 'FocusScore' exhibits a large drop (z-score > z_thresh) after a 2-frame shift,
    then copy just those images into `output_dir`.

    Returns a DataFrame (`keepers`) with the selected rows from metadata, including:
      - original metadata columns
      - 'frame_index' (original row index)
      - 'fsdiff', 'z_fsdiff'

    Notes on spacing:
      We enforce a minimum `min_separation` frames between selected frames to avoid adjacent picks.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata robustly
    try:
        df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("Metadata file is empty or unreadable.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error reading metadata: {e}")

    # Validate required columns
    required_cols = {"FocusScore", "ImageFileName"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in metadata: {missing}")

    # Preserve original row index for spacing checks
    df = df.reset_index().rename(columns={"index": "frame_index"})

    # 2-frame drawdown diff and z-score over rolling window
    df["fsdiff"] = df["FocusScore"].shift(2) - df["FocusScore"]
    ma = df["fsdiff"].rolling(window=window, min_periods=1, center=False).mean()
    sd = df["fsdiff"].rolling(window=window, min_periods=1, center=False).std().replace(0, np.nan)
    df["z_fsdiff"] = (df["fsdiff"] - ma) / sd
    df["z_fsdiff"] = df["z_fsdiff"].fillna(0)

    # Filter by z-score
    keepers = df[df["z_fsdiff"] > z_thresh].copy()

    # Enforce min separation between chosen frames
    keepers = keepers.sort_values("frame_index").copy()
    keepers = keepers[keepers["frame_index"].diff().fillna(np.inf) >= min_separation]
    keepers = keepers.reset_index(drop=True)

    # Copy images
    copied = 0
    for file in keepers["ImageFileName"]:
        src = os.path.join(image_folder, str(file))
        dst = os.path.join(output_dir, str(file))
        if not os.path.isfile(src):
            # Skip missing files but keep record
            continue
        shutil.copy2(src, dst)
        copied += 1

    if copied == 0:
        print("[filterImages] Warning: no files were copied (check paths and filters).")

    return keepers


def renameFiles(
    dir_path: str,
    start: int = 2,
    step: int = 2,
    pad: int = 4,
    suffix: str = ".png",
) -> pd.DataFrame:
    """
    Rename all files in `dir_path` to a numeric sequence:
        0002.png, 0004.png, 0006.png, ...
    (configurable via start/step/pad/suffix)

    To avoid name collisions, performs a two-pass rename via temporary names.

    Returns a DataFrame with columns:
        ['old_name', 'new_name']
    """
    files = sorted(
        [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    )

    # Build target names
    mapping = []
    counter = start
    for filename in files:
        new_filename = f"{counter:0{pad}d}{suffix}"
        mapping.append((filename, new_filename))
        counter += step

    # First pass: rename to temp names to avoid collisions
    temp_mapping = []
    for old, new in mapping:
        old_path = os.path.join(dir_path, old)
        temp_path = os.path.join(dir_path, f".tmp_{new}")
        os.rename(old_path, temp_path)
        temp_mapping.append((temp_path, new))

    # Second pass: rename from temp to final
    results = []
    for temp_path, new in temp_mapping:
        final_path = os.path.join(dir_path, new)
        os.rename(temp_path, final_path)
        results.append({"old_name": os.path.basename(temp_path).replace(".tmp_", ""), "new_name": new})

    return pd.DataFrame(results)


if __name__ == "__main__":
    input_path = "/path/to/image/stack"
    metadata_path = "path/to/metadata/file"

    filterImages(metadata_path, image_folder=input_path, output_dir="autoselected_frames")
    renameFiles(dir="autoselected_frames")
