import os

from src import img_filtering as filt
from src import img_normalization as norm
from src import img_masking as im
from src import skeletonize_masks as sm

if __name__ == "__main__":
    metadata_path = "/path/to/metadata"
    input_dir = "/path/to/raw/image/stack"
    checkpoint = "/path/to/checkpoints/sam2.1_hiera_large.pt" # This can be exact or relative to sam2 repo
    cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" # Needs to be this path exactly; do not change

    filt_metadata = filt.filterImages(metadata_path, input_dir, output_dir="original_filtered")
    filt.renameFiles("original_filtered")

    norm.normalizeImages(input_dir="original_filtered", output_dir="normalized")
    norm.negatize(input_dir="normalized", output_dir="normalized_inverted")
    os.chdir("normalized_inverted")

    im.processFolder(".", checkpoint, cfg)
    df = sm.skeletonizedDataFrame(input_dir)

    df.join(metadata_path, on="ImageFileName")
    df["cycles"] = df["ImageFileName"].map(lambda x: x[:4]).astype(int) * 1000

    # df.to_csv("/choose/your/path/[test ID]_crack_data.csv", index=False)

# TODO:
# see if force profile exists? Could possibly correlate with ncycles,
# but would need force exerted as well (I think)