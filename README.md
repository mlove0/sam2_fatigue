# SAM 2 for Nanoscale Fatigue Experiments

**NOTE:** The code in this repository is meant to be run on, and has only been tested on, Linux machines (specifically Ubuntu LTS 24.04) using Conda as the Python installation medium. SAM 2 will not work on Windows unless installed on Windows Subsystem for Linux (WSL).

This repository provides a complete pipeline for analyzing nanoscale fatigue experiments using **Meta’s SAM 2 (Segment Anything Model 2)**.
It integrates raw fatigue cycle images, metadata, and segmentation masks to extract quantitative measures of crack growth such as **crack length** and **crack width**.

The pipeline is separated into preprocessing, segmentation, postprocessing, and analysis steps, orchestrated by the `main.py` script.

---

## Installation and Setup

1. Ensure Python >= 3.10 is installed using Conda.

2. Clone this repository to your local machine by opening a terminal instance and typing:
   ```bash
   git clone https://github.com/sam2_fatigue.git
   cd sam2_fatigue
   ```

3. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   conda create -f environment.yml
   conda activate sam2_tem
   ```

4. Install SAM 2 by following the instructions in [Meta's SAM 2 repository](https://github.com/facebookresearch/sam2):
    - Clone the repo as a subdirectory of this repository.
    - Rename the folder to something other than sam2 (e.g. sam2_repo) to avoid import conflicts.
    Your final directory structure should look like this:
    ```
    sam2_fatigue
    ├── sam2_repo       # Cloned from Meta
    │   └── model contents
    ├── src
    │   ├── img_filtering.py
    │   ├── img_masking.py
    │   ├── img_normalization.py
    │   └── skeletonize_masks.py
    ├── main.py
    ├── requirements.txt
    └── README.md
    ```

---

## Procedure

1. **Prepare data**:

   After conducting an experiment, open the data in Axon Studio.
   - To the right of the image, click the "Image Metadata" tab.
      - In the "Focus Assist" section, check the box next to "Focus Score".
      - In the "Image" section, check the boxes next to "Image Size X", "Image Size Y", "Image Pixels X", "Image Pixels Y"
      - In the "Position" section, check the boxes next to "Stage X", "Stage 
   - Export the data as a PNG stack by clicking the "Export/Publish" button to the lower left of the image.
      - Make sure to check the “Metadata (CSV)” box before choosing an export location and clicking "Publish", as this is not enabled by default.

2. **Edit configuration**:

   Open the file `main.py` and update the following variables:
   - `metadata_path` → path to the exported experiment metadata file
   - `input_dir` → path to the raw exported image stack
   - `checkpoint` → path to the SAM 2 checkpoint file (e.g., `sam2.1_hiera_large.pt`)
   - `cfg` → must remain exactly as `configs/sam2.1/sam2.1_hiera_l.yaml` within the SAM 2 repo
   - Inside the script, adjust the `df.to_csv()` call to set the desired output path for your CSV.

3. **Run the pipeline**:

   From the repository root directory, run:
   ```bash
   python3 main.py
   ```
    Processing may take a significant amount of time if running on CPU.

4. **Output**:

   The script generates a CSV containing quantitative crack growth data, aligned with experiment metadata.

---
## How It Works

This pipeline is separated into four stages, managed by ```main.py```:

1. **Image Filtering** (```img_filtering.py```)
   - Retains only images captured immediately after fatigue cycles end.
   - Renames images for consistency.

2. **Image Normalization** (```img_normalization.py```)
   - Converts images to grayscale.
   - Enhances contrast using CLAHE and contrast stretching.
   - Optionally creates negative images for better segmentation.

3. **Mask Generation** (```img_masking.py```)
   - Runs SAM 2 on the normalized images.
   - Produces segmentation masks highlighting cracks.

4. **Skeletonization & Measurement** (```skeletonize_masks.py```)
   - Converts crack masks into line skeletons.
   - Extracts median crack paths, widths, and lengths.
   - Outputs results as a clean Pandas DataFrame.

---

## Notes
- Ensure you have sufficient compute resources; running on GPU is \*highly\* recommended.
   - The minimum version of Torch (2.7.1) required by SAM 2 requires CUDA 12.1 or newer. If running this project on a GPU machine, ensure that your CUDA installation is up to date.
- The pipeline assumes cracks are the darkest regions in the images (assuming you choose to segment inverted images), so preprocessing (inversion/CLAHE) is tuned for that.
- Further analysis can be added downstream using the exported CSV.
