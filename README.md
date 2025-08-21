# SAM 2 for Nanoscale Fatigue Experiments

**NOTE:** The code in this repository is meant to be run on, and has only been tested on, Linux machines (specifically Ubuntu LTS 24.04). SAM 2 will not work on Windows, but its use on Mac machines is unknown.

This repository provides a complete pipeline for analyzing nanoscale fatigue experiments using **Meta’s SAM 2 (Segment Anything Model 2)**.
It integrates raw fatigue cycle images, metadata, and segmentation masks to extract quantitative measures of crack growth such as **crack length** and **crack width**.

The pipeline is modularized into preprocessing, segmentation, and postprocessing steps, orchestrated by the `main.py` script.

---

## Installation and Setup

1. Clone this repository to your local machine by opening a terminal instance and typing:
   ```bash
   git clone https://github.com/sam2_fatigue.git
   cd sam2_fatigue
   ```

2. (Optional but recommended) Create and activate a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3. Install required Python dependencies:
    ```bash
    pip install -r requirements.txt
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
   - [procedure for adding positional metadata — to be completed]
   - Export the data as a PNG stack.
   - Make sure to check the “Export Metadata” box before exporting, as this is not enabled by default.

3. **Edit configuration**:

   Open the file `main.py` and update the following variables:
   - `metadata_path` → path to the exported experiment metadata file
   - `input_dir` → path to the raw exported image stack
   - `checkpoint` → path to the SAM 2 checkpoint file (e.g., `sam2.1_hiera_large.pt`)
   - `cfg` → must remain exactly as `configs/sam2.1/sam2.1_hiera_l.yaml` within the SAM 2 repo
   - Inside the script, adjust the `df.to_csv()` call to set the desired output path for your CSV.

4. **Run the pipeline**:

   From the repository root directory, run:
   ```bash
   python3 main.py
   ```
    Processing may take a significant amount of time if running on CPU.

5. **Output**:

   The script generates a CSV containing quantitative crack growth data, aligned with experiment metadata.
