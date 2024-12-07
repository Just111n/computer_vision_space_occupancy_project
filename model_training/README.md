# Model Training
This directory facilitates model training and dataset building.

```bash
cd model_training
```

## How to Train Your Model

### Step 1: Building the dataset
The `build_dataset.py` script can be run to generate training, validation and testing datasets.

Each dataset (train/val/test) consists of images, corresponding masks and class mappings.

**By default**, the script will attempt to build the datasets from `../mit_indoor/` directory, which contains image and annotation data for image segmentation and object detection tasks.

For the purpose of this project, the script will also default to extracting data from the `meeting_room` sub-directory.

The `../mit_indoor/` directory (or any input data directory) is expected to have the following structure:

```text
mit_indoor/
|
├─  Images/
|   ├─ .../
|   ├─ meeting_room/
|   ├─ .../
|
├─  Annotations/
|   ├─ .../
|   ├─ meeting_room/
|   ├─ .../
|
|
model_training/
|
├─  build_dataset.py
├─  README.md
├─  data/
├─  ...
```

To build default dataset:

```bash
python build_dataset.py 
```

To build a custom dataset:

```bash
python build_dataset.py --src <src_dir_path> --sub <sub_dir_name>
```

The script will store the generated datasets in the `./data` directory. The directory will have the following structure:

```text
model_training/
|
├─  build_dataset.py
├─  README.md
├─  ...
├─  data/
|   |
|   ├─ images/
|   |   ├─ train/
|   |   ├─ val/
|   |   ├─ test/
|   |
|   ├─ mappings/
|   |   ├─ train/
|   |   ├─ val/
|   |
|   ├─ masks/
|   |   ├─ train/
|   |   ├─ val/
```

### Step 2: Train the Model
TBC