import json
from pathlib import Path
import random
from monai.apps.auto3dseg import (
    AutoRunner,
)
from monai.config import print_config
import os

print_config()


# This lets MLflow use a local folder for tracking training runs. Auto3DSeg uses
# MLflow internally for some logging and experiment tracking.
#! Now Auto3dseg will fail if you dont add this. I think mlflow made an update and auto3dseg hasnt caught up
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

"""
Train or run inference with MONAI Auto3DSeg for DeepTHOMAS.

This script is meant to be run from inside the same folder that contains
this file. The folder should also contain:

    data/dataset_thalamus821/imagesTr/
    data/dataset_thalamus821/labelsTr/
    data/dataset_thalamus821/imagesTs/

"""

current_dir = Path(__file__).parent

# -----------------------------------------------------------------------------
# Basic settings
# -----------------------------------------------------------------------------

# DATASET_HOME is where datasets (e.g. dataset_thalamus821; new_ood_data; ADNI24) live
DATASET_HOME = current_dir / "data"
# WORK_HOME is where model training folders (aka MONAI bundles) will be created
WORK_HOME = current_dir / "training_work_dirs"

# I combine these variables to produce a unique name for a training run
TASK_NAME = "segmentation"
DATASET_ID = "thalamus821"
MODEL_NAME = "segresnet"
RUN_LABEL = "main"  

DATAROOT_NAME = f"dataset_{DATASET_ID}"
WORK_DIR_NAME = f"{TASK_NAME}_{DATASET_ID}_{MODEL_NAME}_{RUN_LABEL}"


NUM_FOLDS = 5

# Setting a random seed ensures that the random operations (like shuffling) 
#   produce the same results every time you run the code. 
# This is important for reproducibility, especially when you want to compare
#   results across different runs 
RANDOM_SEED = 42 

# -----------------------------------------------------------------------------
# Training parameters
# -----------------------------------------------------------------------------

# I trained deepthomas with 250 epochs. That took several hours. For testing code, I set it low so training finishes quickly
TRAIN_PARAMS = { 
    "num_epochs_per_validation": 1,
    # "num_epochs": 250,
    "num_epochs": 5,
    "num_warmup_epochs": 1,
}

def run_auto3dseg(paths):
    """
    This will run the entire Auto3DSeg pipeline, including training and ensemble inference.
     - During training, it will automatically perform cross-validation based on the fold assignments in the datalist.
     - After training, it will perform model ensembling to create a final model that combines the strengths of the individual models trained on each fold. 
        - The default ensemble method used here is "best N models", which selects the top N performing models based on their validation performance during training.
        - The predictions are combined into a single output using a simple majority vote (or averaging the probability maps, I forget which is the default)
    """
    datalist_file = create_datalist(paths)

    print(f"Using datalist file at: {datalist_file}")
    print(f"Using dataroot at: {paths['dataroot']}")

    ensemble_save_params = {
        # output_postfix is a string appended as a suffix to the base filename.
        # leaving it blank causes
        # the same exact name as the ground truth labels, which is okay because they're saved in a unique folder with a descriptive name
        "output_postfix": "", 
        "output_dir": str(paths['labelsTs_infer']),
        "data_root_dir": str(paths['dataroot'] / paths['imagesTs'])
    }
    auto_runner = AutoRunner(
        work_dir=str(paths["work_dir"]),
        algos=["segresnet"],
        input={
            "name": f"{TASK_NAME}_{DATASET_ID}",
            "task": "segmentation",
            "modality": "MRI",
            "datalist": str(datalist_file),
            "dataroot": str(paths["dataroot"]),
        },
        **ensemble_save_params
    )
    auto_runner.set_training_params(params=TRAIN_PARAMS)
    auto_runner.run()


def create_datalist(paths, force_recreate=False):
    """
    This function creates a datalist.json file that MONAI uses to load the training and testing data during the Auto3DSeg pipeline.
    It reads the structure of the directory specified as dataroot and creates a JSON file that lists the paths to the training images, 
    training labels, and testing images.
    - imageTr: training images
    - labelsTr: training labels
    - imageTs: testing images (optional for training, but useful for later inference and evaluation)
    
    First checks whether datalist.json already exists. Only runs if it does not exist or if force_recreate=True

    Returns the path to the datalist.json file

    """

    work_dir = paths["work_dir"]
    datalist_file = work_dir / "datalist.json"

    # check if datalist.json exists and then decides whether to proceed
    if datalist_file.exists() and not force_recreate:
        print(f"datalist.json already exists at: {datalist_file}")
        print("If you want to recreate it, delete the existing datalist.json and run this function again.")
        return datalist_file

    
    dataroot = paths["dataroot"]
    labelsTr_dir = paths["labelsTr"]
    imagesTr_dir = paths["imagesTr"]
    imagesTs_dir = paths["imagesTs"]

    # read all the nifti files from each of the data dirs
    labelsTr = sorted(labelsTr_dir.glob("*.nii.gz"))
    imagesTr = set(imagesTr_dir.glob("*.nii.gz"))
    imagesTs = set(imagesTs_dir.glob("*.nii.gz"))

    training_items = [] 
    expected_images = set()

    # The data Manoj gave me have a consistent format: 
    # - labels have a filename like "case123.nii.gz" 
    # - the corresponding image has a filename like "case123_0000.nii.gz".
    for label in labelsTr:
        case_name = label.name.removesuffix(".nii.gz")
        image = imagesTr_dir / f"{case_name}_0000.nii.gz"
        expected_images.add(image)

        if image not in imagesTr:
            print(f"Missing image for label: {label}")
            print(f"Expected image: {image}")
            continue

        training_items.append(
            {
                "image": str(image.relative_to(dataroot)),
                "label": str(label.relative_to(dataroot)),
            }
        )
    # double check that there are no extra images that don't have corresponding labels
    extra_imagesTr = imagesTr - expected_images
    if extra_imagesTr:
        print(f"Found {len(extra_imagesTr)} image file(s) without labels:")

        for image in sorted(extra_imagesTr):
            print(image)


    # Assign fold numbers to each training item for cross-validation.
    random.seed(RANDOM_SEED)
    random.shuffle(training_items)

    for index, item in enumerate(training_items):
        item["fold"] = index % NUM_FOLDS


    # Read all the nifti files in the testing images folder
    testing_items = [
        {
            "image": str(image.relative_to(dataroot)),
        }
        for image in sorted(imagesTs)
    ]

    # Create the datalist dict with the training and testing data 
    datalist = {
        "training": training_items,
        "testing": testing_items
    }

    # Dump the dict into a json file
    with open(datalist_file, "w") as file:
        json.dump(datalist, file, indent=4)

    print(f"Saved datalist to: {datalist_file}")
    print(f"Number of training cases: {len(training_items)}")
    print(f"Number of testing cases: {len(testing_items)}")

    return datalist_file


def define_paths():
    """
    This function defines the paths to the various directories and files needed for training and inference.
    - It checks that the expected directories for training data exist (labelsTr and imagesTr).
    - It also checks for the existence of the imagesTs directory, but only prints a warning if it does not exist, 
        since this is not required for training.
    - It creates the training work directory if it does not already exist.
    
    Returns a dictionary of paths that can be used throughout the code to access these directories and files.
    """

    # default
    dataset_home = current_dir / "data"
    dataroot = dataset_home / DATAROOT_NAME
    imagesTr_dir = dataroot / "imagesTr"
    labelsTr_dir = dataroot / "labelsTr"
    imagesTs_dir = dataroot / "imagesTs"
    labelsTs_dir = dataroot / "labelsTs"
    labelsTs_infer_dir = dataroot / f"labelsTs_infer_{MODEL_NAME}_{RUN_LABEL}"

    paths = {
        "work_home": WORK_HOME,
        "work_dir": WORK_HOME / WORK_DIR_NAME,
        "dataset_home": DATASET_HOME,
        "dataroot": current_dir / "data" / DATAROOT_NAME,
        "imagesTr": imagesTr_dir,
        "labelsTr": labelsTr_dir,
        "imagesTs": imagesTs_dir,
        "labelsTs": labelsTs_dir,
        "labelsTs_infer": labelsTs_infer_dir
    }

    if not paths['work_dir'].exists():
        paths['work_dir'].mkdir(parents=True)

    for key in ["labelsTr", "imagesTr"]:
        if not paths[key].exists():
            raise FileNotFoundError(f"Expected directory does not exist: {paths[key]}")

    if not paths["imagesTs"].exists():
        print(f"Warning: Testing images directory does not exist: {paths['imagesTs_dir']}")
        print("This is not required for training, but you might want it later for inference and evaluation.")

    return paths


if __name__ == "__main__":
    paths = define_paths()
    run_auto3dseg(paths)
