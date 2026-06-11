import json
from pathlib import Path
import random
from monai.apps.auto3dseg import (
    AutoRunner,
    AlgoEnsembleBestN,
    AlgoEnsembleBuilder,
    import_bundle_algo_history,
)
from monai.config import print_config
from omegaconf import OmegaConf

print_config()

# These variables define the specific training task we want to run. I came up with this naming paradigm
# so that the training_work_dirs where model training is saved would be expressive and self documenting
TASK_NAME = "segment"
DATASET_ID = "thalamus821"
MODEL_NAME = "segresnet"
RUN_LABEL = "orig" #change this to orig to try running inference with the model I used for the paper

DATAROOT_NAME = f"dataset_{DATASET_ID}"
WORK_DIR_NAME = f"{TASK_NAME}_{DATASET_ID}_{MODEL_NAME}_{RUN_LABEL}"


NUM_FOLDS = 5

# Setting a random seed ensures that the random operations (like shuffling) 
#   produce the same results every time you run the code. 
# This is important for reproducibility, especially when you want to compare
#   results across different runs 
RANDOM_SEED = 42 


current_dir = Path(__file__).parent


def run_auto3dseg(paths):
    """
    This will run the entire Auto3DSeg pipeline, including training and ensemble inference.
     - During training, it will automatically perform cross-validation based on the fold assignments in the datalist.
     - After training, it will perform model ensembling to create a final model that combines the strengths of the individual models trained on each fold. 
        - The default ensemble method used here is "best N models", which selects the top N performing models based on their validation performance during training.
        - The predictions are combined into a single output using a simple majority vote 
    """
    datalist_file = create_datalist(paths)

    print(f"Using datalist file at: {datalist_file}")
    print(f"Using dataroot at: {paths['dataroot']}")

    ensemble_save_params = {
        "output_postfix": "infer",
        "output_dir": str(paths['labelsTs_infer']),
        "data_root_dir": str(paths['dataroot'] / paths['imagesTs'])
    }
    auto_runner = AutoRunner(
        work_dir=str(paths["training_work_dir"]),
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

    auto_runner.run()



def infer(paths):
    """
    Function to run inference after `run_auto3dseg` if you get new images you want to infer
    """

    training_work_dir = paths['training_work_dir']
    save_dir = training_work_dir / "ensemble_output"
    dataroot = paths['dataroot']
    datalist_file = create_datalist(paths)

    # I set these parameters to be the same as the default that AutoRunner uses during its ensemble phase
    save_params = {
        "_target_": "SaveImage",
        "output_dir": save_dir,
        "data_root_dir": dataroot,
        "output_postfix": "ensemble",
        "separate_folder": True
    }


    task_name = f"{TASK_NAME}_{DATASET_ID}"
    task = {
        "name": task_name, # this can be anything you want, it's just for your reference
        "task": "segmentation",
        "modality": "MRI",
        "datalist": datalist_file,
        "dataroot": dataroot,
    }

    # fine grained control over output filenames requires saving a config file with the task definition and passing this file to the ensemble builder
    task_file = save_dir / "inference-task.json"
    with open(task_file, "w") as f:
        json.dump(task, f, indent=4)


    # read the training history
    history = import_bundle_algo_history(training_work_dir, only_trained=True)

    ## model ensemble
    n_best = 5 # we will only have 5 anyways since we used 5 folds with one algo (segresnet)
    builder = AlgoEnsembleBuilder(history, task_file)
    builder.set_ensemble_method(AlgoEnsembleBestN(n_best=n_best))
    ensemble = builder.get_ensemble()
    ensemble(pred_param={"image_save_func": save_params})


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

    training_work_dir = paths["training_work_dir"]
    datalist_file = training_work_dir / "datalist.json"

    # check if datalist.json exists and then decides whether to proceed
    if datalist_file.exists() and not force_recreate:
        print(f"datalist.json already exists at: {datalist_file}")
        print("If you want to recreate it, delete the existing datalist.json and run this function again.")
        return datalist_file

    
    dataroot = paths["dataroot"]
    labelsTr_dir = paths["labelsTr_dir"]
    imagesTr_dir = paths["imagesTr_dir"]
    imagesTs_dir = paths["imagesTs_dir"]

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


def define_paths(config=None):
    """
    This function defines the paths to the various directories and files needed for training and inference.
    - It checks that the expected directories for training data exist (labelsTr and imagesTr).
    - It also checks for the existence of the imagesTs directory, but only prints a warning if it does not exist, 
        since this is not required for training.
    - It creates the training work directory if it does not already exist.
    
    Returns a dictionary of paths that can be used throughout the code to access these directories and files.
    """

    # default
    if config is None:
        dataset_home = current_dir / "data"
        dataroot = dataset_home / DATAROOT_NAME
        imagesTr_dir = dataroot / "imagesTr"
        labelsTr_dir = dataroot / "labelsTr"
        imagesTs_dir = dataroot / "imagesTs"
        labelsTs_dir = dataroot / "labelsTs"
        labelsTs_infer_dir = dataroot / "labelsTs_infer"

        paths = {
            "auto3dseg_bundles_dir": current_dir / "auto3dseg_bundles",
            "training_work_dir": current_dir / "auto3dseg_bundles" / WORK_DIR_NAME,
            "dataset_home": current_dir / "data",
            "dataroot": current_dir / "data" / DATAROOT_NAME,
            "imagesTr_dir": imagesTr_dir,
            "labelsTr_dir": labelsTr_dir,
            "imagesTs_dir": imagesTs_dir,
            "labelsTs_dir": labelsTs_dir,
            "labelsTs_infer_dir": labelsTs_infer_dir
        }

    else:
        paths = parse_omegaconf_config(config)

    if not paths['training_work_dir'].exists():
        paths['training_work_dir'].mkdir(parents=True)

    for key in ["labelsTr_dir", "imagesTr_dir"]:
        if not paths[key].exists():
            raise FileNotFoundError(f"Expected directory does not exist: {paths[key]}")

    if not paths["imagesTs_dir"].exists():
        print(f"Warning: Testing images directory does not exist: {paths['imagesTs_dir']}")
        print("This is not required for training, but you might want it later for inference and evaluation.")

    return paths



def parse_omegaconf_config(config_data, resolve_to_dict=True):
    """
    Parses a YAML configuration file or string using OmegaConf.
    Natively supports ${task.key} and ${oc.env:VAR} interpolation.

    - These are some more advanced parsing techniques to read yaml files. What's important is to
    """
    # 1. Load from file path or string seamlessly
    if isinstance(config_data, str) and config_data.endswith((".yaml", ".yml")):
        config = OmegaConf.load(config_data)
    else:
        config = OmegaConf.create(config_data)
        
    # 2. Return frozen standard dict if explicitly requested, 
    # otherwise return the dynamic DictConfig object
    if resolve_to_dict:
        return OmegaConf.to_container(config, resolve=True)
        
    return config


    

def main(function_to_run=run_auto3dseg): 
    paths = define_paths("config.yaml")
    function_to_run(paths)


if __name__ == "__main__":
    main(function_to_run=run_auto3dseg)