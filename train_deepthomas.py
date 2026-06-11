"""
Train or run inference with MONAI Auto3DSeg for DeepTHOMAS.

This script is meant to be run from inside the same folder that contains
this file. The folder should also contain:

    configs/config_test.yaml
    data/dataset_thalamus821/imagesTr/
    data/dataset_thalamus821/labelsTr/
    data/dataset_thalamus821/imagesTs/
    training_work_dirs/segment_thalamus821_segresnet_orig/  # optional

The default command runs a short test training run using the settings in
configs/config_test.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from monai.apps.auto3dseg import (
    AlgoEnsembleBestN,
    AlgoEnsembleBuilder,
    AutoRunner,
    import_bundle_algo_history,
)
from monai.bundle.config_parser import ConfigParser
from monai.config import print_config
from omegaconf import OmegaConf


# -----------------------------------------------------------------------------
# Basic settings
# -----------------------------------------------------------------------------

# This lets MLflow use a local folder for tracking training runs. Auto3DSeg uses
# MLflow internally for some logging and experiment tracking.
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# The project folder is the folder that contains this script.
PROJECT_DIR = Path(__file__).resolve().parent

# Some values in the YAML config use ${oc.env:CWD}. We set CWD here so the
# config can refer to the project folder without hard-coding an absolute path.
os.environ["CWD"] = str(PROJECT_DIR)

# Where config files are stored.
CONFIG_DIR = PROJECT_DIR / "configs"

# This config is intentionally called "test" because it is meant to be a short
# first run. After that works, you can copy it and increase num_epochs.
CONFIG_FILE = "config_test.yaml"

# This random seed controls fold assignment in datalist.json. Keeping this fixed
# means the same cases will be assigned to the same folds each time.
RANDOM_SEED = 42

# This is the name of the copied official/original DeepTHOMAS model directory.
# It is used only when running inference with --use-original-model.
ORIGINAL_MODEL_WORK_DIR_NAME = "segment_thalamus821_segresnet_orig"

# Set this to True if you want MONAI to print detailed package/environment info.
PRINT_MONAI_CONFIG = False


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def run_auto3dseg(config: dict[str, Any], paths: dict[str, Path]) -> None:
    """Run MONAI Auto3DSeg training and ensemble inference.

    Auto3DSeg will train the model specified in the config file. Because the
    datalist contains a "fold" value for each training case, Auto3DSeg can use
    those fold values for cross-validation.
    """

    datalist_file = create_datalist(config, paths)

    print(f"Using datalist file: {datalist_file}")
    print(f"Using dataroot:      {paths['dataroot']}")
    print(f"Using work_dir:      {paths['work_dir']}")

    # AutoRunner expects a small task-definition file. This tells Auto3DSeg
    # where the data are and what kind of task we are running.
    input_cfg = {
        "name": f"{config['task']}_{config['dataset_id']}",
        "task": config["task"],
        "modality": "MRI",
        "datalist": str(datalist_file),
        "dataroot": str(paths["dataroot"]),
    }

    input_file = paths["work_dir"] / "input.yaml"
    ConfigParser.export_config_file(input_cfg, str(input_file), fmt="yaml")

    # These parameters tell Auto3DSeg where to save the testing-set predictions
    # created during the final ensemble inference stage.
    paths["labelsTs_infer_dir"].mkdir(parents=True, exist_ok=True)

    ensemble_save_params = {
        "output_postfix": "infer",
        "output_dir": str(paths["labelsTs_infer_dir"]),
        "data_root_dir": str(paths["dataroot"]),
    }

    auto_runner = AutoRunner(
        work_dir=str(paths["work_dir"]),
        algos=[config["model"]],
        input=str(input_file),
        **ensemble_save_params,
    )

    auto_runner.set_training_params(params=config["train_params"])
    auto_runner.run()


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

def run_inference(config: dict[str, Any], paths: dict[str, Path]) -> None:
    """Run ensemble inference using an existing trained model directory.

    This is useful if new images are added to imagesTs and you want to generate
    segmentation labels without retraining the model.
    """

    datalist_file = create_datalist(config, paths)
    work_dir = paths["work_dir"]
    save_dir = paths["labelsTs_infer_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using trained model directory: {work_dir}")
    print(f"Saving inference outputs to:  {save_dir}")

    # SaveImage is the MONAI transform that writes prediction files to disk.
    save_params = {
        "_target_": "SaveImage",
        "output_dir": str(save_dir),
        "data_root_dir": str(paths["dataroot"]),
        "output_postfix": "ensemble",
        "separate_folder": True,
    }

    task = {
        "name": f"{config['task']}_{config['dataset_id']}",
        "task": config["task"],
        "modality": "MRI",
        "datalist": str(datalist_file),
        "dataroot": str(paths["dataroot"]),
    }

    # The ensemble builder needs a task file describing the inference data.
    task_file = save_dir / "inference-task.json"
    with open(task_file, "w", encoding="utf-8") as file:
        json.dump(task, file, indent=4)

    # Read the training history from the Auto3DSeg work directory. This tells
    # MONAI which trained model bundles are available for inference.
    history = import_bundle_algo_history(str(work_dir), only_trained=True)

    # For 5-fold training with one algorithm, there are usually 5 trained models.
    # AlgoEnsembleBestN selects the best N models based on validation results.
    n_best = int(config.get("num_folds", 5))
    builder = AlgoEnsembleBuilder(history, str(task_file))
    builder.set_ensemble_method(AlgoEnsembleBestN(n_best=n_best))
    ensemble = builder.get_ensemble()

    ensemble(pred_param={"image_save_func": save_params})


# -----------------------------------------------------------------------------
# Datalist creation
# -----------------------------------------------------------------------------

def create_datalist(
    config: dict[str, Any],
    paths: dict[str, Path],
    force_recreate: bool = False,
) -> Path:
    """Create datalist.json for MONAI Auto3DSeg.

    The datalist is a JSON file that tells MONAI where each image and label is.
    For this dataset, training images and labels follow this naming pattern:

        image:  imagesTr/case_name_0000.nii.gz
        label:  labelsTr/case_name.nii.gz

    The "_0000" part marks image channel 0. This is a common convention in
    MONAI/nnU-Net-style datasets.
    """

    work_dir = paths["work_dir"]
    datalist_file = work_dir / "datalist.json"

    if datalist_file.exists() and not force_recreate:
        print(f"datalist.json already exists: {datalist_file}")
        print("Using the existing datalist.")
        print("Delete it, or run with --force-datalist, to recreate it.")
        return datalist_file

    dataroot = paths["dataroot"]
    labels_tr_dir = paths["labelsTr_dir"]
    images_tr_dir = paths["imagesTr_dir"]
    images_ts_dir = paths["imagesTs_dir"]

    labels_tr = sorted(labels_tr_dir.glob("*.nii.gz"))
    images_tr = set(images_tr_dir.glob("*.nii.gz"))

    if images_ts_dir.exists():
        images_ts = sorted(images_ts_dir.glob("*.nii.gz"))
    else:
        images_ts = []

    if not labels_tr:
        raise FileNotFoundError(f"No label files found in: {labels_tr_dir}")

    if not images_tr:
        raise FileNotFoundError(f"No image files found in: {images_tr_dir}")

    training_items: list[dict[str, Any]] = []
    expected_images: set[Path] = set()
    missing_images: list[tuple[Path, Path]] = []

    for label in labels_tr:
        case_name = label.name.removesuffix(".nii.gz")
        image = images_tr_dir / f"{case_name}_0000.nii.gz"
        expected_images.add(image)

        if image not in images_tr:
            missing_images.append((label, image))
            continue

        # The datalist stores paths relative to dataroot. Auto3DSeg combines
        # these relative paths with the dataroot path when it loads the files.
        training_items.append(
            {
                "image": str(image.relative_to(dataroot)),
                "label": str(label.relative_to(dataroot)),
            }
        )

    if missing_images:
        print("Found labels without matching images:")
        for label, expected_image in missing_images:
            print(f"  label:          {label}")
            print(f"  expected image: {expected_image}")

        raise FileNotFoundError(
            "Some training labels do not have matching images. "
            "Please fix the dataset before training."
        )

    # This is a useful check for data organization mistakes. Extra images are
    # not fatal, but they will not be used for training unless they have labels.
    extra_images_tr = images_tr - expected_images
    if extra_images_tr:
        print(f"Found {len(extra_images_tr)} training image(s) without labels:")
        for image in sorted(extra_images_tr):
            print(f"  {image}")

    # Assign fold numbers to each training item for cross-validation.
    #
    # Auto3DSeg can use this "fold" value during training:
    #   - cases with fold == current fold become validation cases
    #   - cases with fold != current fold become training cases
    #
    # We use a fixed seed so that the assignments are reproducible.
    random_generator = random.Random(RANDOM_SEED)
    random_generator.shuffle(training_items)

    for index, item in enumerate(training_items):
        item["fold"] = index % int(config["num_folds"])

    testing_items = [
        {"image": str(image.relative_to(dataroot))}
        for image in images_ts
    ]

    datalist = {
        "training": training_items,
        "testing": testing_items,
    }

    work_dir.mkdir(parents=True, exist_ok=True)
    with open(datalist_file, "w", encoding="utf-8") as file:
        json.dump(datalist, file, indent=4)

    print(f"Saved datalist to: {datalist_file}")
    print(f"Number of training cases: {len(training_items)}")
    print(f"Number of testing cases:  {len(testing_items)}")

    fold_counts = {}
    for item in training_items:
        fold = item["fold"]
        fold_counts[fold] = fold_counts.get(fold, 0) + 1

    print("Fold counts:")
    for fold in sorted(fold_counts):
        print(f"  fold {fold}: {fold_counts[fold]} cases")

    return datalist_file


# -----------------------------------------------------------------------------
# Config and path handling
# -----------------------------------------------------------------------------

def read_config(config_file: Path | str) -> tuple[dict[str, Any], dict[str, Path]]:
    """Read the YAML config file and create useful project paths."""

    full_config = parse_omegaconf_config(config_file)
    config = full_config["config"]
    raw_paths = full_config["paths"]

    training_run_name = (
        f"{config['task']}_{config['dataset_id']}_"
        f"{config['model']}_{config['run_label']}"
    )

    dataset_home = Path(raw_paths["dataset_home"])
    work_home = Path(raw_paths["work_home"])
    dataroot = dataset_home / f"dataset_{config['dataset_id']}"

    paths = {
        "dataset_home": dataset_home,
        "work_home": work_home,
        "work_dir": work_home / training_run_name,
        "original_model_work_dir": work_home / ORIGINAL_MODEL_WORK_DIR_NAME,
        "dataroot": dataroot,
        "imagesTr_dir": dataroot / "imagesTr",
        "labelsTr_dir": dataroot / "labelsTr",
        "imagesTs_dir": dataroot / "imagesTs",
        "labelsTs_dir": dataroot / "labelsTs",
        "labelsTs_infer_dir": dataroot / f"labelsTs_{config['model']}_{config['run_label']}",
        "labelsTs_orig_infer_dir": dataroot / f"labelsTs_{config['model']}_orig",
    }

    paths["work_dir"].mkdir(parents=True, exist_ok=True)

    for key in ["labelsTr_dir", "imagesTr_dir"]:
        if not paths[key].exists():
            raise FileNotFoundError(f"Expected directory does not exist: {paths[key]}")

    if not paths["imagesTs_dir"].exists():
        print(f"Warning: testing images directory does not exist: {paths['imagesTs_dir']}")
        print("Training can still run, but inference needs imagesTs.")

    return config, paths


def parse_omegaconf_config(
    config_data: Path | str,
    resolve_to_dict: bool = True,
) -> dict[str, Any] | Any:
    """Parse a YAML config file using OmegaConf.

    OmegaConf allows useful placeholders in YAML files. For example, the config
    can use ${oc.env:CWD} to mean "the project directory".
    """

    if isinstance(config_data, str) and config_data.endswith((".yaml", ".yml")):
        config_data = Path(config_data)

    if isinstance(config_data, Path):
        config = OmegaConf.load(config_data)
    else:
        config = OmegaConf.create(config_data)

    if resolve_to_dict:
        return OmegaConf.to_container(config, resolve=True)

    return config


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Train or run inference with MONAI Auto3DSeg for DeepTHOMAS."
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="train",
        choices=["train", "infer", "make-datalist"],
        help="What to do. Default is 'train'.",
    )

    parser.add_argument(
        "--force-datalist",
        action="store_true",
        help="Recreate datalist.json even if it already exists.",
    )

    parser.add_argument(
        "--use-original-model",
        action="store_true",
        help=(
            "For inference, use training_work_dirs/"
            f"{ORIGINAL_MODEL_WORK_DIR_NAME} instead of the test run."
        ),
    )

    return parser


def main() -> None:
    """Read the config, prepare paths, and run the selected command."""

    if PRINT_MONAI_CONFIG:
        print_config()

    parser = build_arg_parser()
    args = parser.parse_args()

    config_file = CONFIG_DIR / CONFIG_FILE
    config, paths = read_config(config_file)

    if args.use_original_model:
        paths["work_dir"] = paths["original_model_work_dir"]
        paths["labelsTs_infer_dir"] = paths["labelsTs_orig_infer_dir"]

    if args.command == "make-datalist":
        create_datalist(config, paths, force_recreate=args.force_datalist)
    elif args.command == "train":
        if args.force_datalist:
            create_datalist(config, paths, force_recreate=True)
        run_auto3dseg(config, paths)
    elif args.command == "infer":
        run_inference(config, paths)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
