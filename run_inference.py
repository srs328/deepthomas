
import json
from pathlib import Path

from monai.apps.auto3dseg import (
    AlgoEnsembleBestN,
    AlgoEnsembleBuilder,
    import_bundle_algo_history,
)


current_dir = Path(__file__).parent # path to the current file

WORK_HOME = current_dir / "training_work_dirs"
DATASET_HOME = current_dir / "data"

# the name of the folder containing the fully trained model
WORK_DIR_NAME = "segmentation_thalamus821_segresnet_main"
training_alias = "segresnet_main" # a string for uniquely identifying this training run for staying organized

dataset_to_infer = "new_ood_data" # this has that one case of the kid with Tay Sachs that manoj sent me

def run_inference():
    work_dir = WORK_HOME / WORK_DIR_NAME 
    dataroot = DATASET_HOME / dataset_to_infer
    save_dir = dataroot / f"labelTs_infer_{training_alias}"
    save_dir.mkdir(parents=True, exist_ok=True)

    datalist_file = "inference-datalist.json"
    create_datalist(dataroot, datalist_file)

    print(f"Using trained model directory: {work_dir}")
    print(f"Saving inference outputs to:  {save_dir}")

    # SaveImage is the MONAI transform that writes prediction files to disk.
    save_params = {
        "_target_": "SaveImage",
        "output_dir": str(save_dir),
        "data_root_dir": str(dataroot/"imagesTs"),
        "output_postfix": "ensemble",
        "separate_folder": False,
    }

    task = {
        "name": "inference-task",
        "task": "segmentation",
        "modality": "MRI",
        "datalist": str(datalist_file),
        "dataroot": str(dataroot),
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
    n_best = 5
    builder = AlgoEnsembleBuilder(history, str(task_file))
    builder.set_ensemble_method(AlgoEnsembleBestN(n_best=n_best))
    ensemble = builder.get_ensemble()

    ensemble(pred_param={"image_save_func": save_params})

def create_datalist(
    dataroot,
    datalist_file
) -> Path:

    images_ts_dir = dataroot / "imagesTs"
    images_ts = sorted(images_ts_dir.glob("*.nii.gz"))

    testing_items = [
        {"image": str(image.relative_to(dataroot))}
        for image in images_ts
    ]

    datalist = {
        "testing": testing_items,
    }

    with open(datalist_file, "w", encoding="utf-8") as file:
        json.dump(datalist, file, indent=4)

    print(f"Saved datalist to: {datalist_file}")
    print(f"Number of testing cases:  {len(testing_items)}")


if __name__ == "__main__":
    run_inference()