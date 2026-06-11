# Auto3Dseg Notes

The Auto3Dseg pipeline is run using the class `AutoRunner` in `monai/apps/auto3dseg/auto_runner.py` ([installed locally here](.venv/lib/python3.12/site-packages/monai/apps/auto3dseg/auto_runner.py)).

## Datalist

Auto3Dseg looks for certain keys in `datalist.json`, explained below. It ignores everything else, so you can add whatever else you want to make `datalist.json` expressive, self-documenting, and organized.

## Training

The minimal `datalist.json` file for training has one key called "training" that is an array containing the training cases. Each training case is an object with two required keys: "image" and "label" that define the paths to the training data.

**minimal datalist**
```json
{
    "training": [
        {
            "image": "/path/to/image1",
            "label": "/path/to/label1"
        },
        {
            "image": "/path/to/image2",
            "label": "/path/to/label2"
        }
    ]
}
```

Optionally, you can assign each training case to a fold number. If you don't, Auto3Dseg will automatically partition the data randomly into equal sized folds based on the value of `num_fold`. Auto3Dseg uses a default value of 5 for `num_fold` if it is undefined. If specifying fold partitioning, `datalist.json` would look like:

**datalist with folds**
```json
{
    "training": [
        {
            "fold": 0,
            "image": "/path/to/image1",
            "label": "/path/to/label1"
        },
        {
            "fold": 1,
            "image": "/path/to/image2",
            "label": "/path/to/label2"
        }
    ]
}
```

Folds are zero-indexed, so the first fold is labeled `0`, the last fold is labeled `num_fold - 1`. Otherwise, Auto3Dseg will raise: `ValueError: Fold numbers are not continuous from 0 to <num_fold-1>`

### Inference

To define cases for inference post training, Auto3Dseg looks for the key "testing", containing objects that require only the "image" key. A full `datalist.json` defining training and testing cases could look like:

**training and testing datalist**
```json
{
    "training": [
        {
            "fold": 0,
            "image": "/path/to/train_image1",
            "label": "/path/to/train_label1"
        },
        {
            "fold": 1,
            "image": "/path/to/train_image2",
            "label": "/path/to/train_label2"
        }
    ],
    "testing": [
        {
            "image": "/path/to/test_image1",
        }
    ]
}
```

## AutoRunner

The Auto3dSeg pipeline begins with the instantiation of an `AutoRunner` object and a call to its run function.

### Minimal Auto3Dseg Run

Once a `datalist.json` file is created, a minimal instantiation of `AutoRunner` for training SegResNet can look like:

```python
runner = AutoRunner(
    algos="segresnet"
    input={
        "modality": "MRI",
        "datalist": "/path/to/datalist.json",
        "dataroot": "/path/to/dataroot"
    }
)
runner.run()
```

- This will create a folder called `work_dir` (the Auto3Dseg default) located in your current working directory. All the training outputs would be saved into `work_dir`.
- Omitting `algos="segresnet"` would cause Auto3Dseg to train all of its models one by one.

### Specifying Auto3Dseg Stages

Auto3dSeg exposes more fine grained control over which steps to run as follows:

```python
auto_runner = AutoRunner(
    work_dir=str(paths["training_work_dir"]),
    algos=ALGOS,
    input={
        "modality": "MRI",
        "datalist": str(datalist_file),
        "dataroot": str(paths["dataroot"]),
    },
    analyze=True,
    algo_gen=True,
    train=True,
    ensemble=True,
)
```

If any of the four steps above are unspecified, Auto3Dseg will check to see if they have been run before to decide whether to run the step. Explicitely setting True will cause Auto3Dseg to run them anyways and overwrite previous output; setting False will cause Auto3Dseg to skip the step.

1. `analyse` runs the data analysis and produces `work_dir/datastats.yaml` and `work_dir/datastats_by_case.yaml`
2. `algo_gen` copies MONAI default templates for each of the algorithms specified by `algos=ALGOS` into `work_dir`. These folders are called "bundles"
   - An algorithm bundle contains configurations for network parameters, the scripts that Auto3Dseg will call to run the training, inference, etc, and training logs
   - Auto3Dseg fills out the configuration files inside bundle folders based on information it reads in `datastats.yaml` as well as information it checks about the computer you're running it from
     - `datastats.yaml` is used to set network parameters like image size, crop sizes, etc
     - Based on your computer's specs (CPU, GPU, etc), Aut3Dseg sets certain configuration parameters to optimize performance on your machine
3. The `train` step involves Auto3Dseg running the training scripts inside each of the bundles to train the model
4. The `ensemble` step invovles Auto3Dseg ranking all the algorithms that have been trained, selecting the top N (default N=5) algorithms to run inference, and then combining the N predictions (via averaging or majority voting) to produce a single inferred label.

### More Options

Many more options can be configured, like where to save inference labels, filename formats, training parameters, etc. I created a more [advanced guide](auto3dseg_advanced_guide.md) for myself awhile back with the help of Claude Code. That guide is geared more towards SegResNet in its discussion about parameters and bundle folder structure.

The options for output filenames from the ensemble inference stage was hidden deep inside MONAI's Python code but I finally figured it out. I pasted their full defaults below. I'll never have to touch most of these

```python
save_image = {
    "_target_": "SaveImage",
    "output_dir": output_dir,
    "output_postfix": kwargs.pop("output_postfix", "ensemble"),
    "output_dtype": kwargs.pop("output_dtype", "$np.uint8"),
    "resample": kwargs.pop("resample", False),
    "print_log": False,
    "savepath_in_metadict": True,
    "data_root_dir": kwargs.pop("data_root_dir", data_root_dir),
    "separate_folder": kwargs.pop("separate_folder", False),
}
```

I configured this in the code as follows:

- `output_postfix` is a string appended to the end of the original image's name 
- `output_dir` specifies where all the inference labels will be saved
- If I don't set `data_root_dir`, then MONAI will save the output inside: `{output_dir}/imagesTs`
    - The images are stored in `{dataroot}/imagesTs`. By default, `data_root_dir` equals the path you set as `dataroot`. MONAI determines the final output folder by doing `output_dir/<image_location - data_root_dir>`. By default, the subtraction leaves behind "imagesTs" which was bugging me since I wanted my labels to be directly inside the folder I set as the `output_dir`
    - that was very confusing and may not have made much sense, but its not important enough to worry about
```python
ensemble_save_params = {
    "output_postfix": "infer",
    "output_dir": str(paths['labelsTs_infer']),
    "data_root_dir": str(paths['dataroot'] / paths['imagesTs'])
}
```

