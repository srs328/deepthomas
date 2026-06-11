# DeepTHOMAS Auto3DSeg training package

This folder contains a small MONAI Auto3DSeg workflow for training and running
DeepTHOMAS thalamus segmentation.

The package is designed so that everything lives in one project folder:

```text
deepthomas/
├── README.md
├── run_auto3dseg.py
├── run_inference.py
├── train_deepthomas.py
├── requirements.txt
├── configs/
│   └── config_test.yaml
├── data/
│   └── dataset_thalamus821/
│       ├── imagesTr/
│       ├── labelsTr/
│       └── imagesTs/
├── training_work_dirs/
│   └── segment_thalamus821_segresnet_main/
├── scratch/
│   └── random_crap (till I feel comfortable to delete)
└── utils
    └── summarize_datta.py
```

The `data/` folder stores datasets
`training_work_dirs/` folder is where MONAI is told to save model training outputs.

The included directory
`training_work_dirs/segment_thalamus821_segresnet_main/` is the copied original
DeepTHOMAS model. It can be used for inference without having to run training again.

---

## 1. Install Python requirements

Use Python 3.10 or newer. From inside the `deepthomas/` folder, create and
activate a virtual environment.

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 2. Check the data layout

The dataset folder should look like this:

```text
data/dataset_thalamus821/
├── imagesTr/
│   ├── S01496_0000.nii.gz
│   ├── LAB_S01724_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── S01496.nii.gz
│   ├── LAB_S01724.nii.gz
│   └── ...
└── imagesTs/
    ├── some_test_case_0000.nii.gz
    └── ...
```

The naming pattern matters:

```text
training image:  imagesTr/case_name_0000.nii.gz
training label:  labelsTr/case_name.nii.gz
```

The `_0000` means image channel 0. This is a common convention in MONAI and
nnU-Net-style medical image segmentation datasets.

The script checks that each label has a matching image before training starts.

---

## Running inference with the original model

Running inference on new images only requires the `imagesTs` folder under a dataset folder in `data/` (e.g. `data/new_ood_data/imagesTs`). Image file naming can be more freeform here (i.e. `case_name_0000` is not enforced)
If new images are added to:

```text
data/dataset_thalamus821/imagesTs/
```

you can run inference using the copied original DeepTHOMAS model:

```bash
python train_deepthomas.py infer --use-original-model
```

This uses:

```text
training_work_dirs/segment_thalamus821_segresnet_orig/
```

and saves outputs in:

```text
data/dataset_thalamus821/labelsTs_segresnet_orig/
```

Make sure new inference images follow the same naming convention:

```text
case_name_0000.nii.gz
```

---

## 3. Run the first test training job

The default config is intentionally called `config_test.yaml`. It is meant for a
short first run so you can make sure the code, data paths, and MONAI setup work.

From inside the `deepthomas/` folder, run:

```bash
python train_deepthomas.py
```

This is the same as:

```bash
python train_deepthomas.py train
```

The test config currently uses a small number of epochs. It is not meant to be a
final scientific training run.

Training outputs will be saved in:

```text
training_work_dirs/segment_thalamus821_segresnet_test/
```

Inference outputs from that test run will be saved in:

```text
data/dataset_thalamus821/labelsTs_segresnet_test/
```

---

## 4. Make or remake the datalist

MONAI uses a file called `datalist.json` to know where the images and labels are.
The training script creates this automatically.

To create only the datalist, without training, run:

```bash
python train_deepthomas.py make-datalist
```

To force the script to recreate the datalist:

```bash
python train_deepthomas.py make-datalist --force-datalist
```

The datalist contains fold assignments for cross-validation. For this project,
each training case gets a `fold` number. During Auto3DSeg training, cases from
the current fold are used as validation cases, and cases from the other folds are
used as training cases.

---

## 6. Run inference with your test-trained model

After running the test training job, you can run inference using that test model:

```bash
python train_deepthomas.py infer
```

This uses:

```text
training_work_dirs/segment_thalamus821_segresnet_test/
```

and saves outputs in:

```text
data/dataset_thalamus821/labelsTs_segresnet_test/
```

Usually, the original model is the better choice for real inference. The test
model is mainly useful for checking that the training workflow runs.

---

## 7. Change training settings

Training settings live in:

```text
configs/config_test.yaml
```

Important fields:

```yaml
config:
  task: segmentation
  dataset_id: thalamus821
  model: segresnet
  run_label: test
  num_folds: 5
  train_params:
    num_epochs_per_validation: 1
    num_epochs: 20
    num_warmup_epochs: 1
```

For a longer run, copy the config file, give it a new name, and increase
`num_epochs`. If you create a new config file, update this line near the top of
`train_deepthomas.py`:

```python
CONFIG_FILE = "config_test.yaml"
```

For example, you could copy `config_test.yaml` to `config_baseline.yaml`, change
`run_label` to `baseline`, and increase `num_epochs`.

---

## 8. What to send if something fails

If you get an error, send Shridhar:

1. The exact command you ran.
2. The full error message.
3. The contents of `configs/config_test.yaml`.
4. The first several lines printed by `train_deepthomas.py`.
5. A screenshot or listing of the relevant folder, especially:

```text
data/dataset_thalamus821/
training_work_dirs/
```

A useful folder listing command is:

```bash
find data/dataset_thalamus821 -maxdepth 2 -type f | head -50
```

On Windows PowerShell, a similar command is:

```powershell
Get-ChildItem data\dataset_thalamus821 -Recurse | Select-Object -First 50
```

---

## 9. Notes on the original model directory

Please do not delete this folder unless Shridhar tells you to:

```text
training_work_dirs/segment_thalamus821_segresnet_orig/
```

It contains the copied original model and is useful for inference on new images.

It is okay if new training runs create additional folders inside
`training_work_dirs/`. Those are generated outputs and can become large.
