# Dog Breed Classifier on AWS SageMaker

Fine-tunes a pretrained ResNet-18 to classify 133 dog breeds, with SageMaker hyperparameter tuning, Debugger, Profiler and a deployed inference endpoint.

## Results

| Stage | Result |
|---|---|
| Tuning (2 epochs per job, 4 jobs) | Best validation accuracy 76.53% (batch size 42, learning rate 0.002458) |
| Final training (10 epochs) | 80.84% validation accuracy at epoch 10; peak 80.00% at epoch 7 before that |
| Endpoint check | One Afghan hound photo returned class index 1, which is Afghan hound |

- No baseline model was trained, so there is no comparison number. The 80.84% figure is on the `valid` split; the `test` split was not evaluated.
- The final training job did not use the tuner's best values. It ran with batch size 46 and learning rate 0.00358 (typed into the notebook by hand) and 10 epochs, whereas the tuner's best job was batch size 42 and learning rate 0.002458 with 2 epochs.
- The notebook's markdown cell also states 46 and 0.00358 as the "best" values, which does not match the tuner output printed above it.

## Problem

Classify a dog photo into one of 133 breeds and serve the model behind an HTTPS endpoint.

## Data

The Udacity `dogImages.zip` dataset (about 1.1 GB, roughly 8,000 images) in `train`, `valid` and `test` folders, one subfolder per breed. See `viz.png` for the structure. The notebook downloads and unzips it with `wget` (cell 5). The upload to S3 (cell 9) is commented out, so the S3 `inputs` path (cell 11) must already hold the data unless that cell is uncommented.

![Dataset structure](viz.png)

## Approach

- Transfer learning: ResNet-18 pretrained on ImageNet, all backbone weights frozen, new 133-way linear head trained with Adam. This keeps training cheap and fast on a single GPU instance.
- Tuning searched only learning rate (0.001 to 0.1) and batch size (32 to 64), with 4 jobs and 2 in parallel, to limit cost. The objective was validation accuracy.
- Final training (`train_model.py`) adds random horizontal flip and ImageNet normalisation, and logs loss through the `smdebug` hook.
- Debugger rules checked for vanishing gradients, overfitting, overtraining and poor weight initialisation. In the final training job (`pytorch-training-2025-03-30-22-33-28-306`), the `PoorWeightInitialization` rule reported `IssuesFound` at epoch 4. The other three rules were still `InProgress` in the printed log, and no final status for them is shown. This issue was not investigated. The notebook's markdown note after the loss plot says there are no anomalies, which the printed output does not support.
- Profiler: CPU was busy during data loading and GPU was busy during training steps, which supports using a GPU instance. The profiler screenshots (`cpu.png`, `gpu.png`) come from an earlier job, `pytorch-training-2025-03-30-19-02-34-503`, not the final job `pytorch-training-2025-03-30-22-33-28-306`.
- `inference.py` serves the model on an `ml.m5.large` endpoint, taking raw image bytes (`application/x-image`) and returning a predicted class index as JSON.

| Tuning jobs | | |
|---|---|---|
| ![Tuning 1](screenshots/tuning1.JPG) | ![Tuning 2](screenshots/tunin2.JPG) | ![Tuning 3](screenshots/tuning3.JPG) |

| Loss | CPU utilisation | GPU utilisation |
|---|---|---|
| ![Loss](screenshots/loss.png) | ![CPU](screenshots/cpu.png) | ![GPU](screenshots/gpu.png) |

![Endpoint](screenshots/endpoint.JPG)

## Repo contents

| File | Description |
|---|---|
| `train_and_deploy.ipynb` | Notebook that runs tuning, training, profiling and deployment |
| `hpo.py` | Training script used by the tuning jobs |
| `train_model.py` | Training script with Debugger hooks |
| `inference.py` | Model loading and request handlers for the endpoint |
| `Debugger Profiler Report.pdf` | SageMaker profiler report |
| `afghan_hound.jpg` | Test image sent to the endpoint |
| `screenshots/`, `viz.png` | Images used above |

## How to run

This is notebook-only; there is no local entry point. In SageMaker Studio:

1. Open `train_and_deploy.ipynb` with the scripts in the same folder.
2. Run the cells in order. The upload-to-S3 cell is commented out; uncomment it for a first run. The S3 path and account id are shown as `<ACCOUNT_ID>` and must be replaced with your own bucket.
3. The notebook uses PyTorch 1.8.0 / Python 3.6 SageMaker containers on `ml.g4dn.xlarge` for training.
4. Delete the endpoint when finished to avoid charges.

## Limitations and next steps

- Backbone is frozen and the tuning budget was small (4 jobs, 2 epochs), so accuracy likely has room to grow by fine-tuning deeper layers.
- Final hyperparameters were not the tuner's best, and no baseline or test-split score was recorded.
- The Debugger `PoorWeightInitialization` rule reported `IssuesFound` at epoch 4 of the final job and has not been diagnosed.
- Validation accuracy moved between epochs (73.8% at epoch 1, 80.8% at the end, not rising every epoch) and there is a train/validation loss gap, so some overfitting is likely; stronger augmentation would help.
- The endpoint returns a class index only; mapping to breed names is done by hand in the notebook.

Coursework for the Udacity AWS Machine Learning Engineer Nanodegree.
