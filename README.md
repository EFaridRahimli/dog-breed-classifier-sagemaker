# Dog Breed Image Classification on AWS SageMaker

Fine-tunes a pretrained **ResNet-18** (PyTorch) to classify dog breeds, using the full SageMaker ML-engineering toolkit: **hyper-parameter tuning, Debugger, Profiler and a deployed inference endpoint**. Built for the Udacity AWS Machine Learning Engineer programme.

## Pipeline
1. **Data:** the dog-breed dataset is uploaded to S3 so SageMaker training jobs can read it.

   ![Dataset structure](viz.png)
2. **Hyper-parameter tuning** (`hpo.py`): transfer learning on ResNet-18, training only the new fully connected head with Adam. Search space:
   - learning rate: continuous, 0.001 – 0.1
   - batch size: integer, 32 – 64
3. **Training with Debugger and Profiler** (`train_model.py`), using the best hyper-parameters.
4. **Deployment** (`inference.py`): the trained model is served as a SageMaker endpoint via `PyTorchModel` and queried with a test image.

## Hyper-parameter tuning
| | | |
|---|---|---|
| ![Tuning 1](screenshots/tuning1.JPG) | ![Tuning 2](screenshots/tunin2.JPG) | ![Tuning 3](screenshots/tuning3.JPG) |

## Debugging and profiling
Debugger rules watched for **vanishing gradients, overfitting, overtraining and poor weight initialisation**.

- **Loss:** a healthy curve, dropping fast early and decreasing steadily through training.
- **CPU:** high at the start, during data loading, as expected.
- **GPU:** peaks during training steps and idles otherwise, which justified the GPU instance.

| Loss | CPU utilisation | GPU utilisation |
|---|---|---|
| ![Loss](screenshots/loss.png) | ![CPU](screenshots/cpu.png) | ![GPU](screenshots/gpu.png) |

The full profiler report is in `Debugger Profiler Report.pdf`.

## Deployed endpoint
![Endpoint](screenshots/endpoint.JPG)

## Repository contents
| File | Description |
|------|-------------|
| `train_and_deploy.ipynb` | Orchestration notebook: tuning, training, deployment, querying |
| `hpo.py` | Training script for the tuning jobs |
| `train_model.py` | Training script with Debugger/Profiler hooks |
| `inference.py` | Model loading and inference handlers for the endpoint |
| `Debugger Profiler Report.pdf` | SageMaker profiler report |

## Run it
Open `train_and_deploy.ipynb` in SageMaker Studio, upload the dataset to an S3 bucket you own, update the bucket path, and run the cells in order. **Delete the endpoint afterwards** to avoid charges.

## Tech stack
AWS SageMaker (HPO, Debugger, Profiler, Endpoints) · S3 · PyTorch · torchvision
