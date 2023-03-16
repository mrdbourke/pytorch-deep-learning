# PyTorch 2.0 Brief Testing Results

## Setup
* **Model:** ResNet50 (from TorchVision)
* **Data:** CIFAR10 (from TorchVision)
* **Epochs:** 5 (single run) and 3x 5 (multiple runs)
* **Batch size:** 128
* **Image size:** 224

See full code in the [PyTorch 2.0 Intro notebook](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/pytorch_2_intro.ipynb).

## Single run (5 epochs once)

### NVIDIA RTX 4080

![results of training a resnet50 model on a nvidia rtx 4080 for 5 epochs with a batch size of 128 and image size of 224](figures/single_run_NVIDIA_GeForce_RTX_4080_ResNet50_CIFAR10_224_train_epoch_time.png)

### NVIDIA A100

![results of training a resnet50 model on a nvidia a100 for 5 epochs with a batch size of 128 and image size of 224](figures/single_run_NVIDIA_A100-SXM4-40GB_ResNet50_CIFAR10_224_train_epoch_time.png)

## Multi run (5 epochs 3x)

### NVIDIA RTX 4080

![results of training a resnet50 model on an rtx 4080 for 5 epochs with a batch size of 128 and image size of 224 for 3 rounds](figures/multi_run_NVIDIA_GeForce_RTX_4080_ResNet50_CIFAR10_224_train_epoch_time.png)

### NVIDIA A100

![results of training a resnet50 model on a nvidia a100 for 5 epochs with a batch size of 128 and image size of 224 for 3 rounds](figures/multi_run_NVIDIA_A100-SXM4-40GB_ResNet50_CIFAR10_224_train_epoch_time.png)