# PyTorch 2.0 Brief Testing Results

## Setup
* Model: ResNet50 (from TorchVision)
* Data: CIFAR10 (from TorchVision)
* Epochs: 5 (single run) and 3x 5 (multiple runs)
* Batch size: 128
* Image size: 224

See full code: TK - link

## Single run (5 epochs once)

### NVIDIA RTX 4080

![]("figures/single_run_NVIDIA GeForce RTX 4080_ResNet50_CIFAR10_224_train_epoch_time.png")

### NVIDIA A100

![]("figures/single_run_NVIDIA A100-SXM4-40GB_ResNet50_CIFAR10_224_train_epoch_time.png")

## Multi run (5 epochs 3x)

### NVIDIA RTX 4080

![]("figures/multi_run_NVIDIA GeForce RTX 4080_ResNet50_CIFAR10_224_train_epoch_time.png")

### NVIDIA A100

![]("figures/multi_run_NVIDIA A100-SXM4-40GB_ResNet50_CIFAR10_224_train_epoch_time.png")