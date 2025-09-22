# 05. PyTorch Going Modular

이 섹션은 "노트북 코드를 어떻게 Python 스크립트로 바꾸나요?"라는 질문에 답합니다.

이를 위해 [노트북 04. PyTorch 사용자 정의 데이터셋](https://www.learnpytorch.io/04_pytorch_custom_datasets/)에서 가장 유용한 코드 셀들을 추려 [`going_modular`](https://github.com/mrdbourke/pytorch-deep-learning/tree/main/going_modular) 디렉터리에 저장되는 일련의 Python 스크립트로 전환하겠습니다.

## 모듈화(going modular)란?

모듈화란 노트북 코드(Jupyter/Colab)를 유사한 기능을 제공하는 여러 개의 Python 스크립트로 분리·구성하는 것을 뜻합니다.

예를 들어, 노트북의 여러 셀을 다음과 같은 Python 파일들로 바꿀 수 있습니다:

* `data_setup.py` - 필요 시 데이터 준비/다운로드를 담당
* `engine.py` - 다양한 학습 함수 모음
* `model_builder.py` 또는 `model.py` - PyTorch 모델 생성
* `train.py` - 다른 파일들을 조합해 대상 PyTorch 모델 학습
* `utils.py` - 유틸리티 함수 모음

> **참고:** 위 파일들의 이름/구성은 사용 사례와 코드 요구사항에 따라 달라질 수 있습니다. Python 스크립트는 개별 노트북 셀만큼 일반적이므로 거의 모든 기능을 분리해 만들 수 있습니다.

## 왜 모듈화를 할까요?

노트북은 반복적 탐색과 빠른 실험에 탁월합니다.

하지만 규모가 큰 프로젝트에서는 Python 스크립트가 재현성과 실행 편의성 측면에서 더 유리한 경우가 많습니다.

물론 이는 논쟁의 여지가 있는 주제입니다. 예를 들어 [Netflix는 프로덕션 코드에 노트북을 활용하는 방식](https://netflixtechblog.com/notebook-innovation-591ee3221233)을 공유하기도 했습니다.

**프로덕션 코드**란 누군가(또는 무언가)에게 서비스를 제공하기 위해 실제로 운영 환경에서 실행되는 코드를 의미합니다.

예를 들어, 다른 사람들이 접근해 사용하는 온라인 앱을 운영 중이라면 그 앱을 구동하는 코드는 **프로덕션 코드**입니다.

또한 fast.ai의 [`nb-dev`](https://github.com/fastai/nbdev) 같은 도구를 사용하면 Jupyter 노트북만으로 전체 Python 라이브러리(문서 포함)를 작성할 수도 있습니다.

### 노트북 vs Python 스크립트: 장단점

양측 모두 설득력 있는 주장이 있습니다.

아래 표는 핵심만 간략히 요약한 것입니다.

|               | **장점(Pros)**                                       | **단점(Cons)**                               |
| ------------- | ----------------------------------------------------- | -------------------------------------------- |
| **Notebooks** | 실험/시작이 쉬움                                     | 버저닝이 어려울 수 있음                      |
|               | 공유가 쉬움(예: Colab 링크)                          | 특정 부분만 떼어 쓰기 어려움                 |
|               | 시각화에 강함                                        | 텍스트/그래픽이 코드 가독성을 방해할 수 있음 |

|                    | **장점(Pros)**                                                                      | **단점(Cons)**                                                                 |
| ------------------ | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Python scripts** | 코드를 모듈로 패키징 가능(여러 노트북 간 중복 작성 방지)                           | 실험 시 시각성이 떨어짐(셀 단위 실행 대신 전체 스크립트 실행 필요)            |
|                    | git으로 버전 관리 용이                                                               |                                                                                |
|                    | 많은 오픈소스 프로젝트가 스크립트 중심                                               |                                                                                |
|                    | 대규모 프로젝트를 클라우드에서 실행하기 용이(노트북은 지원이 제한적일 수 있음)       |                                                                                |

### 나의 워크플로

저는 보통 빠른 실험과 시각화를 위해 Jupyter/Colab 노트북에서 머신러닝 프로젝트를 시작합니다.

그다음 무언가 동작하기 시작하면, 가장 유용한 코드 조각들을 Python 스크립트로 옮깁니다.

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-my-workflow-for-experimenting.png" alt="one possible workflow for writing machine learning code, start with jupyter or google colab notebooks and then move to Python scripts when you've got something working." width=1000/>

*머신러닝 코드를 작성하는 워크플로는 다양합니다. 어떤 사람은 처음부터 스크립트로 시작하고, 어떤 사람(저처럼)은 노트북으로 시작해 나중에 스크립트로 옮기기도 합니다.*

### 실전에서의 PyTorch

PyTorch 기반 ML 프로젝트의 많은 저장소는 Python 스크립트 형태로 코드를 실행하는 방법을 안내합니다.

예를 들어, 터미널/명령행에서 다음과 같이 실행해 모델을 학습하라고 안내할 수 있습니다:

```
python train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS
```

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-python-train-command-line-annotated.png" alt="command line call for training a PyTorch model with different hyperparameters" width=1000/> 

*여러 하이퍼파라미터 설정으로 커맨드라인에서 PyTorch `train.py` 스크립트를 실행하는 예.*

이 경우 `train.py`는 타깃 Python 스크립트이며, PyTorch 모델을 학습하는 함수들이 들어있을 가능성이 큽니다.

`--model`, `--batch_size`, `--lr`, `--num_epochs`는 인자 플래그(argument flags)입니다.

원하는 값으로 설정할 수 있으며, `train.py`와 호환되면 동작하고 아니라면 에러가 발생합니다.

예를 들어, 노트북 04의 TinyVGG를 배치 크기 32, 학습률 0.001, 10 에포크로 학습하려면 다음과 같이 실행할 수 있습니다:

```
python train.py --model tinyvgg --batch_size 32 --lr 0.001 --num_epochs 10
```

필요에 맞춰 `train.py`에서 원하는 만큼의 인자 플래그를 지원하도록 설정할 수 있습니다.

최신 CV 모델 학습을 다룬 PyTorch 블로그 포스트도 이 스타일을 사용합니다.

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-training-sota-recipe.png" alt="PyTorch training script recipe for training state of the art computer vision models" width=800/>

*8개의 GPU로 SOTA 컴퓨터 비전 모델을 학습하는 PyTorch 커맨드라인 학습 스크립트 레시피. 출처: [PyTorch 블로그](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#the-training-recipe).* 

## 이 섹션에서 다룰 내용

핵심 개념: **노트북의 유용한 코드 셀을 재사용 가능한 Python 파일로 바꾼다.**

이렇게 하면 같은 코드를 반복 작성하지 않아도 됩니다.

이 섹션을 위한 노트북은 두 개입니다:

1. [**05. Going Modular: Part 1 (cell mode)**](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_cell_mode.ipynb) - 일반적인 Jupyter/Colab 노트북으로, [노트북 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/)를 간추린 버전입니다.
2. [**05. Going Modular: Part 2 (script mode)**](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipynb) - 1과 동일하지만, 각 주요 섹션을 `data_setup.py`, `train.py` 등 Python 스크립트로 변환하는 기능이 추가되어 있습니다.

본 문서는 `%%writefile ...`로 시작하는 셀들이 포함된 05. Going Modular: Part 2(script mode)의 코드 셀에 초점을 맞춥니다.

### 왜 두 파트로 나뉘나요?

어떤 것을 배우는 가장 좋은 방법 중 하나는 “다른 것과 어떻게 다른지”를 보는 것입니다.

두 노트북을 나란히 실행해 보면 차이점이 보이고, 거기서 핵심을 배울 수 있습니다.

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-notebook-cell-mode-vs-script-mode.png" alt="running cell mode notebook vs a script mode notebook" width=1000/>

*섹션 05의 두 노트북을 나란히 실행한 모습. **script mode 노트북에는** cell mode 코드를 Python 스크립트로 변환하는 **추가 코드 셀**이 있습니다.*

### 최종 목표

이 섹션이 끝나면 다음 두 가지를 달성합니다:

1. 커맨드라인 한 줄(`python train.py`)로 노트북 04의 모델(Food Vision Mini)을 학습할 수 있는 능력.
2. 다음과 같은 재사용 가능한 Python 스크립트 디렉터리 구조 확보: 

```
going_modular/
├── going_modular/
│   ├── data_setup.py
│   ├── engine.py
│   ├── model_builder.py
│   ├── train.py
│   └── utils.py
├── models/
│   ├── 05_going_modular_cell_mode_tinyvgg_model.pth
│   └── 05_going_modular_script_mode_tinyvgg_model.pth
└── data/
    └── pizza_steak_sushi/
        ├── train/
        │   ├── pizza/
        │   │   ├── image01.jpeg
        │   │   └── ...
        │   ├── steak/
        │   └── sushi/
        └── test/
            ├── pizza/
            ├── steak/
            └── sushi/
```

### 참고 사항

* **Docstrings** - 재현 가능하고 이해하기 쉬운 코드는 중요합니다. 이를 위해 스크립트에 들어갈 각 함수/클래스는 Google의 [Python docstring 스타일](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)을 염두에 두고 작성되었습니다.
* **스크립트 상단의 import** - 작성할 Python 스크립트들은 각각 독립된 작은 프로그램으로 볼 수 있으므로, 필요한 모듈을 스크립트 상단에서 import합니다. 예:

```python
# Import modules required for train.py
import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms
```

## 도움을 얻을 수 있는 곳

이 강의의 모든 자료는 [GitHub](https://github.com/mrdbourke/pytorch-deep-learning)에 있습니다.

문제가 생기면 강의의 [GitHub Discussions 페이지](https://github.com/mrdbourke/pytorch-deep-learning/discussions)에 질문을 남길 수 있습니다.

물론 [PyTorch 문서](https://pytorch.org/docs/stable/index.html)와 [PyTorch 개발자 포럼](https://discuss.pytorch.org/)도 있습니다. PyTorch 관련 정보가 매우 풍부합니다.

## 0. Cell mode vs. script mode

셀 모드 노트북([05. Going Modular Part 1 (cell mode)](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_cell_mode.ipynb))은 일반적으로 실행되는 노트북으로, 각 셀은 코드 또는 마크다운입니다.

스크립트 모드 노트북([05. Going Modular Part 2 (script mode)](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipynb))은 셀 모드와 매우 유사하지만, 많은 코드 셀을 Python 스크립트로 변환한다는 점이 다릅니다.

> **참고:** 노트북을 통해서만 Python 스크립트를 만들어야 하는 것은 아닙니다. [VS Code](https://code.visualstudio.com/) 같은 IDE에서 직접 만들 수도 있습니다. 이 섹션에서 스크립트 모드 노트북을 소개하는 이유는 노트북에서 Python 스크립트로 넘어가는 한 가지 방법을 보여주기 위함입니다.

## 1. 데이터 가져오기

섹션 05의 각 노트북에서 데이터 가져오기는 [노트북 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/#1-get-data)와 동일합니다.

Python의 `requests` 모듈로 GitHub에 요청해 `.zip` 파일을 다운로드하고 압축을 풉니다.

```python 
import os
import requests
import zipfile
from pathlib import Path

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it... 
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
    
# Download pizza, steak, sushi data
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza, steak, sushi data...")
    f.write(request.content)

# Unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...") 
    zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path / "pizza_steak_sushi.zip")
```

그 결과 `data` 폴더 안에 표준 이미지 분류 형식의 `pizza_steak_sushi` 디렉터리가 생성됩니다(피자/스테이크/스시 이미지 포함).

```
data/
└── pizza_steak_sushi/
    ├── train/
    │   ├── pizza/
    │   │   ├── train_image01.jpeg
    │   │   ├── test_image02.jpeg
    │   │   └── ...
    │   ├── steak/
    │   │   └── ...
    │   └── sushi/
    │       └── ...
    └── test/
        ├── pizza/
        │   ├── test_image01.jpeg
        │   └── test_image02.jpeg
        ├── steak/
        └── sushi/
```

## 2. Datasets/DataLoaders 생성 (`data_setup.py`)

데이터를 확보했으면 이를 PyTorch `Dataset`과 `DataLoader`(학습/테스트 각각)로 변환합니다.

유용한 `Dataset`/`DataLoader` 생성 코드를 `create_dataloaders()` 함수로 만들고,

`%%writefile going_modular/data_setup.py`로 파일에 작성합니다.

```py title="data_setup.py"
%%writefile going_modular/data_setup.py
"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
```

이제 `DataLoader`를 만들고 싶다면 `data_setup.py`의 함수를 다음처럼 사용할 수 있습니다:

```python
# Import data_setup.py
from going_modular import data_setup

# Create train/test dataloader and get class names as a list
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(...)
```

## 3. 모델 만들기 (`model_builder.py`)

최근 몇 개의 노트북(노트북 03, 04)에서 TinyVGG 모델을 여러 번 구현했습니다.

따라서 재사용을 위해 모델을 별도 파일로 분리하는 것이 합리적입니다.

`%%writefile going_modular/model_builder.py`로 `TinyVGG()` 모델 클래스를 스크립트에 저장해봅시다:

```python title="model_builder.py"
%%writefile going_modular/model_builder.py
"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn 

class TinyVGG(nn.Module):
  """Creates the TinyVGG architecture.

  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
  See the original architecture here: https://poloclub.github.io/cnn-explainer/
  
  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          # Where did this in_features shape come from? 
          # It's because each layer of our network compresses and changes the shape of our inputs data.
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
      )
    
  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      x = self.conv_block_2(x)
      x = self.classifier(x)
      return x
      # return self.classifier(self.block_2(self.block_1(x))) # <- leverage the benefits of operator fusion
```

이제 매번 TinyVGG를 처음부터 구현하는 대신 다음처럼 가져와 사용할 수 있습니다:

```python
import torch
# Import model_builder.py
from going_modular import model_builder
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate an instance of the model from the "model_builder.py" script
torch.manual_seed(42)
model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=10, 
                              output_shape=len(class_names)).to(device)
```

## 4. `train_step()`/`test_step()` 함수와 이를 묶는 `train()` 만들기  

[노트북 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/#75-create-train-test-loop-functions)에서 여러 학습 함수를 만들었습니다:

1. `train_step()` - 모델/`DataLoader`/손실 함수/옵티마이저를 받아 `DataLoader`로 모델을 학습
2. `test_step()` - 모델/`DataLoader`/손실 함수를 받아 `DataLoader`로 모델을 평가
3. `train()` - 주어진 에포크 동안 1과 2를 묶어 실행하고 결과 딕셔너리를 반환

이들은 학습의 *엔진* 역할을 하므로, `%%writefile going_modular/engine.py`로 `engine.py` 스크립트에 모두 담겠습니다:

```python title="engine.py"
%%writefile going_modular/engine.py
"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:
    
    (0.1112, 0.8743)
  """
  # Put model in train mode
  model.train()
  
  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0
  
  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate  and accumulate loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:
    
    (0.0223, 0.8985)
  """
  # Put model in eval mode
  model.eval() 
  
  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0
  
  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)
  
          # 1. Forward pass
          test_pred_logits = model(X)

          # 2. Calculate and accumulate loss
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()
          
          # Calculate and accumulate accuracy
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
          
  # Adjust metrics to get average loss and accuracy per batch 
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
  """
  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }
  
  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
      
      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  # Return the filled results at the end of the epochs
  return results
```

이제 `engine.py` 스크립트가 준비되었으니 다음처럼 함수들을 가져와 쓸 수 있습니다:

```python
# Import engine.py
from going_modular import engine

# Use train() by calling it from engine.py
engine.train(...)
```

## 5. 모델 저장 함수 만들기 (`utils.py`)

학습 중 또는 학습 후 모델을 저장하고 싶을 때가 많습니다.

이전 노트북들에서 모델 저장 코드를 여러 번 작성했으니, 함수로 만들어 파일에 담아두는 것이 좋겠습니다.

일반적으로 보조 함수들은 `utils.py`(utilities의 약자)에 보관합니다.

`%%writefile going_modular/utils.py`로 `save_model()` 함수를 저장하겠습니다: 

```python title="utils.py"
%%writefile going_modular/utils.py
"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
  
  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  
  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
```

이제 `save_model()` 함수를 다시 작성할 필요 없이 다음처럼 가져와 사용할 수 있습니다:

```python
# Import utils.py
from going_modular import utils

# Save a model to file
save_model(model=...
           target_dir=...,
           model_name=...)
```

## 6. 모델 학습/평가/저장 (`train.py`)

앞서 언급했듯 PyTorch 저장소에서는 여러 기능을 하나의 `train.py`로 묶는 경우가 흔합니다.

이 파일은 본질적으로 “사용 가능한 데이터를 사용해 모델을 학습하라”는 역할을 합니다.

우리의 `train.py`에서는 지금까지 만든 다른 스크립트들의 기능을 모두 조합해 모델을 학습하겠습니다.

이렇게 하면 커맨드라인 한 줄로 PyTorch 모델을 학습할 수 있습니다:

```
python train.py
```

`train.py`를 만들기 위해 다음 단계를 진행합니다:

1. 의존성 import: `torch`, `os`, `torchvision.transforms`, 그리고 `going_modular` 디렉터리의 `data_setup`, `engine`, `model_builder`, `utils`.
  * **참고:** `train.py`는 `going_modular` 디렉터리 안에 있으므로 `from going_modular import ...` 대신 `import ...`로 내부 모듈을 가져올 수 있습니다.
2. 배치 크기, 에포크 수, 학습률, 히든 유닛 수 등 하이퍼파라미터 설정(추후 [Python `argparse`](https://docs.python.org/3/library/argparse.html)로 받도록 할 수 있음).
3. 학습/테스트 디렉터리 설정.
4. 디바이스 독립적 코드 설정.
5. 필요한 데이터 변환 생성.
6. `data_setup.py`로 DataLoader 생성.
7. `model_builder.py`로 모델 생성.
8. 손실 함수와 옵티마이저 설정.
9. `engine.py`로 모델 학습.
10. `utils.py`로 모델 저장. 

노트북 셀에서 `%%writefile going_modular/train.py`로 파일을 생성합니다:

```python title="train.py"
%%writefile going_modular/train.py
"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
```

좋아요!

이제 커맨드라인에서 다음 한 줄로 PyTorch 모델을 학습할 수 있습니다:

```
python train.py
```

이렇게 하면 지금까지 만든 모든 코드 스크립트를 활용하게 됩니다.

원한다면 `train.py`에 Python의 `argparse`를 적용해 인자 플래그를 받을 수 있도록 수정하여, 앞서 언급한 것처럼 다양한 하이퍼파라미터를 커맨드라인에서 전달할 수도 있습니다:

```
python train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS
```

## 연습 문제

**자료:**

* [Exercise template notebook for 05](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/05_pytorch_going_modular_exercise_template.ipynb)
* [Example solutions notebook for 05](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/solutions/05_pytorch_going_modular_exercise_solutions.ipynb)
    * Live coding run through of [solutions notebook for 05 on YouTube](https://youtu.be/ijgFhMK3pp4)

**문제:**

1. 위 1장 “데이터 가져오기”의 코드를 `get_data.py` 같은 Python 스크립트로 바꾸세요.
    * `python get_data.py` 실행 시 이미 데이터가 있으면 다운로드를 건너뛰도록 하세요.
    * 성공적으로 다운로드되면 `data` 디렉터리에서 `pizza_steak_sushi` 이미지를 확인할 수 있어야 합니다.
2. [Python `argparse` 모듈](https://docs.python.org/3/library/argparse.html)을 사용해 `train.py`가 학습에 필요한 하이퍼파라미터를 커맨드라인에서 받을 수 있도록 하세요.
    * 다음 항목들에 대한 인자를 추가합니다:
        * 학습/테스트 디렉터리
        * 학습률(learning rate)
        * 배치 크기(batch size)
        * 학습 에포크 수
        * TinyVGG의 히든 유닛 수
    * 기본값은 노트북 05에 적힌 값과 동일하게 둡니다.
    * 예: 학습률 0.003, 배치 크기 64, 20 에포크로 TinyVGG를 학습하려면 `python train.py --learning_rate 0.003 --batch_size 64 --num_epochs 20`처럼 실행할 수 있어야 합니다.
    * **참고:** `train.py`는 05장에서 만든 `model_builder.py`, `utils.py`, `engine.py` 등을 활용하므로, 이 파일들이 함께 사용 가능해야 합니다(강의 GitHub의 [`going_modular` 폴더](https://github.com/mrdbourke/pytorch-deep-learning/tree/main/going_modular/going_modular) 참고).
3. 저장된 모델과 이미지 경로를 받아 예측하는 스크립트(예: `predict.py`)를 작성하세요.
    * 예: `python predict.py some_image.jpeg` 실행 시 학습된 PyTorch 모델이 이미지를 입력받아 예측 라벨을 출력.
    * 예측 코드 예시는 [노트북 04의 사용자 정의 이미지 예측 섹션](https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function)을 참고하세요.
    * 학습된 모델을 로드하는 코드가 필요할 수 있습니다.

## 추가 학습

* Python 프로젝트 구조화를 더 배우고 싶다면 Real Python의 [Python Application Layouts](https://realpython.com/python-application-layouts/) 가이드를 참고하세요. 
* PyTorch 코드 스타일 아이디어는 [Igor Susmelj의 PyTorch 스타일 가이드](https://github.com/IgorSusmelj/pytorch-styleguide#recommended-code-structure-for-training-your-model)를 참고하세요(이 장의 스타일링은 이 가이드와 유사한 여러 PyTorch 저장소를 바탕으로 합니다).
* SOTA 이미지 분류 모델 학습을 위한 `train.py` 및 다양한 PyTorch 스크립트 예시는 PyTorch 팀의 [`classification` GitHub 저장소](https://github.com/pytorch/vision/tree/main/references/classification)에서 확인할 수 있습니다. 