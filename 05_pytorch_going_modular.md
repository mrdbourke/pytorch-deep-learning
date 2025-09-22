[소스 코드 보기](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/05_pytorch_going_modular.md) | [슬라이드 보기](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/05_pytorch_going_modular.pdf) 

> **원본:** 이 문서는 [Daniel Bourke](https://github.com/mrdbourke)의 [Learn PyTorch for Deep Learning](https://github.com/mrdbourke/pytorch-deep-learning) 자료를 한국어로 번역한 것입니다. 원본 저장소: https://github.com/mrdbourke/pytorch-deep-learning

# 05. PyTorch 모듈화

이 섹션은 "노트북 코드를 Python 스크립트로 어떻게 변환할까요?"라는 질문에 답합니다.

이를 위해 [노트북 04. PyTorch 커스텀 데이터셋](https://www.learnpytorch.io/04_pytorch_custom_datasets/)의 가장 유용한 코드 셀들을 [`going_modular`](https://github.com/mrdbourke/pytorch-deep-learning/tree/main/going_modular)라는 디렉토리에 저장된 일련의 Python 스크립트로 변환할 것입니다.

## 모듈화란 무엇인가요?

모듈화는 노트북 코드(Jupyter Notebook이나 Google Colab 노트북에서)를 유사한 기능을 제공하는 일련의 다른 Python 스크립트로 변환하는 것을 포함합니다.

예를 들어, 우리는 노트북 코드를 일련의 셀에서 다음과 같은 Python 파일들로 변환할 수 있습니다:

* `data_setup.py` - 필요시 데이터를 준비하고 다운로드하는 파일.
* `engine.py` - 다양한 훈련 함수를 포함하는 파일.
* `model_builder.py` 또는 `model.py` - PyTorch 모델을 생성하는 파일.
* `train.py` - 다른 모든 파일을 활용하여 대상 PyTorch 모델을 훈련하는 파일.
* `utils.py` - 유용한 유틸리티 함수에 전념하는 파일.

> **참고:** 위 파일들의 명명과 레이아웃은 사용 사례와 코드 요구사항에 따라 달라집니다. Python 스크립트는 개별 노트북 셀만큼 일반적이므로, 거의 모든 종류의 기능에 대해 하나를 만들 수 있습니다.

## 왜 모듈화를 원할까요?

노트북은 반복적으로 탐색하고 실험을 빠르게 실행하는 데 훌륭합니다.

하지만 더 큰 규모의 프로젝트에서는 Python 스크립트가 더 재현 가능하고 실행하기 쉬울 수 있습니다.

하지만 이것은 논쟁의 여지가 있는 주제입니다. [Netflix가 프로덕션 코드에 노트북을 사용하는 방법을 보여준 것처럼](https://netflixtechblog.com/notebook-innovation-591ee3221233) 말입니다.

**프로덕션 코드**는 누군가나 무언가에게 서비스를 제공하기 위해 실행되는 코드입니다.

예를 들어, 다른 사람들이 접근하고 사용할 수 있는 온라인 앱이 있다면, 그 앱을 실행하는 코드는 **프로덕션 코드**로 간주됩니다.

그리고 fast.ai의 [`nb-dev`](https://github.com/fastai/nbdev) (노트북 개발의 줄임말)와 같은 라이브러리는 Jupyter Notebooks로 전체 Python 라이브러리(문서 포함)를 작성할 수 있게 해줍니다.

### 노트북 vs Python 스크립트의 장단점

양쪽 모두에 대한 논쟁이 있습니다.

하지만 이 목록은 주요 주제들을 요약합니다.

|               | **장점**                                               | **단점**                                     |
| ------------- | ------------------------------------------------------ | -------------------------------------------- |
| **노트북** | 실험하기/시작하기 쉬움                         | 버전 관리가 어려울 수 있음                       |
|               | 공유하기 쉬움 (예: Google Colab 노트북 링크) | 특정 부분만 사용하기 어려움              |
|               | 매우 시각적                                            | 텍스트와 그래픽이 코드를 방해할 수 있음 |

|                    | **장점**                                                                            | **단점**                                                                                  |
| ------------------ | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Python 스크립트** | 코드를 함께 패키징할 수 있음 (다른 노트북에서 유사한 코드를 다시 작성하는 것을 절약) | 실험이 시각적이지 않음 (보통 한 셀보다는 전체 스크립트를 실행해야 함) |
|                    | git을 사용한 버전 관리 가능                                                          |                                                                                           |
|                    | 많은 오픈소스 프로젝트가 스크립트 사용                                               |                                                                                           |
|                    | 더 큰 프로젝트를 클라우드 벤더에서 실행 가능 (노트북에 대한 지원이 많지 않음)     |                                                                                           |

### 제 워크플로우

저는 보통 빠른 실험과 시각화를 위해 Jupyter/Google Colab 노트북에서 머신러닝 프로젝트를 시작합니다.

그런 다음 뭔가 작동하는 것을 얻으면, 가장 유용한 코드 조각들을 Python 스크립트로 이동시킵니다.

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-my-workflow-for-experimenting.png" alt="머신러닝 코드 작성을 위한 하나의 가능한 워크플로우, jupyter 또는 google colab 노트북으로 시작한 다음 뭔가 작동하는 것을 얻으면 Python 스크립트로 이동" width=1000/>

*머신러닝 코드 작성을 위한 많은 가능한 워크플로우가 있습니다. 일부는 스크립트로 시작하는 것을 선호하고, 다른 사람들(저처럼)은 노트북으로 시작해서 나중에 스크립트로 가는 것을 선호합니다.*

### 실제 환경에서의 PyTorch

여러분의 여행에서, PyTorch 기반 ML 프로젝트의 많은 코드 저장소들이 Python 스크립트 형태로 PyTorch 코드를 실행하는 방법에 대한 지침을 가지고 있다는 것을 보게 될 것입니다.

예를 들어, 모델을 훈련하기 위해 터미널/명령줄에서 다음과 같은 코드를 실행하라는 지시를 받을 수 있습니다:

```
python train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS
```

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-python-train-command-line-annotated.png" alt="다양한 하이퍼파라미터로 PyTorch 모델을 훈련하기 위한 명령줄 호출" width=1000/> 

*다양한 하이퍼파라미터 설정으로 명령줄에서 PyTorch `train.py` 스크립트 실행.*

이 경우, `train.py`는 대상 Python 스크립트이며, PyTorch 모델을 훈련하는 함수들을 포함할 것입니다.

그리고 `--model`, `--batch_size`, `--lr`, `--num_epochs`는 인수 플래그로 알려져 있습니다.

이것들을 원하는 값으로 설정할 수 있으며, `train.py`와 호환되면 작동하고, 그렇지 않으면 오류가 발생합니다.

예를 들어, 노트북 04의 TinyVGG 모델을 배치 크기 32, 학습률 0.001로 10 에포크 동안 훈련하고 싶다고 가정해봅시다:

```
python train.py --model tinyvgg --batch_size 32 --lr 0.001 --num_epochs 10
```

필요에 맞게 `train.py` 스크립트에서 이러한 인수 플래그를 원하는 만큼 설정할 수 있습니다.

최첨단 컴퓨터 비전 모델 훈련을 위한 PyTorch 블로그 포스트가 이 스타일을 사용합니다.

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-training-sota-recipe.png" alt="최첨단 컴퓨터 비전 모델 훈련을 위한 PyTorch 훈련 스크립트 레시피" width=800/>

*8개 GPU로 최첨단 컴퓨터 비전 모델을 훈련하기 위한 PyTorch 명령줄 훈련 스크립트 레시피. 출처: [PyTorch 블로그](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#the-training-recipe).*

## 우리가 다룰 내용

이 섹션의 주요 개념은: **유용한 노트북 코드 셀을 재사용 가능한 Python 파일로 변환하는 것입니다.**

이렇게 하면 같은 코드를 계속 반복해서 작성하는 것을 절약할 수 있습니다.

이 섹션에는 두 개의 노트북이 있습니다:

1. [**05. 모듈화: 파트 1 (셀 모드)**](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_cell_mode.ipynb) - 이 노트북은 전통적인 Jupyter Notebook/Google Colab 노트북으로 실행되며 [노트북 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/)의 압축된 버전입니다.
2. [**05. 모듈화: 파트 2 (스크립트 모드)**](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipynb) - 이 노트북은 1번과 동일하지만 각 주요 섹션을 `data_setup.py`와 `train.py`와 같은 Python 스크립트로 변환하는 추가 기능이 있습니다.

이 문서의 텍스트는 05. 모듈화: 파트 2 (스크립트 모드)의 코드 셀들, 즉 상단에 `%%writefile ...`이 있는 것들에 중점을 둡니다.

### 왜 두 부분인가요?

때로는 무언가를 배우는 가장 좋은 방법은 그것이 다른 것과 어떻게 *다른지* 보는 것이기 때문입니다.

각 노트북을 나란히 실행하면 어떻게 다른지 볼 수 있고, 그곳이 핵심 학습이 있는 곳입니다.

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-notebook-cell-mode-vs-script-mode.png" alt="셀 모드 노트북 vs 스크립트 모드 노트북 실행" width=1000/>

*섹션 05의 두 노트북을 나란히 실행. **스크립트 모드 노트북에는 셀 모드 노트북의 코드를 Python 스크립트로 변환하는 추가 코드 셀들**이 있다는 것을 알 수 있습니다.*

### 우리가 목표로 하는 것

이 섹션의 끝에서 우리는 두 가지를 갖고 싶습니다:

1. 명령줄에서 한 줄의 코드로 노트북 04에서 구축한 모델(Food Vision Mini)을 훈련할 수 있는 능력: `python train.py`.
2. 재사용 가능한 Python 스크립트의 디렉토리 구조, 예를 들어: 

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

### 주의할 점

* **독스트링** - 재현 가능하고 이해하기 쉬운 코드를 작성하는 것이 중요합니다. 이를 염두에 두고, 스크립트에 넣을 함수/클래스들은 Google의 [Python 독스트링 스타일](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)을 염두에 두고 생성되었습니다.
* **스크립트 상단의 임포트** - 우리가 만들 Python 스크립트들은 모두 자체적으로 작은 프로그램으로 간주될 수 있으므로, 모든 스크립트는 스크립트 시작 부분에서 입력 모듈을 임포트해야 합니다. 예를 들어:

```python
# train.py에 필요한 모듈 임포트
import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms
```

## 어디서 도움을 받을 수 있나요?

이 코스의 모든 자료는 [GitHub에서 사용할 수 있습니다](https://github.com/mrdbourke/pytorch-deep-learning).

문제가 발생하면 코스 [GitHub Discussions 페이지](https://github.com/mrdbourke/pytorch-deep-learning/discussions)에서 질문할 수 있습니다.

물론 [PyTorch 문서](https://pytorch.org/docs/stable/index.html)와 [PyTorch 개발자 포럼](https://discuss.pytorch.org/)도 있습니다. 이는 PyTorch와 관련된 모든 것에 대해 매우 도움이 되는 곳입니다. 

## 0. 셀 모드 vs. 스크립트 모드

[05. 모듈화 파트 1 (셀 모드)](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_cell_mode.ipynb)와 같은 셀 모드 노트북은 일반적으로 실행되는 노트북으로, 노트북의 각 셀은 코드이거나 마크다운입니다.

[05. 모듈화 파트 2 (스크립트 모드)](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipynb)와 같은 스크립트 모드 노트북은 셀 모드 노트북과 매우 유사하지만, 많은 코드 셀들이 Python 스크립트로 변환될 수 있습니다.

> **참고:** 노트북을 통해 Python 스크립트를 만들 *필요는 없습니다*. [VS Code](https://code.visualstudio.com/)와 같은 IDE(통합 개발 환경)를 통해 직접 만들 수 있습니다. 이 섹션의 일부로 스크립트 모드 노트북을 갖는 것은 노트북에서 Python 스크립트로 가는 한 가지 방법을 보여주기 위한 것입니다.

## 1. 데이터 가져오기

05 노트북들 각각에서 데이터를 가져오는 것은 [노트북 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/#1-get-data)와 동일하게 발생합니다.

Python의 `requests` 모듈을 통해 GitHub에 호출하여 `.zip` 파일을 다운로드하고 압축을 해제합니다.

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

이것은 피자, 스테이크, 스시 이미지가 표준 이미지 분류 형식으로 포함된 `pizza_steak_sushi`라는 다른 디렉토리를 포함하는 `data`라는 파일을 갖게 됩니다.

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

## 2. 데이터셋과 데이터로더 생성하기 (`data_setup.py`)

데이터를 얻으면, PyTorch `Dataset`과 `DataLoader`로 변환할 수 있습니다 (훈련 데이터용 하나, 테스트 데이터용 하나).

유용한 `Dataset`과 `DataLoader` 생성 코드를 `create_dataloaders()`라는 함수로 변환합니다.

그리고 `%%writefile going_modular/data_setup.py` 라인을 사용하여 파일에 작성합니다. 

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
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
```

`DataLoader`를 만들고 싶다면 이제 `data_setup.py` 내의 함수를 다음과 같이 사용할 수 있습니다:

```python
# data_setup.py 임포트
from going_modular import data_setup

# 훈련/테스트 데이터로더 생성하고 클래스 이름을 리스트로 가져오기
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(...)
```

## 3. 모델 만들기 (`model_builder.py`)

지난 몇 개의 노트북들(노트북 03과 노트북 04)에서 TinyVGG 모델을 몇 번 구축했습니다.

따라서 모델을 자체 파일에 넣어서 계속 재사용할 수 있도록 하는 것이 합리적입니다.

`TinyVGG()` 모델 클래스를 `%%writefile going_modular/model_builder.py` 라인으로 스크립트에 넣어봅시다:

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
      # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion
```

이제 매번 TinyVGG 모델을 처음부터 코딩하는 대신, 다음과 같이 임포트할 수 있습니다:

```python
import torch
# model_builder.py 임포트
from going_modular import model_builder
device = "cuda" if torch.cuda.is_available() else "cpu"

# "model_builder.py" 스크립트에서 모델의 인스턴스 생성
torch.manual_seed(42)
model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=10, 
                              output_shape=len(class_names)).to(device)
```

## 4. `train_step()`과 `test_step()` 함수 생성하기 및 `train()`으로 결합하기

[노트북 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/#75-create-train-test-loop-functions)에서 여러 훈련 함수를 작성했습니다:

1. `train_step()` - 모델, `DataLoader`, 손실 함수, 옵티마이저를 받아서 `DataLoader`에서 모델을 훈련합니다.
2. `test_step()` - 모델, `DataLoader`, 손실 함수를 받아서 `DataLoader`에서 모델을 평가합니다.
3. `train()` - 주어진 에포크 수에 대해 1번과 2번을 함께 수행하고 결과 딕셔너리를 반환합니다.

이것들이 우리 모델 훈련의 *엔진*이 될 것이므로, `%%writefile going_modular/engine.py` 라인으로 `engine.py`라는 Python 스크립트에 모두 넣을 수 있습니다:

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

이제 `engine.py` 스크립트를 갖게 되었으므로, 다음과 같이 함수를 임포트할 수 있습니다:

```python
# engine.py 임포트
from going_modular import engine

# engine.py에서 호출하여 train() 사용
engine.train(...)
```

## 5. 모델을 저장하는 함수 생성하기 (`utils.py`)

훈련 중이거나 훈련 후에 모델을 저장하고 싶을 때가 많습니다.

이전 노트북들에서 모델을 저장하는 코드를 몇 번 작성했으므로, 이를 함수로 변환하여 파일에 저장하는 것이 합리적입니다.

유틸리티 함수를 `utils.py`(utilities의 줄임말)라는 파일에 저장하는 것이 일반적인 관행입니다.

`save_model()` 함수를 `%%writefile going_modular/utils.py` 라인으로 `utils.py`라는 파일에 저장해봅시다: 

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

이제 `save_model()` 함수를 사용하고 싶다면, 다시 모두 작성하는 대신 다음과 같이 임포트하고 사용할 수 있습니다:

```python
# utils.py 임포트
from going_modular import utils

# 모델을 파일에 저장
save_model(model=...
           target_dir=...,
           model_name=...)
```

## 6. 모델 훈련, 평가 및 저장하기 (`train.py`)

앞서 논의한 바와 같이, PyTorch 저장소들이 모든 기능을 `train.py` 파일에 함께 결합하는 것을 자주 보게 될 것입니다.

이 파일은 본질적으로 "사용 가능한 모든 데이터를 사용하여 모델을 훈련하라"고 말하고 있습니다.

우리의 `train.py` 파일에서, 우리가 만든 다른 Python 스크립트들의 모든 기능을 결합하여 모델을 훈련하는 데 사용할 것입니다.

이렇게 하면 명령줄에서 한 줄의 코드로 PyTorch 모델을 훈련할 수 있습니다:

```
python train.py
```

`train.py`를 만들기 위해 다음 단계들을 거칠 것입니다:

1. `torch`, `os`, `torchvision.transforms`와 `going_modular` 디렉토리의 모든 스크립트들인 `data_setup`, `engine`, `model_builder`, `utils`를 포함한 다양한 의존성을 임포트합니다.
  * **참고:** `train.py`가 `going_modular` 디렉토리 *내부*에 있을 것이므로, `from going_modular import ...` 대신 `import ...`를 통해 다른 모듈들을 임포트할 수 있습니다.
2. 배치 크기, 에포크 수, 학습률, 은닉 유닛 수와 같은 다양한 하이퍼파라미터를 설정합니다 (이것들은 나중에 [Python의 `argparse`](https://docs.python.org/3/library/argparse.html)를 통해 설정할 수 있습니다).
3. 훈련 및 테스트 디렉토리를 설정합니다.
4. 디바이스 독립적인 코드를 설정합니다.
5. 필요한 데이터 변환을 생성합니다.
6. `data_setup.py`를 사용하여 DataLoader를 생성합니다.
7. `model_builder.py`를 사용하여 모델을 생성합니다.
8. 손실 함수와 옵티마이저를 설정합니다.
9. `engine.py`를 사용하여 모델을 훈련합니다.
10. `utils.py`를 사용하여 모델을 저장합니다. 

그리고 `%%writefile going_modular/train.py` 라인을 사용하여 노트북 셀에서 파일을 생성할 수 있습니다:

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

와!

이제 명령줄에서 다음 라인을 실행하여 PyTorch 모델을 훈련할 수 있습니다:

```
python train.py
```

이렇게 하면 우리가 만든 다른 모든 코드 스크립트들을 활용할 것입니다.

원한다면, Python의 `argparse` 모듈을 사용하여 인수 플래그 입력을 사용하도록 `train.py` 파일을 조정할 수 있습니다. 이렇게 하면 앞서 논의한 것처럼 다른 하이퍼파라미터 설정을 제공할 수 있습니다:

```
python train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS
```

## 연습문제

**자료:**

* [05번 연습문제 템플릿 노트북](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/05_pytorch_going_modular_exercise_template.ipynb)
* [05번 예제 해답 노트북](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/solutions/05_pytorch_going_modular_exercise_solutions.ipynb)
    * [YouTube에서 05번 해답 노트북 라이브 코딩 실행](https://youtu.be/ijgFhMK3pp4)

**연습문제:**

1. 데이터를 가져오는 코드(위의 섹션 1. 데이터 가져오기에서)를 `get_data.py`와 같은 Python 스크립트로 변환하세요.
    * `python get_data.py`를 사용하여 스크립트를 실행할 때 데이터가 이미 존재하는지 확인하고, 존재한다면 다운로드를 건너뛰어야 합니다.
    * 데이터 다운로드가 성공하면 `data` 디렉토리에서 `pizza_steak_sushi` 이미지에 접근할 수 있어야 합니다.
2. [Python의 `argparse` 모듈](https://docs.python.org/3/library/argparse.html)을 사용하여 훈련 절차를 위해 `train.py`에 커스텀 하이퍼파라미터 값을 보낼 수 있도록 하세요.
    * 다음에 대해 다른 것을 사용하는 인수를 추가하세요:
        * 훈련/테스트 디렉토리
        * 학습률
        * 배치 크기
        * 훈련할 에포크 수
        * TinyVGG 모델의 은닉 유닛 수
    * 위 인수들 각각의 기본값을 노트북 05에서와 동일하게 유지하세요.
    * 예를 들어, 학습률 0.003, 배치 크기 64로 20 에포크 동안 TinyVGG 모델을 훈련하려면 다음과 같은 라인을 실행할 수 있어야 합니다: `python train.py --learning_rate 0.003 --batch_size 64 --num_epochs 20`.
    * **참고:** `train.py`가 섹션 05에서 만든 다른 스크립트들(`model_builder.py`, `utils.py`, `engine.py` 등)을 활용하므로, 이것들도 사용할 수 있도록 해야 합니다. 이것들은 [코스 GitHub의 `going_modular` 폴더](https://github.com/mrdbourke/pytorch-deep-learning/tree/main/going_modular/going_modular)에서 찾을 수 있습니다.
3. 저장된 모델이 있는 파일 경로가 주어진 대상 이미지에 대해 예측하는 스크립트(예: `predict.py`)를 생성하세요.
    * 예를 들어, `python predict.py some_image.jpeg` 명령을 실행할 수 있어야 하며, 훈련된 PyTorch 모델이 이미지에 대해 예측하고 예측 결과를 반환해야 합니다.
    * 예제 예측 코드를 보려면 [노트북 04의 커스텀 이미지 예측 섹션](https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function)을 확인하세요.
    * 훈련된 모델을 로드하는 코드도 작성해야 할 수 있습니다.

## 추가 학습

* Python 프로젝트 구조화에 대해 더 배우려면 Real Python의 [Python Application Layouts](https://realpython.com/python-application-layouts/) 가이드를 확인하세요.
* PyTorch 코드 스타일링에 대한 아이디어를 얻으려면 [Igor Susmelj의 PyTorch 스타일 가이드](https://github.com/IgorSusmelj/pytorch-styleguide#recommended-code-structure-for-training-your-model)를 확인하세요 (이 챕터의 많은 스타일링이 이 가이드 + 다양한 유사한 PyTorch 저장소들을 기반으로 합니다).
* 최첨단 이미지 분류 모델을 훈련하기 위해 PyTorch 팀이 작성한 예제 `train.py` 스크립트와 다양한 다른 PyTorch 스크립트를 보려면 GitHub의 [`classification` 저장소](https://github.com/pytorch/vision/tree/main/references/classification)를 확인하세요. 
