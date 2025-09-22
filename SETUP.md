# PyTorch 코딩을 위한 환경 설정

> **원본:** 이 문서는 [Daniel Bourke](https://github.com/mrdbourke)의 [Learn PyTorch for Deep Learning](https://github.com/mrdbourke/pytorch-deep-learning) 자료를 한국어로 번역한 것입니다. 원본 저장소: https://github.com/mrdbourke/pytorch-deep-learning

딥러닝 코딩을 위한 머신 설정은 꽤 복잡할 수 있습니다.

하드웨어부터 소프트웨어까지, 그리고 여러분의 머신에서 실행된 것처럼 다른 사람의 머신에서도 코드가 실행되도록 하는 모든 작은 부분들까지.

이 코스의 목적상, 우리는 간단하게 유지하겠습니다.

하지만 여기서 사용하는 것을 다른 곳에서도 사용할 수 없을 정도로 간단하지는 않습니다.

두 가지 설정 옵션이 있습니다. 하나는 다른 것보다 쉽지만, 다른 하나는 장기적으로 더 많은 옵션을 제공합니다.

1. Google Colab 사용 (가장 쉬움)
2. 자신의 로컬/원격 머신에 설정 (몇 단계가 있지만 여기서는 조금 더 유연성을 가집니다)

**참고** 이것들 중 어느 것도 [PyTorch 공식 설정 문서](https://pytorch.org/get-started/locally/)의 대체재가 아닙니다. 장기적으로 PyTorch 코딩을 시작하려면 그것들에 익숙해져야 합니다.

## 1. Google Colab으로 설정하기 (가장 쉬움)

Google Colab은 무료 온라인 대화형 컴퓨팅 엔진입니다 (데이터 사이언스 표준인 Jupyter Notebooks 기반).

Google Colab의 장점:
* 거의 제로 설정 (Google Colab은 PyTorch와 pandas, NumPy, Matplotlib 등 많은 다른 데이터 사이언스 패키지가 이미 설치되어 있음)
* 링크로 작업 공유
* GPU 무료 접근 (GPU는 딥러닝 코드를 더 빠르게 만듦), 유료 옵션으로 *더 많은* GPU 파워에 접근 가능

Google Colab의 단점:
* 타임아웃 (대부분의 Colab 노트북은 최대 2-3시간 동안만 상태를 보존하지만, 유료 옵션으로 증가 가능)
* 로컬 저장소 접근 불가 (하지만 이를 우회하는 방법들이 있음)
* 스크립팅(코드를 모듈로 변환)에는 잘 설정되지 않음

### Google Colab으로 시작하고, 필요할 때 확장하기

코스의 시작 노트북들(00-04)에서는 Google Colab만 사용할 것입니다.

이는 우리의 필요를 충분히 만족시키기 때문입니다.

사실, 이것은 제가 자주 하는 워크플로우입니다.

저는 Google Colab에서 많은 초보자용 및 실험적 작업을 합니다.

그리고 더 큰 프로젝트로 만들거나 더 작업하고 싶은 것을 발견하면, 로컬 컴퓨팅이나 클라우드 호스팅 컴퓨팅으로 이동합니다.

### Google Colab 시작하기

Google Colab을 시작하려면, 먼저 [Google Colab 소개 노트북](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)을 살펴보는 것을 추천합니다 (모든 기능과 버튼에 익숙해지기 위해서입니다).

### 원클릭으로 코스 노트북 열기

Google Colab 인터페이스에 익숙해진 후, 온라인 책 버전이나 GitHub 버전 상단의 "Open in Colab" 버튼을 눌러 코스 노트북을 Google Colab에서 직접 실행할 수 있습니다.

![Open in Colab 버튼을 통해 Google Colab에서 코스 노트북 열기](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/setup-open-in-colab-cropped.gif)

노트북의 복사본을 만들어 Google Drive에 저장하고 싶다면 "Copy to Drive" 버튼을 누를 수 있습니다.

### 링크로 Google Colab에서 노트북 열기

GitHub의 노트북 링크를 Google Colab에 직접 입력하여 동일한 결과를 얻을 수도 있습니다.

![GitHub 링크를 통해 Google Colab에서 코스 노트북 열기](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/setup-open-notebook-in-colab-via-link.png)

이렇게 하면 Google Colab에서 바로 실행 가능한 노트북을 얻을 수 있습니다.

하지만 이것은 테스트 목적으로만 사용해야 하며, 코스를 진행할 때는 기존 코드를 실행하는 것보다 **직접 코드를 작성하는** 것을 강력히 추천합니다.

### Google Colab에서 GPU 접근하기

Google Colab에서 CUDA 지원 NVIDIA GPU(CUDA는 딥러닝 코드가 GPU에서 더 빠르게 실행되도록 하는 프로그래밍 인터페이스)에 접근하려면 `Runtime -> Change runtime type -> Hardware Accelerator -> GPU`로 이동할 수 있습니다 (참고: 런타임 재시작이 필요합니다).

![Google Colab에서 GPU 접근하기](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/setup-get-gpu-colab-cropped.gif)

Google Colab에서 GPU가 활성화되어 있는지 확인하려면 다음을 실행할 수 있습니다:

```
!nvidia-smi
```

GPU에 접근할 수 있다면, 어떤 종류의 GPU에 접근할 수 있는지 보여줄 것입니다.

PyTorch가 GPU에 접근할 수 있는지 확인하려면 다음을 실행할 수 있습니다:

```python
import torch # Google Colab에는 torch가 이미 설치되어 있음
print(torch.cuda.is_available()) # PyTorch가 GPU를 사용할 수 있으면 True를 반환
```

PyTorch가 Google Colab에서 GPU를 볼 수 있다면, 위의 코드는 `True`를 출력할 것입니다.

## TK - 2. 로컬 설정하기 (Linux 버전)

> **참고:** 이것은 [로컬 설정을 위한 PyTorch 문서](https://pytorch.org/get-started/locally/)의 대체재가 아니라는 것을 상기시켜드립니다. 이것은 설정하는 한 가지 방법일 뿐이며(많은 방법이 있습니다) 이 코스를 위해 특별히 설계되었습니다.

이 **설정은 Linux 시스템에 중점을 둡니다** (세계에서 가장 일반적인 운영 체제). Windows나 macOS를 실행 중이라면 PyTorch 문서를 참고해야 합니다.

이 설정은 또한 **NVIDIA GPU에 접근할 수 있다고 가정합니다**.

왜 이 설정인가요?

머신러닝 엔지니어로서 저는 거의 매일 이것을 사용합니다. 많은 워크플로우에서 작동하며 필요에 따라 변경할 수 있을 만큼 충분히 유연합니다.

시작해봅시다.

### GPU가 있는 Linux 시스템의 로컬 설정 단계
TK TODO - CUDA 드라이버 설치 단계 추가
TK image - 코스 환경의 전체 설정 (예: conda 환경 내의 Jupyter Lab)

1. [Miniconda 설치](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) (이미 있다면 Anaconda를 사용할 수 있습니다). 중요한 것은 명령줄에서 `conda`에 접근할 수 있어야 한다는 것입니다. 다음 단계로 넘어가기 전에 Miniconda 설치 가이드의 모든 단계를 따르는지 확인하세요.
2. 코스 자료를 위한 디렉토리를 만들고, 원하는 이름으로 지정한 후 그 안으로 이동하세요. 예를 들어:
```
mkdir ztm-pytorch-course
cd ztm-pytorch-course
```
3. 방금 만든 디렉토리에서 `conda` 환경을 생성하세요. 다음 명령은 방금 만든 폴더 안에 있는 `env`라는 폴더에 살고 있는 `conda` 환경을 생성합니다 (예: `ztm-pytorch-course/env`). 아래 명령이 `y/n?`을 묻는 경우 `y`를 누르세요.
```
conda create --prefix ./env python=3.8.13
```
4. 방금 만든 환경을 활성화하세요.
```
conda activate ./env
```
5. PyTorch와 GPU에서 PyTorch를 실행하기 위한 CUDA Toolkit과 같이 코스에 필요한 코드 의존성을 설치하세요. 이 모든 것을 동시에 실행할 수 있습니다 (**참고:** 이것은 NVIDIA GPU가 있는 Linux 시스템을 위한 것이며, 다른 옵션은 [PyTorch 설정 문서](https://pytorch.org/get-started/locally/)를 참고하세요):
```
conda install -c pytorch pytorch=1.10.0 torchvision cudatoolkit=11.3 -y
conda install -c conda-forge jupyterlab torchinfo torchmetrics -y
conda install -c anaconda pip -y
conda install pandas matplotlib scikit-learn -y
```
6. Jupyter Lab 서버를 시작하여 설치가 올바르게 실행되었는지 확인하세요:

```bash
jupyter lab
```

7. Jupyter Lab이 실행된 후, Jupyter Notebook을 시작하고 셀에서 다음 코드를 실행하세요.
```python
import pandas as pd
import numpy as np
import torch
import sklearn
import matplotlib
import torchinfo, torchmetrics

# PyTorch 접근 확인 (텐서를 출력해야 함)
print(torch.randn(3, 3))

# GPU 확인 (True를 반환해야 함)
print(torch.cuda.is_available())
```

위의 코드가 오류 없이 실행되면 준비가 완료된 것입니다.

오류가 발생하면 [Learn PyTorch GitHub Discussions 페이지](https://github.com/mrdbourke/pytorch-deep-learning/discussions)를 참고하여 질문하거나 [PyTorch 설정 문서 페이지](https://pytorch.org/get-started/locally/)를 참고하세요.