# PyTorch 딥러닝 학습하기

> **원본:** 이 저장소는 [Daniel Bourke](https://github.com/mrdbourke)의 [Learn PyTorch for Deep Learning](https://github.com/mrdbourke/pytorch-deep-learning) 자료를 한국어로 번역한 것입니다. 원본 저장소: https://github.com/mrdbourke/pytorch-deep-learning

[Zero to Mastery Learn PyTorch for Deep Learning 코스](https://dbourke.link/ZTMPyTorch)에 오신 것을 환영합니다. 이는 인터넷에서 PyTorch를 배우기에 두 번째로 좋은 곳입니다 (첫 번째는 [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)입니다).

* **2023년 4월 업데이트:** 새로운 [PyTorch 2.0 튜토리얼](https://www.learnpytorch.io/pytorch_2_intro/)이 출시되었습니다! PyTorch 2.0은 추가 기능과 하위 호환성을 제공하는 릴리스이므로, 이전의 모든 코스 자료들이 *여전히* PyTorch 2.0과 함께 작동합니다.

<div align="center">
    <a href="https://learnpytorch.io">
        <img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/misc-pytorch-course-launch-cover-white-text-black-background.jpg" width=750 alt="pytorch deep learning by zero to mastery cover photo with different sections of the course">
    </a>
</div>

## 이 페이지의 내용

* [코스 자료/개요](https://github.com/mrdbourke/pytorch-deep-learning#course-materialsoutline)
* [이 코스에 대해](https://github.com/mrdbourke/pytorch-deep-learning#about-this-course)
* [상태](https://github.com/mrdbourke/pytorch-deep-learning#status) (코스 제작 진행 상황)
* [로그](https://github.com/mrdbourke/pytorch-deep-learning#log) (코스 자료 제작 과정의 로그)

## 코스 자료/개요

* 📖 **온라인 책 버전:** 모든 코스 자료는 [learnpytorch.io](https://learnpytorch.io)에서 읽기 가능한 온라인 책으로 제공됩니다.
* 🎥 **YouTube의 첫 5개 섹션:** [첫 25시간 분량의 자료](https://youtu.be/Z_ikDlimN6A)를 시청하여 하루 만에 PyTorch를 배워보세요.
* 🔬 **코스 초점:** 코드, 코드, 코드, 실험, 실험, 실험.
* 🏃‍♂️ **교육 스타일:** [https://sive.rs/kimo](https://sive.rs/kimo).
* 🤔 **질문하기:** 기존 질문을 보거나 새로운 질문을 하려면 [GitHub Discussions 페이지](https://github.com/mrdbourke/pytorch-deep-learning/discussions)를 참고하세요.

| **섹션** | **다루는 내용** | **연습문제 & 추가 학습** | **슬라이드** |
| ----- | ----- | ----- | ----- |
| [00 - PyTorch 기초](https://www.learnpytorch.io/00_pytorch_fundamentals/) | 딥러닝과 신경망에 사용되는 많은 기본적인 PyTorch 연산들. | [연습문제 & 추가 학습으로 이동](https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises) | [슬라이드로 이동](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/00_pytorch_and_deep_learning_fundamentals.pdf) |
| [01 - PyTorch 워크플로우](https://www.learnpytorch.io/01_pytorch_workflow/) | 딥러닝 문제에 접근하고 PyTorch로 신경망을 구축하는 방법에 대한 개요를 제공합니다. | [연습문제 & 추가 학습으로 이동](https://www.learnpytorch.io/01_pytorch_workflow/#exercises) | [슬라이드로 이동](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/01_pytorch_workflow.pdf) |
| [02 - PyTorch 신경망 분류](https://www.learnpytorch.io/02_pytorch_classification/) | 01의 PyTorch 워크플로우를 사용하여 신경망 분류 문제를 다룹니다. | [연습문제 & 추가 학습으로 이동](https://www.learnpytorch.io/02_pytorch_classification/#exercises) | [슬라이드로 이동](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/02_pytorch_classification.pdf) |
| [03 - PyTorch 컴퓨터 비전](https://www.learnpytorch.io/03_pytorch_computer_vision/) | 01 & 02의 동일한 워크플로우를 사용하여 PyTorch가 컴퓨터 비전 문제에 어떻게 사용될 수 있는지 살펴봅시다. | [연습문제 & 추가 학습으로 이동](https://www.learnpytorch.io/03_pytorch_computer_vision/#exercises) | [슬라이드로 이동](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/03_pytorch_computer_vision.pdf) |
| [04 - PyTorch 커스텀 데이터셋](https://www.learnpytorch.io/04_pytorch_custom_datasets/) | PyTorch에 커스텀 데이터셋을 어떻게 로드할까요? 또한 이 노트북에서 모듈화된 코드(05에서 다룸)의 기초를 다질 것입니다. | [연습문제 & 추가 학습으로 이동](https://www.learnpytorch.io/04_pytorch_custom_datasets/#exercises) | [슬라이드로 이동](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/04_pytorch_custom_datasets.pdf) |
| [05 - PyTorch 모듈화](https://www.learnpytorch.io/05_pytorch_going_modular/) | PyTorch는 모듈화되도록 설계되었습니다. 우리가 만든 것을 일련의 Python 스크립트로 변환해봅시다 (실제로 PyTorch 코드를 찾을 때 자주 보게 될 방식입니다). | [연습문제 & 추가 학습으로 이동](https://www.learnpytorch.io/05_pytorch_going_modular/#exercises) | [슬라이드로 이동](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/05_pytorch_going_modular.pdf) |
| [06 - PyTorch 전이 학습](https://www.learnpytorch.io/06_pytorch_transfer_learning/) | 잘 작동하는 사전 훈련된 모델을 가져와서 우리만의 문제 중 하나에 맞게 조정해봅시다. | [연습문제 & 추가 학습으로 이동](https://www.learnpytorch.io/06_pytorch_transfer_learning/#exercises) | [슬라이드로 이동](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/06_pytorch_transfer_learning.pdf) |
| [07 - 마일스톤 프로젝트 1: PyTorch 실험 추적](https://www.learnpytorch.io/07_pytorch_experiment_tracking/) | 우리는 많은 모델을 구축했습니다... 그들이 모두 어떻게 작동하는지 추적하는 것이 좋지 않을까요? | [연습문제 & 추가 학습으로 이동](https://www.learnpytorch.io/07_pytorch_experiment_tracking/#exercises) | [슬라이드로 이동](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/07_pytorch_experiment_tracking.pdf) |
| [08 - 마일스톤 프로젝트 2: PyTorch 논문 재현](https://www.learnpytorch.io/08_pytorch_paper_replicating/) | PyTorch는 머신러닝 연구를 위한 가장 인기 있는 딥러닝 프레임워크입니다. 머신러닝 논문을 재현하여 그 이유를 알아봅시다. | [연습문제 & 추가 학습으로 이동](https://www.learnpytorch.io/08_pytorch_paper_replicating/#exercises) | [슬라이드로 이동](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/08_pytorch_paper_replicating.pdf) |
| [09 - 마일스톤 프로젝트 3: 모델 배포](https://www.learnpytorch.io/09_pytorch_model_deployment/) | 작동하는 PyTorch 모델을 구축했습니다... 다른 사람들이 사용할 수 있도록 하려면 어떻게 해야 할까요? 힌트: 인터넷에 배포하세요. | [연습문제 & 추가 학습으로 이동](https://www.learnpytorch.io/09_pytorch_model_deployment/#exercises) | [슬라이드로 이동](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/09_pytorch_model_deployment.pdf) |
| [PyTorch 추가 자료](https://www.learnpytorch.io/pytorch_extra_resources/) | 이 코스는 PyTorch와 딥러닝의 많은 부분을 다루지만 머신러닝 분야는 광범위합니다. 여기서 PyTorch와 딥러닝, ML 엔지니어링, NLP(자연어 처리), 시계열 데이터, 데이터셋을 찾을 수 있는 곳 등에 대한 추천 도서와 자료를 찾을 수 있습니다. | - | - |
| [PyTorch 치트시트](https://www.learnpytorch.io/pytorch_cheatsheet/) | PyTorch의 주요 기능들에 대한 매우 빠른 개요와 코스와 PyTorch 문서에서 더 많은 정보를 찾을 수 있는 다양한 자료에 대한 링크입니다. | - | - |
| [빠른 PyTorch 2.0 튜토리얼](https://www.learnpytorch.io/pytorch_2_intro/) | PyTorch 2.0의 매우 빠른 소개, 새로운 기능과 시작하는 방법, 더 배울 수 있는 자료들입니다. | - | - |

## 상태

모든 자료가 완성되고 Zero to Mastery에서 비디오가 게시되었습니다!

진행 중인 작업 보드는 프로젝트 페이지를 참고하세요 - https://github.com/users/mrdbourke/projects/1 

* **총 비디오 수:** 321
* **완료된 스켈레톤 코드:** 00, 01, 02, 03, 04, 05, 06, 07, 08, 09
* **완료된 주석 (텍스트):** 00, 01, 02, 03, 04, 05, 06, 07, 08, 09
* **완료된 이미지:** 00, 01, 02, 03, 04, 05, 06, 07, 08, 09
* **완료된 키노트:** 00, 01, 02, 03, 04, 05, 06, 07, 08, 09
* **완료된 연습문제와 해답:** 00, 01, 02, 03, 04, 05, 06, 07, 08, 09

거의 매일 업데이트되는 내용은 [로그](https://github.com/mrdbourke/pytorch-deep-learning#log)를 참고하세요.

## 이 코스에 대해

### 이 코스는 누구를 위한 것인가요?

**여러분:** 머신러닝이나 딥러닝 분야의 초보자이고 PyTorch를 배우고 싶어합니다.

**이 코스:** 실습 위주, 코드 우선 방식으로 PyTorch와 많은 머신러닝 개념을 가르칩니다.

이미 1년 이상의 머신러닝 경험이 있다면 이 코스가 도움이 될 수 있지만, 특히 초보자 친화적으로 설계되었습니다.

### 전제 조건은 무엇인가요?

1. 3-6개월간 Python 코딩 경험.
2. 최소 하나의 초보자용 머신러닝 코스 (하지만 이것은 건너뛸 수 있을 수도 있습니다. 다양한 주제에 대한 자료가 링크되어 있습니다).
3. Jupyter Notebooks 또는 Google Colab 사용 경험 (하지만 진행하면서 배울 수 있습니다).
4. 학습하려는 의지 (가장 중요합니다).

1번과 2번의 경우, [Zero to Mastery Data Science and Machine Learning Bootcamp](https://dbourke.link/ZTMMLcourse)를 추천합니다. 머신러닝과 Python의 기초를 가르쳐줄 것입니다 (편견이 있지만, 저도 그 코스를 가르칩니다).

### 코스는 어떻게 진행되나요?

모든 코스 자료는 [learnpytorch.io](https://learnpytorch.io)에서 온라인 책으로 무료로 제공됩니다. 읽기를 좋아한다면 그곳의 자료를 살펴보는 것을 추천합니다.

비디오를 통한 학습을 선호한다면, 이 코스는 도제식 형식으로도 진행됩니다. 즉, 제가 PyTorch 코드를 작성하고, 여러분도 PyTorch 코드를 작성합니다.

코스 모토에 *의심스러우면 코드를 실행하라*와 *실험, 실험, 실험!*이 포함된 이유가 있습니다.

제 전체 목표는 여러분이 한 가지를 할 수 있도록 돕는 것입니다: PyTorch 코드를 작성하여 머신러닝을 배우는 것입니다.

코드는 모두 [Google Colab Notebooks](https://colab.research.google.com)를 통해 작성됩니다 (Jupyter Notebooks도 사용할 수 있습니다). 이는 머신러닝을 실험하기 위한 놀라운 무료 리소스입니다.

### 코스를 완료하면 무엇을 얻을 수 있나요?

비디오를 시청하면 수료증과 그런 것들이 있습니다.

하지만 수료증은 그냥 그런 거죠.

이 코스를 머신러닝 모멘텀 빌더로 생각할 수 있습니다.

마지막에는 수백 줄의 PyTorch 코드를 작성하게 될 것입니다.

그리고 머신러닝의 가장 중요한 개념들 중 많은 것들을 접하게 될 것입니다.

따라서 자신만의 머신러닝 프로젝트를 구축하거나 PyTorch로 만들어진 공개 머신러닝 프로젝트를 검사할 때, 친숙하게 느껴질 것이고 그렇지 않더라도 최소한 어디를 봐야 할지 알게 될 것입니다.

### 코스에서 무엇을 구축하게 되나요?

PyTorch와 머신러닝의 기본적인 기초부터 시작하므로, 머신러닝이 처음이라도 빠르게 따라잡을 수 있습니다.

그런 다음 PyTorch 신경망 분류, PyTorch 워크플로우, 컴퓨터 비전, 커스텀 데이터셋, 실험 추적, 모델 배포, 그리고 제가 개인적으로 좋아하는 전이 학습 등 더 고급 영역을 탐구할 것입니다. 전이 학습은 한 머신러닝 모델이 다른 문제에서 배운 것을 가져와서 자신의 문제에 적용하는 강력한 기법입니다!

그 과정에서 음식 이미지를 분류하는 신경망 컴퓨터 비전 모델인 FoodVision이라는 전체 프로젝트를 중심으로 세 개의 마일스톤 프로젝트를 구축하게 됩니다.

이러한 마일스톤 프로젝트들은 PyTorch를 사용하여 중요한 머신러닝 개념을 다루는 연습을 하고, 고용주에게 보여줄 수 있는 포트폴리오를 만들어 "이것이 제가 한 일입니다"라고 말할 수 있게 도와줄 것입니다.

### 어떻게 시작하나요?

어떤 기기에서든 자료를 읽을 수 있지만, 이 코스는 데스크톱 브라우저에서 함께 보면서 코딩하는 것이 가장 좋습니다.

이 코스는 Google Colab이라는 무료 도구를 사용합니다. 경험이 없다면 무료 [Google Colab 소개 튜토리얼](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)을 먼저 살펴보고 여기로 돌아오는 것을 추천합니다.

시작하려면:

1. 위의 "[00. PyTorch 기초](https://www.learnpytorch.io/00_pytorch_fundamentals/)"와 같은 노트북이나 섹션 링크 중 하나를 클릭하세요.
2. 상단의 "Open in Colab" 버튼을 클릭하세요.
3. SHIFT+Enter를 몇 번 누르고 무슨 일이 일어나는지 보세요.

### 제 질문에 답이 없어요

[토론](https://github.com/mrdbourke/pytorch-deep-learning/discussions)에 글을 남기거나 직접 이메일을 보내주세요: daniel (at) mrdbourke (dot) com.

## 로그

무슨 일이 일어나고 있는지 거의 매일 업데이트됩니다.

* 15 May 2023 - PyTorch 2.0 tutorial finished + videos added to ZTM/Udemy, see code: https://www.learnpytorch.io/pytorch_2_intro/
* 13 Apr 2023 - update PyTorch 2.0 notebook
* 30 Mar 2023 - update PyTorch 2.0 notebook with more info/clean code
* 23 Mar 2023 - upgrade PyTorch 2.0 tutorial with annotations and images
* 13 Mar 2023 - add starter code for PyTorch 2.0 tutorial 
* 18 Nov 2022 - add a reference for 3 most common errors in PyTorch + links to course sections for more: https://www.learnpytorch.io/pytorch_most_common_errors/ 
* 9 Nov 2022 - add PyTorch cheatsheet for a very quick overview of the main features of PyTorch + links to course sections: https://www.learnpytorch.io/pytorch_cheatsheet/ 
* 9 Nov 2022 - full course materials (300+ videos) are now live on Udemy! You can sign up here: https://www.udemy.com/course/pytorch-for-deep-learning/?couponCode=ZTMGOODIES7 (launch deal code valid for 3-4 days from this line)
* 4 Nov 2022 - add a notebook for PyTorch Cheatsheet in `extras/` (a simple overview of many of the most important functionality of PyTorch)
* 2 Oct 2022 - all videos for section 08 and 09 published (100+ videos for the last two sections)!
* 30 Aug 2022 - recorded 15 videos for 09, total videos: 321, finished section 09 videos!!!! ... even bigger than 08!!
* 29 Aug 2022 - recorded 16 videos for 09, total videos: 306
* 28 Aug 2022 - recorded 11 videos for 09, total videos: 290
* 27 Aug 2022 - recorded 16 videos for 09, total videos: 279
* 26 Aug 2022 - add finishing touchs to notebook 09, add slides for 09, create solutions and exercises for 09
* 25 Aug 2022 - add annotations and cleanup 09, remove TK's, cleanup images, make slides for 09
* 24 Aug 2022 - add annotations to 09, main takeaways, exercises and extra-curriculum done
* 23 Aug 2022 - add annotations to 09, add plenty of images/slides
* 22 Aug 2022 - add annotations to 09, start working on slides/images
* 20 Aug 2022 - add annotations to 09 
* 19 Aug 2022 - add annotations to 09, check out the awesome demos!
* 18 Aug 2022 - add annotations to 09 
* 17 Aug 2022 - add annotations to 09
* 16 Aug 2022 - add annotations to 09
* 15 Aug 2022 - add annotations to 09
* 13 Aug 2022 - add annotations to 09
* 12 Aug 2022 - add demo files for notebook 09 to `demos/`, start annotating notebook 09 with explainer text
* 11 Aug 2022 - finish skeleton code for notebook 09, course finishes deploying 2x models, one for FoodVision Mini & one for (secret)
* 10 Aug 2022 - add section for PyTorch Extra Resources (places to learn more about PyTorch/deep learning): https://www.learnpytorch.io/pytorch_extra_resources/ 
* 09 Aug 2022 - add more skeleton code to notebook 09
* 08 Aug 2022 - create draft notebook for 09, end goal to deploy FoodVision Mini model and make it publically accessible
* 05 Aug 2022 - recorded 11 videos for 08, total videos: 263, section 08 videos finished!... the biggest section so far
* 04 Aug 2022 - recorded 13 videos for 08, total videos: 252
* 03 Aug 2022 - recorded 3 videos for 08, total videos: 239
* 02 Aug 2022 - recorded 12 videos for 08, total videos: 236
* 30 July 2022 - recorded 11 videos for 08, total videos: 224
* 29 July 2022 - add exercises + solutions for 08, see live walkthrough on YouTube: https://youtu.be/tjpW_BY8y3g
* 28 July 2022 - add slides for 08
* 27 July 2022 - cleanup much of 08, start on slides for 08, exercises and extra-curriculum next
* 26 July 2022 - add annotations and images for 08
* 25 July 2022 - add annotations for 08 
* 24 July 2022 - launched first half of course (notebooks 00-04) in a single video (25+ hours!!!) on YouTube: https://youtu.be/Z_ikDlimN6A 
* 21 July 2022 - add annotations and images for 08
* 20 July 2022 - add annotations and images for 08, getting so close! this is an epic section 
* 19 July 2022 - add annotations and images for 08
* 15 July 2022 - add annotations and images for 08 
* 14 July 2022 - add annotations for 08
* 12 July 2022 - add annotations for 08, woo woo this is bigggg section! 
* 11 July 2022 - add annotations for 08 
* 9 July 2022 - add annotations for 08
* 8 July 2022 - add a bunch of annotations to 08
* 6 July 2022 - course launched on ZTM Academy with videos for sections 00-07! 🚀 - https://dbourke.link/ZTMPyTorch 
* 1 July 2022 - add annotations and images for 08 
* 30 June 2022 - add annotations for 08
* 28 June 2022 - recorded 11 videos for section 07, total video count 213, all videos for section 07 complete!
* 27 June 2022 - recorded 11 videos for section 07, total video count 202
* 25 June 2022 - recreated 7 videos for section 06 to include updated APIs, total video count 191
* 24 June 2022 - recreated 12 videos for section 06 to include updated APIs
* 23 June 2022 - finish annotations for 07, add exercise template and solutions for 07 + video walkthrough on YouTube: https://youtu.be/cO_r2FYcAjU
* 21 June 2022 - make 08 runnable end-to-end, add images and annotations for 07
* 17 June 2022 - fix up 06, 07 v2 for upcoming torchvision version upgrade, add plenty of annotations to 08
* 13 June 2022 - add notebook 08 first version, starting to replicate the Vision Transformer paper
* 10 June 2022 - add annotations for 07 v2
* 09 June 2022 - create 07 v2 for `torchvision` v0.13 (this will replace 07 v1 when `torchvision=0.13` is released)
* 08 June 2022 - adapt 06 v2 for `torchvision` v0.13 (this will replace 06 v1 when `torchvision=0.13` is released)
* 07 June 2022 - create notebook 06 v2 for upcoming `torchvision` v0.13 update (new transfer learning methods)
* 04 June 2022 - add annotations for 07
* 03 June 2022 - huuuuuuge amount of annotations added to 07 
* 31 May 2022 - add a bunch of annotations for 07, make code runnable end-to-end
* 30 May 2022 - record 4 videos for 06, finished section 06, onto section 07, total videos 186
* 28 May 2022 - record 10 videos for 06, total videos 182
* 24 May 2022 - add solutions and exercises for 06
* 23 May 2022 - finished annotations and images for 06, time to do exercises and solutions 
* 22 May 2202 - add plenty of images to 06
* 18 May 2022 - add plenty of annotations to 06
* 17 May 2022 - added a bunch of annotations for section 06
* 16 May 2022 - recorded 10 videos for section 05, finish videos for section 05 ✅
* 12 May 2022 - added exercises and solutions for 05
* 11 May 2022 - clean up part 1 and part 2 notebooks for 05, make slides for 05, start on exercises and solutions for 05
* 10 May 2022 - huuuuge updates to the 05 section, see the website, it looks pretty: https://www.learnpytorch.io/05_pytorch_going_modular/ 
* 09 May 2022 - add a bunch of materials for 05, cleanup docs
* 08 May 2022 - add a bunch of materials for 05
* 06 May 2022 - continue making materials for 05
* 05 May 2022 - update section 05 with headings/outline
* 28 Apr 2022 - recorded 13 videos for 04, finished videos for 04, now to make materials for 05
* 27 Apr 2022 - recorded 3 videos for 04
* 26 Apr 2022 - recorded 10 videos for 04
* 25 Apr 2022 - recorded 11 videos for 04
* 24 Apr 2022 - prepared slides for 04
* 23 Apr 2022 - recorded 6 videos for 03, finished videos for 03, now to 04 
* 22 Apr 2022 - recorded 5 videos for 03
* 21 Apr 2022 - recorded 9 videos for 03
* 20 Apr 2022 - recorded 3 videos for 03
* 19 Apr 2022 - recorded 11 videos for 03
* 18 Apr 2022 - finish exercises/solutions for 04, added live-coding walkthrough of 04 exercises/solutions on YouTube: https://youtu.be/vsFMF9wqWx0
* 16 Apr 2022 - finish exercises/solutions for 03, added live-coding walkthrough of 03 exercises/solutions on YouTube: https://youtu.be/_PibmqpEyhA
* 14 Apr 2022 - add final images/annotations for 04, begin on exercises/solutions for 03 & 04
* 13 Apr 2022 - add more images/annotations for 04
* 3 Apr 2022 - add more annotations for 04
* 2 Apr 2022 - add more annotations for 04
* 1 Apr 2022 - add more annotations for 04
* 31 Mar 2022 - add more annotations for 04
* 29 Mar 2022 - add more annotations for 04
* 27 Mar 2022 - starting to add annotations for 04
* 26 Mar 2022 - making dataset for 04
* 25 Mar 2022 - make slides for 03
* 24 Mar 2022 - fix error for 03 not working in docs (finally)
* 23 Mar 2022 - add more images for 03
* 22 Mar 2022 - add images for 03
* 20 Mar 2022 - add more annotations for 03
* 18 Mar 2022 - add more annotations for 03
* 17 Mar 2022 - add more annotations for 03 
* 16 Mar 2022 - add more annotations for 03
* 15 Mar 2022 - add more annotations for 03
* 14 Mar 2022 - start adding annotations for notebook 03, see the work in progress here: https://www.learnpytorch.io/03_pytorch_computer_vision/
* 12 Mar 2022 - recorded 12 videos for 02, finished section 02, now onto making materials for 03, 04, 05
* 11 Mar 2022 - recorded 9 videos for 02
* 10 Mar 2022 - recorded 10 videos for 02
* 9 Mar 2022 - cleaning up slides/code for 02, getting ready for recording
* 8 Mar 2022 - recorded 9 videos for section 01, finished section 01, now onto 02
* 7 Mar 2022 - recorded 4 videos for section 01
* 6 Mar 2022 - recorded 4 videos for section 01
* 4 Mar 2022 - recorded 10 videos for section 01
* 20 Feb 2022 - recorded 8 videos for section 00, finished section, now onto 01
* 18 Feb 2022 - recorded 13 videos for section 00
* 17 Feb 2022 - recorded 11 videos for section 00 
* 16 Feb 2022 - added setup guide 
* 12 Feb 2022 - tidy up README with table of course materials, finish images and slides for 01
* 10 Feb 2022 - finished slides and images for 00, notebook is ready for publishing: https://www.learnpytorch.io/00_pytorch_fundamentals/
* 01-07 Feb 2022 - add annotations for 02, finished, still need images, going to work on exercises/solutions today 
* 31 Jan 2022 - start adding annotations for 02
* 28 Jan 2022 - add exercies and solutions for 01
* 26 Jan 2022 - lots more annotations to 01, should be finished tomorrow, will do exercises + solutions then too
* 24 Jan 2022 - add a bunch of annotations to 01
* 21 Jan 2022 - start adding annotations for 01 
* 20 Jan 2022 - finish annotations for 00 (still need to add images), add exercises and solutions for 00
* 19 Jan 2022 - add more annotations for 00
* 18 Jan 2022 - add more annotations for 00
* 17 Jan 2022 - back from holidays, adding more annotations to 00 
* 10 Dec 2021 - start adding annotations for 00
* 9 Dec 2021 - Created a website for the course ([learnpytorch.io](https://learnpytorch.io)) you'll see updates posted there as development continues 
* 8 Dec 2021 - Clean up notebook 07, starting to go back through code and add annotations
* 26 Nov 2021 - Finish skeleton code for 07, added four different experiments, need to clean up and make more straightforward
* 25 Nov 2021 - clean code for 06, add skeleton code for 07 (experiment tracking)
* 24 Nov 2021 - Update 04, 05, 06 notebooks for easier digestion and learning, each section should cover a max of 3 big ideas, 05 is now dedicated to turning notebook code into modular code 
* 22 Nov 2021 - Update 04 train and test functions to make more straightforward
* 19 Nov 2021 - Added 05 (transfer learning) notebook, update custom data loading code in 04
* 18 Nov 2021 - Updated vision code for 03 and added custom dataset loading code in 04
* 12 Nov 2021 - Added a bunch of skeleton code to notebook 04 for custom dataset loading, next is modelling with custom data
* 10 Nov 2021 - researching best practice for custom datasets for 04
* 9 Nov 2021 - Update 03 skeleton code to finish off building CNN model, onto 04 for loading custom datasets
* 4 Nov 2021 - Add GPU code to 03 + train/test loops + `helper_functions.py`
* 3 Nov 2021 - Add basic start for 03, going to finish by end of week
* 29 Oct 2021 - Tidied up skeleton code for 02, still a few more things to clean/tidy, created 03
* 28 Oct 2021 - Finished skeleton code for 02, going to clean/tidy tomorrow, 03 next week
* 27 Oct 2021 - add a bunch of code for 02, going to finish tomorrow/by end of week
* 26 Oct 2021 - update 00, 01, 02 with outline/code, skeleton code for 00 & 01 done, 02 next
* 23, 24 Oct 2021 - update 00 and 01 notebooks with more outline/code
* 20 Oct 2021 - add v0 outlines for 01 and 02, add rough outline of course to README, this course will focus on less but better 
* 19 Oct 2021 - Start repo 🔥, add fundamentals notebook draft v0
