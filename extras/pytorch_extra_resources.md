# PyTorch Extra Resources

Despite the full Zero to Mastery PyTorch course being over 40 hours, you’ll likely finish being excited to learn more.

After all, the course is a PyTorch momentum builder.

The following resources are collected to extend the course.

A warning though: there’s a lot here.

Best to choose 1 or 2 resources from each section (or less) to explore more. And put the rest in your bag for later. 

Which one's the best? 

Well, if they’ve made it on this list, you can consider them a quality resource.

Most are PyTorch-specific, fitting extensions to the course but a couple are non PyTorch-specific, however, they’re still valuable in the world of machine learning.

## 🔥 Pure PyTorch resources

- [**PyTorch blog**](https://pytorch.org/blog/) — Stay up to date on the latest from PyTorch right from the source. I check the blog once a month or so for updates.
- [**PyTorch documentation**](https://pytorch.org/docs) — We’ll have explored this plenty throughout the course but there’s still a large amount we haven’t touched. No trouble, explore often and when necessary.
- [**PyTorch Performance Tuning Guide**](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#) — One of the first things you’ll likely want to do after the course is to make your PyTorch models faster (training and inference), the PyTorch Performance Tuning Guide helps you do just that.
- [**PyTorch Recipes**](https://pytorch.org/tutorials/recipes/recipes_index.html) — PyTorch recipes is a collection of small tutorials to showcase common PyTorch features and workflows you may want to create, such as [Loading Data in PyTorch](https://pytorch.org/tutorials/recipes/recipes/loading_data_recipe.html) and [Saving and Loading models for Inference in PyTorch](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html).
- [**PyTorch Ecosystem**](https://pytorch.org/ecosystem/) - A vast collection of tools that build on top of pure PyTorch to add specialized features for different fields, from [PyTorch3D](https://pytorch3d.org) for 3D computer vision to [Albumentations](https://github.com/albumentations-team/albumentations) for fast data augmentation to [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/) for model evaluation (thank you [for the tip](https://github.com/mrdbourke/pytorch-deep-learning/issues/64#issuecomment-1175164531) Alessandro).
- [**Setting up PyTorch in VSCode**](https://code.visualstudio.com/docs/datascience/pytorch-support) — VSCode is one of the most popular IDEs out there. And its PyTorch support is getting better and better. Throughout the Zero to Mastery PyTorch course, we use Google Colab because of its ease of use. But chances are you’ll be developing in an IDE like VSCode soon.

## 📈 Libraries that make pure PyTorch better/add features

The course focuses on pure PyTorch (using minimal external libraries) because if you know how to write plain PyTorch, you can learn to use the various extension libraries.

- [**fast.ai**](https://github.com/fastai/fastai) — fastai is an open-source library that takes care of many of the boring parts of building neural networks and makes creating state-of-the-art models possible with a few lines of code. Their free library, [course](https://course.fast.ai) and [documentation](https://docs.fast.ai) are all world-class.
- [**MosaicML for more efficient model training**](https://github.com/mosaicml/composer) — The faster you can train models, the faster you can figure out what works and what doesn’t. MosaicML’s open-source `Composer` library helps you train neural networks with PyTorch faster by implementing speedup algorithms behind the scenes which means you can get better results out of your existing PyTorch models faster. All of their code is open-source and their docs are fantastic.
- [**PyTorch Lightning for reducing boilerplate**](https://www.pytorchlightning.ai) — PyTorch Lightning takes care of many of the steps that you often have to do by hand in vanilla PyTorch, such as writing a training and test loop, model checkpointing, logging and more. PyTorch Lightning builds on top of PyTorch to allow you to make PyTorch models with less code.

![Libraries that extend/make pure PyTorch better.](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/extras-001-libraries-to-make-pytorch-better-or-faster.jpeg)

*Libraries that extend/make pure PyTorch better.*

## 📖 Books for PyTorch

- [**Machine Learning with PyTorch and Scikit-Learn: Develop machine learning and deep learning models with Python by Sebastian Raschka**](https://www.amazon.com/Machine-Learning-PyTorch-Scikit-Learn-scikit-learn-ebook-dp-B09NW48MR1/dp/B09NW48MR1/) — A fantastic introduction to machine learning and deep learning. Starting with traditional machine learning algorithms using Scikit-Learn for problems with structured data (tabular or rows and columns or Excel-style) and then switching to how to use PyTorch for deep learning on unstructured data (such as computer vision and natural language processing).
- [**PyTorch Step-by-Step series by Daniel Voigt Godoy**](https://pytorchstepbystep.com) — Where the Zero to Mastery PyTorch course works from a code-first perspective, the Step-by-Step series covers PyTorch and deep learning from a concept-first perspective with code examples to go along. With three editions, Fundamentals, Computer Vision and Sequences (NLP), the step-by-step series is one of my favourite resources for learning PyTorch from the ground up.
- [**Dive into Deep Learning book**](https://d2l.ai) — Possibly one of the most comprehensive resources on the internet for deep learning concepts along with code examples in PyTorch, TensorFlow and Gluon. And all for free! For example, take a look at the author’s explanation of the [Vision Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html) we cover in [08. PyTorch Paper Replicating](https://www.learnpytorch.io/08_pytorch_paper_replicating/).
- **Bonus:** The [fast.ai course](https://course.fast.ai) (available free online) also comes as a freely available online book, [Deep Learning for Coders with fastai & PyTorch](https://course.fast.ai/Resources/book.html).

![Textbooks to learn more about PyTorch as well as deep learning in general.](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/extras-002-books-for-pytorch.jpeg)

*Textbooks to learn more about PyTorch as well as deep learning in general.*

## 🏗 Resources for Machine Learning and Deep Learning Engineering

Machine Learning Engineering (also referred to as MLOps or ML operations) is the practice of getting the models you create into the hands of others. This may mean via a public app or working behind the scenes to make business decisions.

The following resources will help you learn more about the steps around deploying a machine learning model.

- **[Designing Machine Learning Systems book by Chip Huyen](https://www.amazon.com/Designing-Machine-Learning-Systems-Production-Ready/dp/1098107969)** — If you want to build an ML system, it’d be good to know how others have done it. Chip’s book focuses less on building a single machine learning model (though there’s plenty of content on that in the book) but rather building a cohesive ML system. It covers everything from data engineering to model building to model deployment (online and offline) to model monitoring. Even better, it’s a joy to read, you can tell the book is written by a writer (Chip has previously authored several books).
- **[Made With ML by Goku Mohandas](https://madewithml.com)** — Whenever I want to learn or reference something to do with MLOps, I go to [madewithml.com/mlops](https://madewithml.com/#mlops) and see if there’s a lesson on it. Made with ML not only teaches you the  fundamentals of many different ML models but goes through how to build an end-to-end ML system with plenty of code and tooling examples.
- **[The Machine Learning Engineering book by Andriy Burkov](http://www.mlebook.com)** — Even though this book is available to read online for free, I bought it as soon as it came out. I’ve used it as a reference and to learn more about ML engineering so much it’s basically always on my desk/within arms reach. Burkov does an excellent job at getting to the point and referencing further materials when necessary.
- **[Full Stack Deep Learning course](https://fullstackdeeplearning.com)** — I first did this course in 2021. And it’s continued to evolve to cover the latest and greatest tools in the field. It’ll teach you how to plan a project to solve an ML problem, how to source or create data, how to troubleshoot an ML project when it goes wrong and most of all, how to build ML-powered products.

![Resources to improve your machine learning engineering skills (all of the steps that go around building a machine learning model).](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/extras-003-places-to-learn-ml-ops.jpeg)

*Resources to improve your machine learning engineering skills (all of the steps that go around building a machine learning model).*

## 🗃 Where to find datasets

Machine learning projects begin with data. 

No data, no ML. 

The following resources are some of the best for finding open-source and often ready-to-use datasets on a wide range of topics and problem domains.

- [**Paperswithcode Datasets**](https://paperswithcode.com/datasets) — Search for the most used and common machine learning benchmark datasets, understand what they contain, where they came from and where they can be found. You can often also see the current best-performing model on each dataset.
- [**HuggingFace Datasets**](https://huggingface.co/docs/datasets) — Not just a resource to find datasets across a wide range of problem domains but also a library to download and start using them within a few lines of code.
- **[Kaggle Datasets](https://www.kaggle.com/datasets)** — Find all kinds of datasets that usually accompany Kaggle Competitions, many of which come straight out of industry.
- **[Google Dataset search](https://datasetsearch.research.google.com)** — Just like searching Google but specifically for datasets.

These should be plenty to get started, however, for your own specific problems you’ll likely want to build your own dataset.

![Places to find existing and open-source datasets for a variety of problem spaces.](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/extras-004-places-to-find-datasets.jpeg)

*Places to find existing and open-source datasets for a variety of problem spaces.*

## Tools for Deep Learning Domains

The following resources are focused on libraries and pretrained models for specific problem domains such as computer vision and recommendation engines/systems.

### 😎 Computer Vision

We cover computer vision in [03. PyTorch Computer Vision](https://www.learnpytorch.io/03_pytorch_computer_vision/) but as a quick recap, computer vision is the art of getting computers to see. 

If your data is visual, images, x-rays, production line video or even hand-written documents, it may be a computer vision problem.

- **[TorchVision](https://pytorch.org/vision/stable/index.html)** — PyTorch’s resident computer vision library. Find plenty of methods for loading vision data as well as plenty of pretrained computer vision models to use for your own problems.
- [**timm (Torch Image Models) library**](https://github.com/rwightman/pytorch-image-models) — One of the most comprehensive computer vision libraries and resources for pretrained computer vision models. Almost all new research in that uses PyTorch for computer vision leverages the `timm` library in some way.
- **[Yolov5 for object detection](https://github.com/ultralytics/yolov5)** — If you’re looking to build an object detection model in PyTorch, the `yolov5` GitHub repository might be the quickest way to get started.
- **[VISSL (Vision Self-Supervised Learning) library](https://github.com/facebookresearch/vissl)** — Self-supervised learning is the art of getting data to learn patterns in itself. Rather than providing labels for different classes and learning a representation like that, self-supervised learning tries to replicate similar results without labels. VISSL provides an easy to use way to get started using self-supervised learning computer vision models with PyTorch.

### 📚 Natural Language Processing (NLP)

Natural language processing involves finding patterns in text. 

For example, you might want to extract important entities in support tickets or classify a document into different categories.

If your problem involves a large of amount of text, you’ll want to look into the following resources.

- **[TorchText](https://pytorch.org/text/stable/index.html)** — PyTorch’s in-built domain library for text. Like TorchVision, it contains plenty of pre-built methods for loading data and a healthy collection of pretrained models you can adapt to your own problems.
- [**HuggingFace Transformers library**](https://huggingface.co/docs/transformers/index) — The HuggingFace Transformers library has more stars on GitHub than the PyTorch library itself. And there’s a reason. Not that HuggingFace Transformers is better than PyTorch but because it’s the best at what it does: provide data loaders and pretrained state-of-the-art models for NLP (and a whole bunch more).
- **Bonus:** To learn more about how to HuggingFace Transformers library and all of the pieces around it, the HuggingFace team [offer a free online course](https://huggingface.co/course/chapter1/1).

### 🎤 Speech and Audio

If your problem deals with audio files or speech data, such as trying to classify a sound or transcribe speech into text, you’ll want to look into the following resources.

- [**TorchAudio**](https://pytorch.org/audio/stable/index.html) — PyTorch’s domain library for everything audio. Find in-built methods for preparing data and pre-built model architectures for finding patterns in audio data.
- **[SpeechBrain](https://speechbrain.github.io)** — An open-source library built on top of PyTorch to handle speech problems such as recognition (turning speech into text), speech enhancement, speech processing, text-to-speech and more. You can try out many of their [models on the HuggingFace Hub](https://huggingface.co/speechbrain).

### ❓Recommendation Engines

The internet is powered by recommendations. YouTube recommends videos, Netflix recommends movies and TV shows, Amazon recommends products, Medium recommends articles.

If you’re building an online store or online marketplace, chances are you’ll want to start recommending things to your customers.

For that, you’ll want to look into building a recommendation engine. 

- **[TorchRec](https://pytorch.org/torchrec/)** — PyTorch’s newest in-built domain library for powering recommendation engines with deep learning. TorchRec comes with recommendation datasets and models ready to try and use. Though if a custom recommendation egnine isn’t up to par with what you’re after (or too much work), many cloud vendors offer recommendation engine services.

### ⏳ Time Series

If your data has a time component and you’d like to leverage patterns from the past to predict the future, such as, predicting the price of Bitcoin next year (don’t try this, [stock forecasting is BS](https://dev.mrdbourke.com/tensorflow-deep-learning/10_time_series_forecasting_in_tensorflow/#model-10-why-forecasting-is-bs-the-turkey-problem)) or a more reasonable problem of predicting electricity demand for a city next week, you’ll want to look into time series libraries.

Both of these libraries don’t necessarily use PyTorch, however, since time series is such a common problem, I’ve included them here.

- [**Salesforce Merlion**](https://github.com/salesforce/Merlion) — Turn your time series data into intelligence by using Merlion’s data loaders, pre-built models, AutoML (automated machine learning) hyperparameter tuning and more for time series forecasting and time series anomaly detection all inspired by practical use cases.
- [**Facebook Kats**](https://github.com/facebookresearch/Kats) — Facebook’s entire business depends on prediction: when’s the best time to place an advertisement? So you can bet they’re invested heavily in their time series prediction software. Kats (Kit to Analyze Time Series data) is their open-source library for time series forecasting, detection and data processing.

## 👩‍💻 How to get a job

Once you’ve finished an ML course, it’s likely you’ll want to use your ML skills.

And even better, get paid for them.

The following resources are good guides on what to do to get one.

- **["How can a beginner data scientist like me gain experience?"](https://www.mrdbourke.com/how-can-a-beginner-data-scientist-like-me-gain-experience/) by Daniel Bourke** — I get the question of “how do I get experience?” often because many different job requirements state “experience needed”. Well, it turns out one of the best ways to get experience (and a job) is to: *start the job before you have it*.
- **[You Don’t Really Need Another MOOC](https://eugeneyan.com/writing/you-dont-need-another-mooc/) by Eugene Yan** — MOOC stands for massive online open course (or something similar). MOOCs are beautiful. They enable people all over the world at their own pace. However, it can be tempting to just continually do MOOCs over and over again thinking “if I just do one more, I’ll be ready”. The truth is, a few is enough, the returns of a MOOC quickly start to trail off. Instead, go off the trail, start to build, start to create, start to learn skills that can’t be taught. Showcase those skills to get a job.
- **Bonus:** For the most thorough resource on the internet for machine learning interviews, check out Chip Huyen’s free [Introduction to Machine Learning Interviews book](https://huyenchip.com/ml-interviews-book/).