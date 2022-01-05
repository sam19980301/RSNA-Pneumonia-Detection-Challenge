# RSNA Pneumonia Detection Challenge

This repository is the implementation of [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge). The project compares and analyzes the performance of object detection models such as Faster/Cascade RCNN, Retinanet and Swin Transformer on RSNA dataset. Also, some ensemble and post processing methods are used in this project.

## Installation

In this project, framework [MMDetection](https://github.com/open-mmlab/mmdetection) is used. The package should be installed first to implement further analysis. Refer to [MMDetection official installation](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) for more information.

```
# Create a conda virtual environment and activate it.
conda create -n openmmlab python=3.7 -y
conda activate openmmlab

# Install PyTorch and torchvision following the official instructions
conda install pytorch torchvision -c pytorch

# Install MMDetection
pip install openmim
mim install mmdet
```

## Download & Preprocess Dataset 
The dataset used in this project could be downloaded using the folloing kaggle command or via this [link](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data). To transform labels into [Coco dataset style](https://cocodataset.org/#format-data), run the following command:

```
kaggle competitions download -c rsna-pneumonia-detection-challenge
cd utils
python coco_formatter.py
```

## Training Models

Import the customized model configurations into MMDetection `configs` directory first and then run the training script.

```
mv <model name>/<configs.py> <path of mmdetection>/config/<model name>

cd mmdetection
python tools/train.py configs/<model name>/<configs.py>
```

The weights of the models above is saved in Drive: [Trained Models](https://drive.google.com/drive/folders/1dyMVyAZx8heH-09dhjaxZ4hnYk1gF6Cz?usp=sharing)

## Inference

To infer the testing images using trained model, run this to generate bbox file in `.json` format

```
cd mmdetection
./tools/dist_test.sh \
    /work_dirs/<model name>/<model configs.py> \
    /work_dirs/<mode name>/<model weights.pth> \
    1 \
    --format-only \
    --options "jsonfile_prefix=./<inference filename>"
```

To post-process the result and submit the answer to kaggle, run `submission.py`, `ensemble.py` and `resize_submission.py` in order.

## Results
TBD
