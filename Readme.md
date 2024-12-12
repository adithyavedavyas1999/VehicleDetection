# Vehicle Detection Using YOLOv8

## Overview
This project implements a vehicle detection system using YOLOv8. The system supports:
- Data augmentation
- Training a YOLOv8 model
- Evaluating the model
- Real-time video inference

## Requirements
- Python 3.7+
- Libraries: ultralytics, albumentations, tqdm, opencv-python, supervision, matlplotlib, torchvision, torch

Install dependencies:
```pip install ultralytics
pip install torch torchvision
pip install albumentations
pip install opencv-python-headless
pip install tqdm
pip install supervision
pip install matplotlib
```

or you can use a single command 
```pip install ultralytics albumentations opencv-python-headless tqdm supervision matplotlib torch torchvision```

Download the Vehicle Detection Dataset at https://www.kaggle.com/datasets/alkanerturan/vehicledetection/data

To run our prioject make sure you have the dataset downloaded.
To augment our data use the following command.
- ```python augment_data.py```
After augmentation train the model using the following command
- ```python train_model.py```
After training to evaluate the model and save the results use the following command
- ```python test_model.py```
After evaluating to run the model on a video use the following command
- ```python detect_video.py```

Ensure you change the image paths, label paths and video paths when you run
