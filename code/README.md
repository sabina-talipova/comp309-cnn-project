# COMP 309: Final Project

## Image Classification Project

This is a simple image classification project using PyTorch. The model is trained to classify images into three categories: cherry, strawberry, and tomato. This project includes scripts for training and testing the model, as well as a saved model file.

### Project Structure

- `train.py`: Script for training the image classification model.
- `test.py`: Script for testing the model on new images.
- `model.pth`: Saved model file containing the trained weights.
- `testdata/`: Folder containing test images, organized into subfolders for each class:
  - `testdata/cherry/`
  - `testdata/strawberry/`
  - `testdata/tomato/`

### Requirements

To run this project, you need to install the following Python packages:

```bash
pip install torch torchvision
```

### Usage
**1. Train the Model (optional):**<br/>
If you would like to retrain the model, you can run train.py. This script will create a new model.pth file with updated weights.

```bash
python train.py
```
**2. Test the Model:**<br/>
To test the model on the images in the testdata folder, simply run:

```bash
python test.py
```

### Folder Structure Example

Ensure that your testdata folder is structured as follows:

```
project-folder/
│
├── train.py
├── test.py
├── model.pth
└── testdata/
    ├── cherry/
    │   ├── image1.jpg
    │   └── ...
    ├── strawberry/
    │   ├── image1.jpg
    │   └── ...
    └── tomato/
        ├── image1.jpg
        └── ...

```

> [!NOTE]
> The model architecture and training configuration are defined in `train.py`. You can modify this file to adjust model parameters, training settings, and other options.
`test.py` loads `model.pth` and uses the images in testdata to evaluate performance. Ensure that your images are organized into subfolders by class name for proper classification.
