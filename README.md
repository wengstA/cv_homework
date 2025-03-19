# Computer Vision Homework: Object Detection

This repository contains code and resources for a computer vision homework project focused on object detection tasks, with particular emphasis on elephant detection.

## Project Structure

- `data_converter.ipynb`: Jupyter notebook for converting data between COCO and VOC formats
- `epoch_6 1.pth`: Trained model weights file
- `finnal_pro/`: Main project directory containing the mmdetection framework implementation

## Project Overview

This project utilizes the [MMDetection](https://github.com/open-mmlab/mmdetection) framework for object detection tasks. The implementation focuses on detecting common objects across datasets, with special attention to elephant detection. The data preprocessing includes conversion between COCO and VOC annotation formats.

### Features

- Data conversion between COCO and VOC formats
- Implementation using the MMDetection framework
- Pre-trained model weights (epoch 6)
- Support for common object categories:
  - Person, car, bicycle, dog, etc.
  - Special focus on elephant detection

## Requirements

The project relies on the MMDetection framework's requirements, which include:

- Python 3.6+
- PyTorch
- CUDA (for GPU acceleration)
- Other dependencies as specified in the mmdetection requirements.txt file

## Getting Started

1. Clone this repository:
```bash
git clone https://github.com/wengstA/cv_homework.git
cd cv_homework
```

2. Set up the environment:
```bash
# Install dependencies from MMDetection
cd finnal_pro/mmdetection-main
pip install -r requirements/build.txt
pip install -e .
```

3. Run the data preprocessing (if needed):
```bash
jupyter notebook data_converter.ipynb
```

4. Use the trained model for inference or continue training using the provided model weights.

## Model

The model weights are saved in `epoch_6 1.pth`, representing the state of the model after 6 training epochs.

## License

This project follows the license of the MMDetection framework.

## Acknowledgements

- [MMDetection](https://github.com/open-mmlab/mmdetection) for providing the detection framework
- COCO dataset for training data 