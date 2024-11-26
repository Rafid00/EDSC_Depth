# Monocular Depth Estimation for Autonomous Vehicles Using an Encoder-Decoder Model With Skip Connection

This project implements a monocular depth estimation system for autonomous vehicles using an encoder-decoder deep learning model with skip connections. The model is designed to predict depth maps from single RGB images, leveraging the Virtual KITTI dataset for training and evaluation. 

## Features

- **Dataset Handling**: Prepares and processes the Virtual KITTI dataset, including RGB and depth ground-truth images.
- **Encoder-Decoder Model**: A deep neural network architecture optimized for depth estimation with skip connections and refinement layers.
- **Custom Loss Function**: Combines L1 loss, SSIM loss, and edge-aware smoothness loss for better depth predictions.
- **Early Stopping**: Stops training automatically when validation loss stops improving.
- **Evaluation Metrics**: Computes MAE, RMSE, SILog, and threshold accuracies (δ < 1.25, δ < 1.25², δ < 1.25³).

## Visual Results
### Before Training:
![image](https://github.com/user-attachments/assets/2ffacb2e-28b7-45ee-962e-7f8622ab708d)
### After Training:
![Screenshot 2024-11-27 004419](https://github.com/user-attachments/assets/bb3a14ad-10bf-4be6-8bb7-88218c9d137e)
![Screenshot 2024-11-27 004441](https://github.com/user-attachments/assets/48cde5a0-67a1-47cc-bc1a-d751cd800472)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset Preparation](#dataset-preparation)
- [Acknowledgments](#acknowledgments)

## Installation

1. Clone this repository:
    ```bash
    https://github.com/Rafid00/EDSC_Depth.git
    cd EDSC_Depth
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have CUDA installed for GPU acceleration (optional but recommended).

## Dataset Preparation

The model uses the Virtual KITTI dataset. Organize the dataset as follows:
```plaintext
dataset_directory/
    vkitti_1.3.1_rgb/
        Scene01/
            Camera01/
                *.png
    vkitti_1.3.1_depthgt/
        Scene01/
            Camera01/
                *.png
```

### Set the paths to your dataset in the script:
```bash
rgb_dir = 'path/to/vkitti_1.3.1_rgb'
depth_dir = 'path/to/vkitti_1.3.1_depthgt'
```

## Usage

### Run the script
To run the model, run the following command:
```bash
python depth_estimation.py
```

This script will:
- Load and preprocess the dataset.
- Train the encoder-decoder model.
- Save the best-performing model based on validation loss.
- Run the model

### Pre-trained Weight
The pre-trained weight is saved as:
```bash
depth_checkpoint.pth
```

### Visualization
To visualize the training and validation loss:
```bash
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
```

## Acknowledgments
- Virtual KITTI Dataset: Virtual KITTI Dataset
- PyTorch: PyTorch Framework
- PyTorch-MSSSIM: Library used for structural similarity loss.
