# Encoder-Decoder Model With Skip Connection_Depth
Monocular Depth Estimation for Autonomous Vehicles Using An Encoder-Decoder Model With Skip Connection

# Monocular Depth Estimation for Autonomous Vehicles Using an Encoder-Decoder Model With Skip Connection

This project implements a monocular depth estimation system for autonomous vehicles using an encoder-decoder deep learning model with skip connections. The model is designed to predict depth maps from single RGB images, leveraging the Virtual KITTI dataset for training and evaluation.

## Features

- **Dataset Handling**: Prepares and processes the Virtual KITTI dataset, including RGB and depth ground-truth images.
- **Encoder-Decoder Model**: A deep neural network architecture optimized for depth estimation with skip connections and refinement layers.
- **Custom Loss Function**: Combines L1 loss, SSIM loss, and edge-aware smoothness loss for better depth predictions.
- **Early Stopping**: Stops training automatically when validation loss stops improving.
- **Evaluation Metrics**: Computes MAE, RMSE, SILog, and threshold accuracies (δ < 1.25, δ < 1.25², δ < 1.25³).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/username/depth-estimation.git
    cd depth-estimation
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
Set the paths to your dataset in the script:

python
Copy code
rgb_dir = 'path/to/vkitti_1.3.1_rgb'
depth_dir = 'path/to/vkitti_1.3.1_depthgt'
Usage
Training
To train the model, run the following command:

bash
Copy code
python train.py
This script will:

Load and preprocess the dataset.
Train the encoder-decoder model.
Save the best-performing model based on validation loss.
Evaluation
Evaluate the model using the test set:

bash
Copy code
python evaluate.py
This will compute and display metrics such as MAE, RMSE, SILog, and threshold accuracies.

Visualization
To visualize the training and validation loss:

python
Copy code
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
Results
After training, the best model achieves the following performance on the test set:

Mean Absolute Error (MAE): TBD
Root Mean Square Error (RMSE): TBD
Scale-Invariant Logarithmic Error (SILog): TBD
Accuracy δ < 1.25: TBD
Accuracy δ < 1.25²: TBD
Accuracy δ < 1.25³: TBD
(Note: Replace TBD with your actual results.)

Acknowledgments
Virtual KITTI Dataset: Virtual KITTI Dataset
PyTorch: PyTorch Framework
PyTorch-MSSSIM: Library used for structural similarity loss.
License
This project is licensed under the MIT License. See the LICENSE file for details.
