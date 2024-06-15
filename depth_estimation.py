import torch
from torch import nn
import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import Resize
import numpy as np # Import NumPy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from pytorch_msssim import ssim
import torch.optim as optim
from tqdm import tqdm


# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

class VirtualKITTI(Dataset):
    def __init__(self, rgb_dir, depth_dir, img_size=(256, 256)):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.img_size = img_size

        scene_folders = [f for f in os.listdir(rgb_dir) if os.path.isdir(os.path.join(rgb_dir, f))]
        self.rgb_paths = []
        self.depth_paths = []
        
        for scene in scene_folders:
            camera_folders = [f for f in os.listdir(os.path.join(rgb_dir, scene)) if os.path.isdir(os.path.join(rgb_dir, scene, f))]
            for cam in camera_folders:
                image_files = [f for f in os.listdir(os.path.join(rgb_dir, scene, cam)) if f.endswith('.png')]
                self.rgb_paths.extend([os.path.join(rgb_dir, scene, cam, img_file) for img_file in image_files])
                self.depth_paths.extend([os.path.join(depth_dir, scene, cam, img_file) for img_file in image_files])

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        depth_path = self.depth_paths[idx]

        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise FileNotFoundError(f"Could not load RGB image: {rgb_path}")

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Could not load depth image: {depth_path}")

        depth = depth.astype(np.float32) / 255.0

        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float() / 255.0

        resize = Resize(self.img_size)
        rgb_tensor = resize(rgb_tensor)
        depth_tensor = resize(depth_tensor)

        return rgb_tensor, depth_tensor

rgb_dir = 'C:/Users/rafid/OneDrive/Desktop/Datasets/vkitti_1.3.1_rgb'
depth_dir = 'C:/Users/rafid/OneDrive/Desktop/Datasets/vkitti_1.3.1_depthgt'

dataset = VirtualKITTI(rgb_dir, depth_dir)

# Split into Train, Validation, and Test
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Ensure deterministic split for reproducibility
torch.manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Second Model

class ImprovedDepthEstimationModel(nn.Module):
    def __init__(self):
        super(ImprovedDepthEstimationModel, self).__init__()

        # Encoder (Deeper with 5 levels)
        self.enc_conv1 = self._make_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.enc_conv2 = self._make_block(64, 128, kernel_size=5, stride=2, padding=2)
        self.enc_conv3 = self._make_block(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc_conv4 = self._make_block(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc_conv5 = self._make_block(512, 512, kernel_size=3, stride=2, padding=1)

        # Decoder (No Bottlenecks, More Refinement Layers, Activation Functions)
        self.upconv5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec_conv5 = self._make_block(1024, 512, kernel_size=3, padding=1)
        self.dec_conv5a = self._make_block(512, 256, kernel_size=3, padding=1) # Extra refinement

        self.upconv4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec_conv4 = self._make_block(512, 256, kernel_size=3, padding=1)
        self.dec_conv4a = self._make_block(256, 128, kernel_size=3, padding=1) # Extra refinement

        self.upconv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec_conv3 = self._make_block(256, 128, kernel_size=3, padding=1)
        self.dec_conv3a = self._make_block(128, 64, kernel_size=3, padding=1) # Extra refinement

        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec_conv2 = self._make_block(128, 64, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(32, 1, kernel_size=3, padding=1)  # Output layer


    def _make_block(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(enc1)
        enc3 = self.enc_conv3(enc2)
        enc4 = self.enc_conv4(enc3)
        enc5 = self.enc_conv5(enc4)

        # Decoder with skip connections, no bottlenecks, and more refinement
        dec5 = self.upconv5(enc5)
        dec5 = torch.cat([dec5, enc4], dim=1)
        dec5 = self.dec_conv5(dec5)
        dec5 = self.dec_conv5a(dec5)

        dec4 = self.upconv4(dec5)  # Pass dec5 to upconv4
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec4 = self.dec_conv4(dec4)
        dec4 = self.dec_conv4a(dec4)

        dec3 = self.upconv3(dec4)  # Pass dec4 to upconv3
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.dec_conv3(dec3)
        dec3 = self.dec_conv3a(dec3)

        dec2 = self.upconv2(dec3)  # Pass dec3 to upconv2
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.dec_conv2(dec2)

        dec1 = self.upconv1(dec2)  # Pass dec2 to upconv1
        out = self.dec_conv1(dec1)  # Depth map output
        return out


# Create an instance of the model
# model = DepthEstimationModel()
model = ImprovedDepthEstimationModel()

# (Optional) Move the model to GPU if available
model.to(device)
# model0.to(device)

def depth_loss_function(y_true, y_pred, alpha=0.85, beta=0.15):
    """
    Combined loss function for monocular depth estimation,
    tailored for a simple encoder-decoder model with skip connections.

    Args:
        y_true: Ground truth depth map.
        y_pred: Predicted depth map.
        alpha: Weight for the L1 loss term.
        beta: Weight for the SSIM and smoothness loss terms.
    """

    l1_criterion = nn.L1Loss()
    l_depth = l1_criterion(y_pred, y_true)

    l_ssim = torch.clamp((1 - ssim(y_true, y_pred, data_range=10.0)) * 0.5, 0, 1)

    # Smoothness Loss (simple gradient-based) with specified dimensions
    dy_pred, dx_pred = torch.gradient(y_pred, dim=(2, 3)) # compute gradients along height and width dimensions
    l_edges = torch.mean(torch.abs(dy_pred) + torch.abs(dx_pred))

    loss = alpha * l_depth + beta * (l_ssim + l_edges)

    return loss

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

# New Training Loop with Early Stopping

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, save_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.Inf
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)
        print(f"Model checkpoint saved at {self.save_path}")

# Assuming your model, train_loader, val_loader, depth_loss_function, and device are already defined

# Create loss function instance
criterion = depth_loss_function

# Create optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

# Training Loop
num_epochs = 100  # Set to a high number and use early stopping to halt training when needed

# Early stopping instance
early_stopping = EarlyStopping(patience=5, min_delta=0.001, save_path='best_model.pth')

# Lists to store losses
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0

    # Wrap the data loader with tqdm for a progress bar
    train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

    for images, depths in train_loop:
        images = images.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(depths, outputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Update the progress bar with the current loss
        train_loop.set_postfix(loss=loss.item())

    # Calculate average training loss for the epoch
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Validation
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for images, depths in val_loader:
            images = images.to(device)
            depths = depths.to(device)
            outputs = model(images)
            loss = criterion(depths, outputs)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")

    # Check early stopping criteria
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Plotting the loss curves
plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Load the best model after training
model.load_state_dict(torch.load('best_model.pth'))
print("Best model loaded for further evaluation or inference.")

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

def evaluate_depth_model(model, data_loader, device='cpu'):
    model.eval()  # Set the model to evaluation mode

    mae_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()

    total_mae = 0.0
    total_rmse = 0.0
    total_silog = 0.0
    total_delta1 = 0
    total_delta2 = 0
    total_delta3 = 0
    num_samples = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # Get predicted depth maps

            # Calculate metrics
            total_mae += mae_criterion(output, target).item()
            total_rmse += torch.sqrt(mse_criterion(output, target)).item()

            # Clamp values to avoid log(0) or negative values
            output = torch.clamp(output, min=1e-6)
            target = torch.clamp(target, min=1e-6)

            log_diff = torch.log(output) - torch.log(target)
            silog = torch.sqrt(torch.mean(log_diff ** 2) - torch.mean(log_diff) ** 2)

            if not torch.isnan(silog).item():
                total_silog += silog.item()

            # Threshold accuracy (delta < 1.25, delta < 1.25^2, delta < 1.25^3)
            rel = torch.max((target / output), (output / target))
            total_delta1 += (rel < 1.25).float().mean().item()
            total_delta2 += (rel < 1.25 ** 2).float().mean().item()
            total_delta3 += (rel < 1.25 ** 3).float().mean().item()
            num_samples += 1

    # Average the metrics over all samples
    avg_mae = total_mae / num_samples
    avg_rmse = total_rmse / num_samples
    avg_silog = total_silog / num_samples if num_samples > 0 else float('nan')
    avg_delta1 = total_delta1 / num_samples
    avg_delta2 = total_delta2 / num_samples
    avg_delta3 = total_delta3 / num_samples

    return avg_mae, avg_rmse, avg_silog, avg_delta1, avg_delta2, avg_delta3

# Example Usage:
test_mae, test_rmse, test_silog, avg_delta1, avg_delta2, avg_delta3 = evaluate_depth_model(model, test_loader, device)

print(f"Test MAE: {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test SILog: {test_silog:.4f}")
print(f"Test Delta1: {avg_delta1:.4f}")
print(f"Test Delta2: {avg_delta2:.4f}")
print(f"Test Delta3: {avg_delta3:.4f}")

# (Optional) Save the model checkpoint if needed
torch.save(model.state_dict(), 'depth_checkpoint.pth')