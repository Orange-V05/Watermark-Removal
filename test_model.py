import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from unet_model import UNet


# Hyperparameters
batch_size = 4
epochs = 25  # Try 25 or more
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Dataset
class WatermarkDataset(Dataset):
    def __init__(self, watermarked_dir, clean_dir):
        self.watermarked_dir = watermarked_dir
        self.clean_dir = clean_dir
        self.filenames = [
            f for f in os.listdir(watermarked_dir)
            if os.path.isfile(os.path.join(watermarked_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        wm = cv2.imread(os.path.join(self.watermarked_dir, filename))
        clean = cv2.imread(os.path.join(self.clean_dir, filename))

        if wm is None or clean is None:
            raise FileNotFoundError(f"Missing or unreadable image: {filename}")

        wm = cv2.cvtColor(wm, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        wm = cv2.resize(wm, (512, 512))
        clean = cv2.resize(clean, (512, 512))

        wm = torch.from_numpy(wm.transpose((2, 0, 1)))
        clean = torch.from_numpy(clean.transpose((2, 0, 1)))

        return wm, clean

    def __len__(self):
        return len(self.filenames)

# Load data
watermarked_dir = "D:/watermark images/Watermarked"
clean_dir = "D:/watermark images/Clean"
train_dataset = WatermarkDataset(watermarked_dir, clean_dir)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model setup
model = UNet(in_channels=3, out_channels=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


#  Train
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_dataloader)}")

print("Training completed.")

# Predict
output_dir = "D:/watermark images/Predicted"
os.makedirs(output_dir, exist_ok=True)
model.eval()

with torch.no_grad():
    for filename in os.listdir(watermarked_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(watermarked_dir, filename)
            image = cv2.imread(input_path)

            if image is None:
                print(f"Skipping {filename}: unreadable.")
                continue

            original_size = (image.shape[1], image.shape[0])  # Width, Height
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            resized = cv2.resize(image_rgb, (512, 512))
            image_tensor = torch.from_numpy(resized.transpose((2, 0, 1))).unsqueeze(0).to(device)

            output = model(image_tensor)
            output = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            output = np.clip(output, 0, 1)
            output = cv2.resize((output * 255).astype(np.uint8), original_size)
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            # Save comparison side by side
            combined = cv2.hconcat([image, output_bgr])
            cv2.imwrite(os.path.join(output_dir, f"cleaned_{filename}"), combined)
            print(f"Saved result: cleaned_{filename}")
