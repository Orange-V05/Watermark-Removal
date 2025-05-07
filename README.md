🧼 Watermark Removal Using U-Net (PyTorch)
This project leverages a deep learning model based on the U-Net architecture (implemented in PyTorch) to automatically detect and remove watermarks from images.

It is designed for easy use—even by users with minimal machine learning experience.

📂 Project Structure
Watermark Removal/
├── unet_model.py         # Defines the U-Net architecture
├── test_model.py         # Trains the model and removes watermarks
├── Watermarked/          # Folder containing watermarked input images
├── Clean/                # Folder with corresponding clean (ground truth) images
├── output/               # Output folder for processed, watermark-free images

🔧 Requirements
Make sure you have Python 3.10 or higher installed.

Install the required Python packages:

pip install torch torchvision opencv-python numpy

🚀 How to Use
Follow these steps to run the watermark removal project:

1. Prepare the Image Dataset
Save your watermarked images in the Watermarked/ folder.
Supported formats: .jpg, .jpeg, .png.

Save the clean versions of the same images (without watermark) in the Clean/ folder.
Important: Filenames must match (e.g., W1.jpg in both folders).

2. Train and Run the Model
Open a terminal in the project directory and run:

python test_model.py
3. Output
After training, the model automatically removes watermarks from all images in the Watermarked/ folder.

Processed images are saved to the output/ directory.

🧠 Model Architecture
The model uses a U-Net, a convolutional neural network originally designed for biomedical image segmentation.

🔍 Key Features:
Learns pixel-wise differences between watermarked and clean images.

Trained using Mean Squared Error (MSE) Loss for accurate reconstruction.

Works well with varying image resolutions and watermark styles.

✅ Example Use Cases
Removing stock photo watermarks

Cleaning up scanned documents

Preparing training datasets for other vision tasks

🤝 Contributions
Feel free to fork, improve, or report issues.
Pull requests are welcome!

📬 Contact
Maintainer: Orange-V05 on GitHub
Built by: Vardaan Kapania, Arun Polson, Husnain Ali
University Project – 2025

