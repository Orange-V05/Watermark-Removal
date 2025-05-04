# Watermark Removal with U-Net (PyTorch)

This project uses a deep learning model (U-Net architecture) built with PyTorch to automatically remove watermarks from images. The model is trained on pairs of watermarked and clean images and can process all images in a directory.

## 📁 Folder Structure

- `unet_model.py`: Contains the U-Net model definition.
- `test_model.py`: Runs training and processes images for watermark removal.
- `Watermarked/`: Folder containing images with watermarks.
- `Clean/`: Folder containing clean ground-truth images (for training).
- `output/`: Folder where processed (watermark-free) images are saved.

## ⚙️ Requirements

- Python 3.10 or above
- PyTorch
- OpenCV
- NumPy
- torchvision

Install them via:

```bash
pip install torch torchvision opencv-python numpy
```

## 🚀 How to Run

1. Place your watermarked images (JPEG, JPG, PNG, etc.) inside the `Watermarked/` folder.
2. Place the corresponding clean images in the `Clean/` folder for training (use same filenames).
3. Run the model:

```bash
python test_model.py
```

4. After training, the model will automatically process all images from the `Watermarked/` folder and save the output (with watermark removed) in the `output/` directory.

## 🧠 Model

The U-Net model is trained using MSE loss and is designed to learn a pixel-wise mapping from watermarked images to their clean counterparts.

---

Feel free to clone, use, or contribute to this project. Feedback is welcome!
