import os
import cv2
import random
import numpy as np

# Config
clean_dir = r"D:/watermark images/clean"
watermarked_dir = r"D:/watermark images/watermarked"
os.makedirs(watermarked_dir, exist_ok=True)

# List of sample watermark texts
watermark_texts = ["Sample", "Protected", "Do Not Copy", "Demo", "Watermark"]

# Generate
def apply_watermark(image, text="Sample"):
    h, w = image.shape[:2]
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = random.uniform(1.0, 2.5)
    thickness = random.randint(2, 5)

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = random.randint(0, max(0, w - text_size[0]))
    y = random.randint(text_size[1], h - 10)

    # White text with black shadow
    cv2.putText(overlay, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(overlay, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Blend with some transparency
    alpha = 0.3 + random.uniform(0, 0.3)
    watermarked = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return watermarked

# Process
for filename in os.listdir(clean_dir):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(clean_dir, filename)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to read the {img_path}")
        continue

    text = random.choice(watermark_texts)
    watermarked = apply_watermark(image, text)
    save_path = os.path.join(watermarked_dir, filename)
    cv2.imwrite(save_path, watermarked)

    print(f"Generated: {save_path}")
