import cv2
import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import os
import matplotlib.pyplot as plt

steps = 10
image_url = "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?w=800"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

try:
    response = requests.get(image_url, headers=headers, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
except (requests.exceptions.RequestException, UnidentifiedImageError) as e:
    print("Görsel indirilemedi veya tanınamadı:", e)
    exit()

img = img.resize((256, 256))
img_np = np.array(img).astype(np.float32) / 255.0

os.makedirs("manual_diffusion", exist_ok=True)

noisy_images = []
for i in range(steps):
    noise_level = (i + 1) / steps
    noise = np.random.normal(0, noise_level, img_np.shape)
    noisy_img = np.clip(img_np + noise, 0, 1)
    noisy_images.append(noisy_img)
    plt.imsave(f"manual_diffusion/step_{i+1:02d}_noisy.png", noisy_img)

for i, noisy_img in enumerate(reversed(noisy_images)):
    denoised = cv2.GaussianBlur((noisy_img * 255).astype(np.uint8), (3, 3), sigmaX=1.5)
    Image.fromarray(denoised).save(f"manual_diffusion/step_{steps+i+1:02d}_denoised.png")

print("Görsel başarıyla işlendi. Klasöre bak: manual_diffusion/")
