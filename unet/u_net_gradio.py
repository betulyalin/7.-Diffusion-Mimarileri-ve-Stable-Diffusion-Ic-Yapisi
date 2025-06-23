import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import gradio as gr

# PyTorch'tan hazır segmentasyon modeli (FCN + ResNet50)
model = models.segmentation.fcn_resnet50(pretrained=True).eval()

# Görsel ön işleme
preprocess = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Renk paleti (21 sınıf için Pascal VOC renkleri)
PALETTE = np.array([
    [0, 0, 0],        # background
    [128, 0, 0],      # aeroplane
    [0, 128, 0],      # bicycle
    [128, 128, 0],    # bird
    [0, 0, 128],      # boat
    [128, 0, 128],    # bottle
    [0, 128, 128],    # bus
    [128, 128, 128],  # car
    [64, 0, 0],       # cat
    [192, 0, 0],      # chair
    [64, 128, 0],     # cow
    [192, 128, 0],    # dining table
    [64, 0, 128],     # dog
    [192, 0, 128],    # horse
    [64, 128, 128],   # motorbike
    [192, 128, 128],  # person
    [0, 64, 0],       # potted plant
    [128, 64, 0],     # sheep
    [0, 192, 0],      # sofa
    [128, 192, 0],    # train
    [0, 64, 128]      # tv/monitor
], dtype=np.uint8)

def decode_segmap(segmentation):
    seg_image = PALETTE[segmentation]
    return Image.fromarray(seg_image)

# Segmentasyon fonksiyonu
def segment(image):
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    return decode_segmap(output_predictions)

# Gradio arayüzü
gr.Interface(
    fn=segment,
    inputs=gr.Image(type="pil", label="Görsel Yükle"),
    outputs=gr.Image(type="pil", label="Segmentasyon Sonucu"),
    title="Eğitimli Segmentasyon Demo",
    description="PyTorch FCN-ResNet50 ile hazır segmentasyon modeli. Nesneleri algılar ve renklendirir."
).launch()
