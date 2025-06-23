import torch
import gradio as gr
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerDiscreteScheduler
from controlnet_aux import CannyDetector
from PIL import Image

# Stabil ve optimize scheduler
scheduler = EulerDiscreteScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler"
)

# Model yükleme ve ayarları (float16, GPU optimizasyonu)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16,
).to('cuda')

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    scheduler=scheduler,
    safety_checker=None,
    torch_dtype=torch.float16,
).to('cuda')

# xformers satırı kaldırıldı (hatasız kullanım için)
# pipe.enable_xformers_memory_efficient_attention()

canny = CannyDetector()

def controlnet_infer(input_image, prompt, guidance_scale, steps, low_threshold, high_threshold):
    input_image = input_image.resize((512, 512))

    # Daha net kenar belirleme için canny parametreleri ayarlanabilir
    canny_image = canny(input_image, low_threshold, high_threshold)

    # ControlNet ile temiz ve kaliteli görüntü oluştur
    output = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        image=canny_image,
    ).images[0]

    return output

# Gradio Arayüzü
demo = gr.Interface(
    fn=controlnet_infer,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(label="Prompt", value="a beautiful woman, realistic lighting, professional photography"),
        gr.Slider(minimum=1, maximum=15, value=7.5, label="Guidance Scale"),
        gr.Slider(minimum=10, maximum=100, value=30, step=1, label="Inference Steps"),
        gr.Slider(minimum=50, maximum=200, value=100, label="Canny Low Threshold"),
        gr.Slider(minimum=150, maximum=300, value=200, label="Canny High Threshold"),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Gelişmiş ve Stabil ControlNet Demo",
    description="Gerçekçi ve kaliteli çıktılar için optimize edilmiş ControlNet + Stable Diffusion Demo uygulaması"
)

demo.launch()
