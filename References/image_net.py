import torch
import qrcode
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image

# Set Hugging Face cache to F drive
os.environ["HF_HOME"] = "F:\\huggingface_cache"
os.environ["HF_HUB_CACHE"] = "F:\\huggingface_cache\\hub"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

def generate_qr_ai_art(
    text,
    prompt,
    output_path="qr_ai_art.png"
):
    # 1️⃣ Create QR from text
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_img = qr_img.convert("RGB").resize((512, 512))

    # 2️⃣ Load ControlNet (QR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    cache_dir = "F:\\huggingface_cache"
    
    print(f"Using device: {device}")
    print(f"Cache directory: {cache_dir}")
    
    # Try different ControlNet models
    controlnet_models = [
        "monster-labs/control_v1p_sd15_qrcode_monster",
        "DionTimmer/controlnet_qrcode-control_v11p_sd15",
    ]
    
    controlnet = None
    for model_id in controlnet_models:
        try:
            print(f"Trying ControlNet model: {model_id}")
            controlnet = ControlNetModel.from_pretrained(
                model_id,
                torch_dtype=dtype,
                cache_dir=cache_dir
            )
            print(f"Successfully loaded: {model_id}")
            break
        except Exception as e:
            print(f"Failed to load {model_id}: {e}")
            continue
    
    if controlnet is None:
        raise RuntimeError("Could not load any ControlNet QR code model")

    # 3️⃣ Load Stable Diffusion
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    # Only enable xformers if available and on CUDA
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            print("xformers not available, continuing without it")

    # 4️⃣ Generate image
    image = pipe(
        prompt=prompt,
        negative_prompt="blurry, distorted, unreadable qr, low quality",
        image=qr_img,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.3
    ).images[0]

    image.save(output_path)
    print("Saved:", output_path)


# ▶️ Example usage
generate_qr_ai_art(
    text="https://mywebsite.com",
    prompt="futuristic neon city, cyberpunk style, ultra detailed"
)
