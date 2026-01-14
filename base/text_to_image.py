from diffusers import StableDiffusionPipeline
import torch
import os

# Set Hugging Face cache to F drive
os.environ["HF_HOME"] = "F:\\huggingface_cache"
os.environ["HF_HUB_CACHE"] = "F:\\huggingface_cache\\hub"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Increase timeout for slow connections
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

print("Loading Stable Diffusion model...")
print("This may take several minutes on first run (downloading ~4GB)...")
print("Cache location: F:\\huggingface_cache")

model_id = "runwayml/stable-diffusion-v1-5"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Using device: {device}")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
    cache_dir="F:\\huggingface_cache",
    resume_download=True
).to(device)

print("Model loaded successfully!")

prompt = "A dog who looks in the sky and background there are clouds"

print(f"\nGenerating image with prompt: '{prompt}'...")
print("This may take a few minutes on CPU...")

image = pipe(prompt).images[0]
image.save("output.png")

print(f"\nImage saved to: output.png")

