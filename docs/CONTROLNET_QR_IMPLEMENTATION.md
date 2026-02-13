# ControlNet QR Code Implementation Guide

Based on industry standards and video tutorials (like https://www.youtube.com/watch?v=BPJzfsc_2Qo), here's how to implement ControlNet-based QR code generation.

## Overview

ControlNet is the industry-standard approach for creating artistic QR codes. It uses the QR code pattern as a control signal during image generation, resulting in more seamless integration.

## Implementation

### Step 1: Install Dependencies

```bash
pip install diffusers transformers accelerate controlnet-aux
```

### Step 2: ControlNet QR Code Pipeline

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch
from PIL import Image
import qrcode

# Load ControlNet for QR codes
controlnet = ControlNetModel.from_pretrained(
    "monster-labs/control_v1p_sd15_qrcode_monster",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Load base Stable Diffusion model
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
).to("cuda" if torch.cuda.is_available() else "cpu")

# Generate QR code
def create_qr_code(data: str, size: int = 512) -> Image.Image:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    return qr_img.resize((size, size), Image.Resampling.LANCZOS)

# Generate artistic QR code
def generate_artistic_qr(
    prompt: str,
    qr_data: str,
    controlnet_conditioning_scale: float = 1.5,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
) -> Image.Image:
    # Create QR code
    qr_image = create_qr_code(qr_data, size=512)
    
    # Generate with ControlNet
    image = pipe(
        prompt=prompt,
        image=qr_image,  # QR code as control
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale
    ).images[0]
    
    return image
```

## Key Parameters

### `controlnet_conditioning_scale`
- **Range**: 0.5 - 2.0
- **Default**: 1.5
- **Effect**: 
  - Lower (0.5-1.0): More artistic freedom, QR code less strict
  - Higher (1.5-2.0): QR code structure more strictly followed
  - **Recommended**: 1.3-1.7 for balance

### Prompt Engineering Tips

1. **Anime Style** (like in the video):
   ```
   "anime style, beautiful character, detailed, high quality, 
   vibrant colors, fantasy setting"
   ```

2. **Realistic Style**:
   ```
   "photorealistic, professional photography, high detail, 
   cinematic lighting, 4k"
   ```

3. **Abstract/Artistic**:
   ```
   "abstract art, digital painting, vibrant colors, 
   modern design, artistic"
   ```

## Advantages of ControlNet Approach

1. ✅ **Seamless Integration**: QR code is part of generation process
2. ✅ **Better Artistic Results**: More natural blending
3. ✅ **Industry Standard**: Used by most professional platforms
4. ✅ **Anime Style Support**: Works great for anime-style QR codes
5. ✅ **Maintains Scannability**: ControlNet preserves QR structure

## Comparison with Post-Processing

| Feature | ControlNet | Post-Processing (Current) |
|---------|-----------|---------------------------|
| Integration | During generation | After generation |
| Seamlessness | High | Medium |
| Setup Complexity | Higher | Lower |
| Model Requirements | ControlNet model needed | Any SD model |
| Anime Style | Excellent | Good |
| Scannability | High | Depends on subtlety |

## Implementation in Our Pipeline

To add ControlNet support to `artistic_qr_pipeline.py`:

1. Add ControlNet as optional method
2. Keep post-processing as default (simpler)
3. Allow users to choose method
4. ControlNet for advanced users who want best results

## Example Usage

```python
from controlnet_qr import generate_artistic_qr

# Generate anime-style QR code
image = generate_artistic_qr(
    prompt="anime style, beautiful character, fantasy setting, detailed",
    qr_data="https://example.com",
    controlnet_conditioning_scale=1.5,
    num_inference_steps=50
)

image.save("anime_qr_code.png")
```

## References

- Video Tutorial: https://www.youtube.com/watch?v=BPJzfsc_2Qo
- ControlNet Model: `monster-labs/control_v1p_sd15_qrcode_monster`
- Base Model: `runwayml/stable-diffusion-v1-5`




