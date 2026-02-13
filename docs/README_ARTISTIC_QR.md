# Artistic QR Code Image Generation Pipeline

A complete Python-based pipeline for generating AI-artistic images with embedded scannable QR codes using Stable Diffusion.

## Features

- 🎨 **AI Image Generation**: Uses Stable Diffusion to generate artistic images
- 📱 **QR Code Integration**: Embeds scannable QR codes seamlessly into images
- 🎯 **Artistic Balance**: Image remains prominent, QR code is subtle but scannable
- ⚙️ **Configurable**: Highly customizable parameters for different use cases
- 🚀 **Complete Pipeline**: End-to-end solution from prompt to final image

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- `diffusers>=0.21.0`
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `accelerate>=0.20.0`
- `qrcode[pil]>=7.4.2`
- `Pillow>=10.0.0`
- `numpy>=1.21.0`

## Usage

### Basic Usage

```python
from artistic_qr_pipeline import ArtisticQRPipeline

# Initialize pipeline
pipeline = ArtisticQRPipeline()

# Generate artistic image with QR code
pipeline.process(
    prompt="A beautiful sunset over mountains, cinematic lighting",
    qr_data="https://example.com",
    output_path="output.png"
)
```

### Command Line Interface

```bash
# Basic usage
python artistic_qr_pipeline.py --prompt "A dog in the sky with clouds" --qr-data "Your QR data here"

# Advanced usage with custom parameters
python artistic_qr_pipeline.py \
    --prompt "Futuristic cityscape at night, neon lights" \
    --qr-data "https://yourwebsite.com" \
    --output "my_artistic_qr.png" \
    --size 1024 \
    --subtlety 0.92 \
    --steps 50 \
    --guidance 7.5 \
    --seed 42
```

### Parameters

#### Pipeline Parameters
- `--prompt`: Text prompt for image generation
- `--qr-data`: Data to encode in QR code (URL, text, etc.)
- `--output`: Output image path (default: `artistic_qr_output.png`)
- `--size`: Image size in pixels (default: 512)
- `--subtlety`: QR code subtlety (0.85-0.95, higher = more subtle, default: 0.92)
- `--steps`: Number of inference steps (default: 50)
- `--guidance`: Guidance scale for prompt adherence (default: 7.5)
- `--seed`: Random seed for reproducibility
- `--model`: Stable Diffusion model ID (default: `runwayml/stable-diffusion-v1-5`)
- `--cache-dir`: Model cache directory

## How It Works

1. **QR Code Generation**: Creates a high-error-correction QR code from your data
2. **Image Generation**: Uses Stable Diffusion to generate an artistic image from your prompt
3. **Artistic Embedding**: Intelligently embeds the QR code pattern into the image:
   - Image remains the main visual focus
   - QR code is subtly integrated in the background
   - Maintains scannability through careful contrast adjustments

## Technical Details

### QR Code Embedding Algorithm

The pipeline uses a sophisticated masking technique:

1. **Mask Creation**: Converts QR code to a grayscale mask
2. **Subtle Modulation**: 
   - Darkens image areas where QR code is black (data modules)
   - Lightens image areas where QR code is white (background)
3. **Contrast Boost**: Adds minimal contrast adjustment to ensure scannability
4. **Balance**: Maintains image beauty while preserving QR code structure

### Subtlety Parameter

- **0.85-0.88**: More visible QR code, easier to scan
- **0.90-0.92**: Balanced (recommended) - image prominent, QR scannable
- **0.93-0.95**: Very subtle QR code, image is main focus

## Examples

### Example 1: Product Promotion
```python
pipeline.process(
    prompt="Modern product photography, clean background, professional lighting",
    qr_data="https://myshop.com/product123",
    output_path="product_qr.png",
    subtlety=0.90
)
```

### Example 2: Art Portfolio
```python
pipeline.process(
    prompt="Abstract art, vibrant colors, digital painting style",
    qr_data="https://myportfolio.com",
    output_path="portfolio_qr.png",
    subtlety=0.93  # Very subtle for artistic focus
)
```

### Example 3: Event Poster
```python
pipeline.process(
    prompt="Concert poster design, neon lights, urban style",
    qr_data="https://eventbrite.com/my-event",
    output_path="event_poster.png",
    subtlety=0.88  # More visible for easy scanning
)
```

## Performance

- **GPU (CUDA)**: ~10-30 seconds per image
- **CPU**: ~5-10 minutes per image (depending on hardware)

## Troubleshooting

### Memory Issues
If you encounter `MemoryError`:
- Use a smaller image size (`--size 512` or `--size 256`)
- Close other applications to free RAM
- Consider using CPU offloading (see code for implementation)

### Model Download
On first run, the model (~4GB) will be downloaded to the cache directory.
Ensure you have:
- Sufficient disk space (at least 10GB free)
- Stable internet connection
- Proper cache directory permissions

## Advanced Usage

### Custom Model
```python
pipeline = ArtisticQRPipeline(
    model_id="stabilityai/stable-diffusion-2-1",
    cache_dir="/path/to/cache"
)
```

### Programmatic Control
```python
# Step-by-step control
pipeline = ArtisticQRPipeline()
pipeline.load_model()

qr_code = pipeline.create_qr_code("Your data", size=512)
image = pipeline.generate_image("Your prompt", seed=42)
result = pipeline.embed_qr_artistically(image, qr_code, subtlety=0.92)
result.save("output.png")
```

## License

This project uses:
- Stable Diffusion models (check respective licenses)
- QR Code library (BSD License)
- Diffusers library (Apache 2.0)

## Contributing

Feel free to submit issues, fork the repository, and create pull requests.

## Author

Generated for Text-to-Image project




