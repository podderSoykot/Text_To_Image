"""
Artistic QR Code Image Generation Pipeline
==========================================
A complete pipeline for generating AI-artistic images with embedded scannable QR codes
using Stable Diffusion and advanced image processing techniques.

Author: Generated for Text-to-Image project
"""

import os
import qrcode
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from diffusers import StableDiffusionPipeline
from typing import Optional, Tuple
import argparse


class ArtisticQRPipeline:
    """
    Complete pipeline for generating artistic images with embedded QR codes.
    """
    
    def __init__(self, 
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 cache_dir: str = "F:\\huggingface_cache",
                 device: Optional[str] = None):
        """
        Initialize the Artistic QR Pipeline.
        
        Args:
            model_id: Hugging Face model ID for Stable Diffusion
            cache_dir: Directory to cache models
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        
        # Set up environment variables
        self._setup_environment()
        
    def _setup_environment(self):
        """Configure environment variables for Hugging Face."""
        os.environ["HF_HOME"] = self.cache_dir
        os.environ["HF_HUB_CACHE"] = os.path.join(self.cache_dir, "hub")
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
    
    def load_model(self):
        """Load the Stable Diffusion model."""
        print(f"Loading Stable Diffusion model: {self.model_id}")
        print(f"Device: {self.device}")
        print(f"Cache location: {self.cache_dir}")
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            cache_dir=self.cache_dir
        ).to(self.device)
        
        print("Model loaded successfully!")
        return self.pipe
    
    def create_qr_code(self, 
                      data: str, 
                      size: int = 512, 
                      border: int = 4,
                      error_correction: str = "H") -> Image.Image:
        """
        Create a QR code with specified parameters.
        
        Args:
            data: Data to encode in QR code
            size: Output size in pixels
            border: Border size (modules)
            error_correction: Error correction level ('L', 'M', 'Q', 'H')
        
        Returns:
            PIL Image of the QR code
        """
        error_levels = {
            'L': qrcode.constants.ERROR_CORRECT_L,
            'M': qrcode.constants.ERROR_CORRECT_M,
            'Q': qrcode.constants.ERROR_CORRECT_Q,
            'H': qrcode.constants.ERROR_CORRECT_H
        }
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=error_levels.get(error_correction.upper(), qrcode.constants.ERROR_CORRECT_H),
            box_size=10,
            border=border,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        # Create QR code image with high contrast
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img = qr_img.resize((size, size), Image.Resampling.LANCZOS)
        
        return qr_img
    
    def generate_image(self, 
                      prompt: str, 
                      num_inference_steps: int = 50,
                      guidance_scale: float = 7.5,
                      seed: Optional[int] = None) -> Image.Image:
        """
        Generate an artistic image using Stable Diffusion.
        
        Args:
            prompt: Text prompt for image generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for prompt adherence
            seed: Random seed for reproducibility
        
        Returns:
            Generated PIL Image
        """
        if self.pipe is None:
            self.load_model()
        
        print(f"\nGenerating image with prompt: '{prompt}'")
        print(f"Inference steps: {num_inference_steps}, Guidance scale: {guidance_scale}")
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image = self.pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        print("Image generated successfully!")
        return image
    
    def embed_qr_artistically(self,
                              base_image: Image.Image,
                              qr_image: Image.Image,
                              subtlety: float = 0.92,
                              contrast_boost: float = 0.05) -> Image.Image:
        """
        Embed QR code into image artistically.
        Image remains prominent, QR code is subtle but scannable.
        
        Args:
            base_image: The generated artistic image
            qr_image: The QR code image
            subtlety: How subtle the QR code is (0.85-0.95, higher = more subtle)
            contrast_boost: Contrast boost for scannability (0.0-0.1)
        
        Returns:
            PIL Image with artistically embedded QR code
        """
        # Convert to RGB if needed
        if base_image.mode != 'RGB':
            base_image = base_image.convert('RGB')
        if qr_image.mode != 'RGB':
            qr_image = qr_image.convert('RGB')
        
        # Resize both to same size
        target_size = max(base_image.size)
        base_image = base_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        qr_image = qr_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy arrays
        base_array = np.array(base_image)
        qr_array = np.array(qr_image)
        
        # Convert QR code to grayscale mask
        qr_gray = np.dot(qr_array[...,:3], [0.2989, 0.5870, 0.1140])
        qr_mask = qr_gray / 255.0  # Normalize to 0-1
        
        # Calculate factors for subtle embedding
        dark_factor = subtlety  # Darkening for QR data areas
        light_factor = 2.0 - subtlety  # Lightening for QR background areas
        
        # Apply artistic embedding
        result_array = base_array.copy().astype(float)
        qr_mask_3d = np.expand_dims(qr_mask, axis=2)
        
        # Darken areas where QR code is black (data modules)
        dark_mask = (1 - qr_mask_3d)
        result_array = result_array * (1 - dark_mask * (1 - dark_factor))
        
        # Lighten areas where QR code is white (background)
        light_mask = qr_mask_3d
        result_array = result_array * (1 + light_mask * (light_factor - 1))
        
        # Add contrast boost for scannability
        if contrast_boost > 0:
            qr_contrast = (qr_mask_3d - 0.5) * 2  # -1 to 1 range
            result_array = result_array + (qr_contrast * contrast_boost * 30)
        
        # Ensure values are in valid range
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        result_image = Image.fromarray(result_array)
        
        return result_image
    
    def process(self,
                prompt: str,
                qr_data: str,
                output_path: str = "artistic_qr_output.png",
                image_size: int = 512,
                subtlety: float = 0.92,
                num_inference_steps: int = 50,
                guidance_scale: float = 7.5,
                seed: Optional[int] = None) -> Image.Image:
        """
        Complete pipeline: Generate image and embed QR code.
        
        Args:
            prompt: Text prompt for image generation
            qr_data: Data to encode in QR code
            output_path: Path to save output image
            image_size: Size of output image
            subtlety: QR code subtlety (0.85-0.95)
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            seed: Random seed
        
        Returns:
            Final artistic QR code image
        """
        print("=" * 60)
        print("Artistic QR Code Image Generation Pipeline")
        print("=" * 60)
        
        # Step 1: Create QR code
        print("\n[Step 1/3] Creating QR code...")
        qr_image = self.create_qr_code(qr_data, size=image_size)
        print(f"QR code created: {image_size}x{image_size}")
        
        # Step 2: Generate artistic image
        print("\n[Step 2/3] Generating artistic image...")
        generated_image = self.generate_image(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        # Resize generated image to match QR code size
        generated_image = generated_image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        
        # Step 3: Embed QR code artistically
        print("\n[Step 3/3] Embedding QR code artistically...")
        final_image = self.embed_qr_artistically(
            generated_image,
            qr_image,
            subtlety=subtlety
        )
        
        # Save result
        final_image.save(output_path)
        print(f"\n✓ Artistic QR code image saved to: {output_path}")
        print(f"  Image size: {image_size}x{image_size}")
        print(f"  QR data: {qr_data}")
        print(f"  Prompt: {prompt}")
        print("\n" + "=" * 60)
        
        return final_image


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate artistic images with embedded scannable QR codes"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A dog who looks in the sky and background there are clouds",
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--qr-data",
        type=str,
        default="Soykot Podder(Senior Ai Engineer)(Indian Institute of Information Technology)(IIIT)",
        help="Data to encode in QR code"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artistic_qr_output.png",
        help="Output image path"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Image size (default: 512)"
    )
    parser.add_argument(
        "--subtlety",
        type=float,
        default=0.92,
        help="QR code subtlety (0.85-0.95, higher = more subtle, default: 0.92)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale (default: 7.5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion model ID"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="F:\\huggingface_cache",
        help="Model cache directory"
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ArtisticQRPipeline(
        model_id=args.model,
        cache_dir=args.cache_dir
    )
    
    # Process
    pipeline.process(
        prompt=args.prompt,
        qr_data=args.qr_data,
        output_path=args.output,
        image_size=args.size,
        subtlety=args.subtlety,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

