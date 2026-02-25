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
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
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
        self.controlnet_pipe = None
        self.controlnet_model = None
        
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
    
    def load_controlnet_model(self):
        """Load ControlNet model for QR code generation."""
        if self.controlnet_pipe is not None:
            return self.controlnet_pipe
        
        print("Loading ControlNet QR code model...")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
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
                    cache_dir=self.cache_dir
                )
                print(f"Successfully loaded: {model_id}")
                self.controlnet_model = model_id
                break
            except Exception as e:
                print(f"Failed to load {model_id}: {e}")
                continue
        
        if controlnet is None:
            raise RuntimeError("Could not load any ControlNet QR code model")
        
        # Load ControlNet pipeline
        self.controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        
        if self.device == "cuda":
            try:
                self.controlnet_pipe.enable_xformers_memory_efficient_attention()
            except:
                print("xformers not available, continuing without it")
        
        print("ControlNet model loaded successfully!")
        return self.controlnet_pipe
    
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
                              contrast_boost: float = 0.08) -> Image.Image:
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
        base_array = np.array(base_image, dtype=float)
        qr_array = np.array(qr_image, dtype=float)
        
        # Convert QR code to binary mask (preserve structure exactly)
        qr_gray = np.dot(qr_array[...,:3], [0.2989, 0.5870, 0.1140])
        # Use Otsu's threshold for better binary conversion
        # This ensures the QR code pattern matches the original exactly
        threshold = 127
        # Try to use a more robust threshold
        if qr_gray.max() > qr_gray.min():
            # Use adaptive threshold based on image statistics
            threshold = (qr_gray.max() + qr_gray.min()) / 2
        qr_binary = (qr_gray < threshold).astype(float)  # 1 for black (data modules), 0 for white (background)
        qr_mask_3d = np.expand_dims(qr_binary, axis=2)
        
        # Calculate factors for subtle embedding
        # Adjust factors to ensure minimum contrast for scannability
        # QR scanners need at least 20-25% contrast to reliably decode
        # Map subtlety to effective contrast: lower subtlety = more contrast
        min_contrast = 0.20  # Minimum 20% contrast needed for reliable scanning
        
        # Calculate contrast based on subtlety parameter
        # More aggressive contrast for better scannability
        # subtlety 0.85 -> contrast 0.25 (25%)
        # subtlety 0.88 -> contrast 0.22 (22%)
        # subtlety 0.90 -> contrast 0.20 (20% minimum)
        # subtlety 0.92 -> contrast 0.20 (20% minimum)
        # subtlety 0.95 -> contrast 0.20 (20% minimum)
        if subtlety <= 0.85:
            effective_contrast = 0.25
        elif subtlety <= 0.88:
            # Linear interpolation between 0.85 and 0.88
            effective_contrast = 0.25 - (subtlety - 0.85) * (0.25 - 0.22) / (0.88 - 0.85)
        elif subtlety <= 0.90:
            # Linear interpolation between 0.88 and 0.90
            effective_contrast = 0.22 - (subtlety - 0.88) * (0.22 - 0.20) / (0.90 - 0.88)
        else:
            # For subtlety > 0.90, use minimum contrast
            effective_contrast = min_contrast
        
        # Ensure we never go below minimum
        effective_contrast = max(effective_contrast, min_contrast)
        
        dark_factor = 1.0 - effective_contrast  # Darkening for QR data areas (black modules)
        light_factor = 1.0 + effective_contrast  # Lightening for QR background areas (white modules)
        
        # Apply artistic embedding while preserving QR structure exactly
        result_array = base_array.copy()
        
        # CRITICAL: Preserve QR code structure with sufficient contrast
        # Black QR modules (data) = darken image
        # White QR modules (background) = lighten image
        # This maintains the EXACT same QR code pattern
        
        # Darken areas where QR code is black (data modules)
        dark_mask = qr_mask_3d  # 1 where QR is black (data)
        result_array = result_array * (1 - dark_mask * (1 - dark_factor))
        
        # Lighten areas where QR code is white (background)
        light_mask = 1 - qr_mask_3d  # 1 where QR is white (background)
        result_array = result_array * (1 + light_mask * (light_factor - 1))
        
        # Add stronger contrast boost to ensure QR structure is clear and scannable
        # This preserves the QR code pattern structure
        if contrast_boost > 0:
            # Create contrast based on QR pattern: -1 for black, +1 for white
            qr_contrast = (qr_mask_3d - 0.5) * 2
            # Apply contrast boost - much stronger for better scannability
            # Scale contrast boost based on subtlety - more aggressive for higher subtlety
            contrast_multiplier = 100 if subtlety >= 0.88 else 80
            contrast_strength = contrast_boost * contrast_multiplier
            result_array = result_array + (qr_contrast * contrast_strength)
        
        # Additional aggressive contrast enhancement for QR pattern
        # This ensures the QR code is clearly distinguishable from the background
        # Calculate the difference between black and white QR areas
        qr_pattern_diff = (qr_mask_3d - 0.5) * 2  # -1 for black, +1 for white
        # Apply additional pattern enhancement
        pattern_enhancement = 15.0  # Additional 15 pixel boost to pattern
        result_array = result_array + (qr_pattern_diff * pattern_enhancement)
        
        # Additional edge enhancement to preserve QR module boundaries
        # This helps QR scanners detect module edges more clearly
        try:
            from scipy import ndimage
            # Apply slight edge enhancement to QR pattern areas
            # This helps preserve sharp boundaries between modules
            qr_edges = ndimage.gaussian_filter(qr_binary.astype(float), sigma=0.5)
            qr_edges = np.abs(qr_edges - qr_binary.astype(float))
            edge_mask_3d = np.expand_dims(qr_edges, axis=2)
            # Slight sharpening at edges
            edge_boost = 5.0  # Small boost to edge contrast
            result_array = result_array + (edge_mask_3d * edge_boost * (qr_mask_3d - 0.5) * 2)
        except ImportError:
            # scipy not available, skip edge enhancement
            pass
        
        # Ensure values are in valid range
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        result_image = Image.fromarray(result_array)
        
        return result_image
    
    def enhance_qr_scannability(self,
                               generated_image: Image.Image,
                               qr_reference: Image.Image,
                               enhancement_strength: float = 0.15) -> Image.Image:
        """
        Enhance QR code scannability in ControlNet-generated images.
        Applies subtle contrast enhancement to preserve QR structure while maintaining artistic quality.
        
        Args:
            generated_image: The ControlNet-generated artistic image
            qr_reference: The original QR code reference (for pattern matching)
            enhancement_strength: Strength of enhancement (0.0-0.3, higher = more visible QR)
        
        Returns:
            Enhanced image with improved QR scannability
        """
        # Convert to RGB if needed
        if generated_image.mode != 'RGB':
            generated_image = generated_image.convert('RGB')
        if qr_reference.mode != 'RGB':
            qr_reference = qr_reference.convert('RGB')
        
        # Resize to match
        target_size = generated_image.size[0]
        qr_reference = qr_reference.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy arrays
        gen_array = np.array(generated_image, dtype=float)
        qr_array = np.array(qr_reference, dtype=float)
        
        # Create binary QR mask from reference
        qr_gray = np.dot(qr_array[...,:3], [0.2989, 0.5870, 0.1140])
        threshold = 127
        qr_binary = (qr_gray < threshold).astype(float)  # 1 for black (data), 0 for white (background)
        qr_mask_3d = np.expand_dims(qr_binary, axis=2)
        
        # Apply enhancement: strengthen contrast where QR code should be
        # This makes the QR pattern more visible while keeping it artistic
        result_array = gen_array.copy()
        
        # For black QR modules (data): darken slightly
        dark_mask = qr_mask_3d
        darken_factor = 1.0 - (enhancement_strength * 0.5)  # Subtle darkening
        result_array = result_array * (1 - dark_mask * (1 - darken_factor))
        
        # For white QR modules (background): lighten slightly
        light_mask = 1 - qr_mask_3d
        lighten_factor = 1.0 + (enhancement_strength * 0.5)  # Subtle lightening
        result_array = result_array * (1 + light_mask * (lighten_factor - 1))
        
        # Add contrast boost to QR pattern area
        contrast_boost = enhancement_strength * 30
        qr_contrast = (qr_mask_3d - 0.5) * 2  # -1 for black, +1 for white
        result_array = result_array + (qr_contrast * contrast_boost)
        
        # Ensure values are in valid range
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        result_image = Image.fromarray(result_array)
        
        return result_image
    
    def generate_with_controlnet(self,
                                prompt: str,
                                qr_data: str,
                                output_path: str = "artistic_qr_controlnet.png",
                                image_size: int = 512,
                                num_inference_steps: int = 30,
                                guidance_scale: float = 7.5,
                                controlnet_conditioning_scale: float = 1.5,
                                seed: Optional[int] = None,
                                negative_prompt: str = "blurry, distorted, unreadable qr, low quality",
                                qr_enhancement_strength: float = 0.15) -> Image.Image:
        """
        Generate artistic QR code using ControlNet (industry standard approach).
        
        This method generates the image WITH the QR code structure, creating seamless
        integration where the QR pattern is woven into the artwork itself.
        
        Args:
            prompt: Text prompt for image generation
            qr_data: Data to encode in QR code
            output_path: Path to save output image
            image_size: Size of output image
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            controlnet_conditioning_scale: ControlNet conditioning scale (higher = stronger QR structure)
            seed: Random seed
            negative_prompt: Negative prompt to avoid unwanted features
        
        Returns:
            Final artistic QR code image
        """
        print("=" * 60)
        print("ControlNet Artistic QR Code Generation")
        print("=" * 60)
        
        # Step 1: Create QR code
        print("\n[Step 1/2] Creating QR code...")
        qr_image = self.create_qr_code(qr_data, size=image_size)
        qr_image = qr_image.convert("RGB")
        print(f"QR code created: {image_size}x{image_size}")
        
        # Save original QR code for reference
        qr_reference_path = output_path.replace('.png', '_original_qr.png')
        if not qr_reference_path.endswith('_original_qr.png'):
            qr_reference_path = output_path.rsplit('.', 1)[0] + '_original_qr.png'
        qr_image.save(qr_reference_path)
        print(f"Original QR code saved: {qr_reference_path}")
        
        # Step 2: Load ControlNet model
        print("\n[Step 2/3] Generating image with ControlNet...")
        pipe = self.load_controlnet_model()
        
        # Generate image with QR code as control signal
        print(f"Prompt: {prompt}")
        print(f"Generating with ControlNet (this may take a while)...")
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=qr_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator
        ).images[0]
        
        # Step 3: Enhance QR code scannability (post-processing)
        print("\n[Step 3/3] Enhancing QR code scannability...")
        image = self.enhance_qr_scannability(image, qr_image, enhancement_strength=qr_enhancement_strength)
        
        # Save result
        image.save(output_path)
        print(f"\n✓ ControlNet artistic QR code saved to: {output_path}")
        print(f"  Image size: {image_size}x{image_size}")
        print(f"  QR data: {qr_data}")
        print(f"  Prompt: {prompt}")
        
        return image
    
    def process(self,
                prompt: str,
                qr_data: str,
                output_path: str = "artistic_qr_output.png",
                image_size: int = 512,
                subtlety: float = 0.92,
                num_inference_steps: int = 50,
                guidance_scale: float = 7.5,
                seed: Optional[int] = None,
                use_controlnet: bool = False,
                controlnet_conditioning_scale: float = 1.5,
                qr_enhancement_strength: float = 0.15) -> Image.Image:
        """
        Complete pipeline: Generate image and embed QR code.
        
        Args:
            prompt: Text prompt for image generation
            qr_data: Data to encode in QR code
            output_path: Path to save output image
            image_size: Size of output image
            subtlety: QR code subtlety (0.85-0.95) - only used if use_controlnet=False
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            seed: Random seed
            use_controlnet: If True, use ControlNet for seamless integration (like reference image)
            controlnet_conditioning_scale: ControlNet conditioning scale (1.0-2.0)
        
        Returns:
            Final artistic QR code image
        """
        # Use ControlNet if requested (better integration like reference image)
        if use_controlnet:
            return self.generate_with_controlnet(
                prompt=prompt,
                qr_data=qr_data,
                output_path=output_path,
                image_size=image_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                seed=seed,
                qr_enhancement_strength=qr_enhancement_strength
            )
        
        # Otherwise use post-processing embedding
        print("=" * 60)
        print("Artistic QR Code Image Generation Pipeline")
        print("=" * 60)
        
        # Step 1: Create QR code
        print("\n[Step 1/3] Creating QR code...")
        qr_image = self.create_qr_code(qr_data, size=image_size)
        print(f"QR code created: {image_size}x{image_size}")
        
        # Save original QR code for reference/comparison
        qr_reference_path = output_path.replace('.png', '_original_qr.png')
        if not qr_reference_path.endswith('_original_qr.png'):
            qr_reference_path = output_path.rsplit('.', 1)[0] + '_original_qr.png'
        qr_image.save(qr_reference_path)
        print(f"Original QR code saved: {qr_reference_path} (for comparison)")
        
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
        # Use higher contrast boost for better scannability
        contrast_boost_value = 0.08 if subtlety >= 0.90 else 0.06
        final_image = self.embed_qr_artistically(
            generated_image,
            qr_image,
            subtlety=subtlety,
            contrast_boost=contrast_boost_value
        )
        
        # Save result
        final_image.save(output_path)
        print(f"\n✓ Artistic QR code image saved to: {output_path}")
        print(f"  Image size: {image_size}x{image_size}")
        print(f"  QR data: {qr_data}")
        print(f"  Prompt: {prompt}")
        
        # Validate QR code scannability
        try:
            from qr_validator import validate_qr_image
            print("\n" + "-" * 60)
            print("Validating QR code scannability...")
            print("-" * 60)
            validation_result = validate_qr_image(output_path, qr_data)
            
            # Get scannability level (like QR Code AI)
            from qr_validator import QRCodeValidator
            validator = QRCodeValidator()
            scannability_level = validator.assess_scannability_level(output_path)
            
            # Display scannability indicator
            scannability_icons = {
                'High Scannability': '[HIGH]',
                'Medium Scannability': '[MEDIUM]',
                'Low Scannability': '[LOW]',
                'No Scannable': '[NO SCAN]'
            }
            
            print(f"\nScannability Level: {scannability_icons.get(scannability_level, '')} {scannability_level}")
            
            if validation_result['scannable']:
                print(f"[SUCCESS] QR code is SCANNABLE!")
                if validation_result['data_decoded']:
                    print(f"   Decoded: {validation_result['data_decoded'][:50]}...")
                if validation_result.get('matches_expected'):
                    print(f"[SUCCESS] Data matches expected: {validation_result['matches_expected']}")
            else:
                print(f"[WARNING] QR code may not be scannable")
                print(f"   Methods tried: {', '.join(validation_result.get('methods_tried', []))}")
                if validation_result.get('errors'):
                    print(f"   Errors: {validation_result['errors']}")
                print(f"   Tip: Try adjusting subtlety parameter (current: {subtlety})")
                print(f"        Recommended: 0.85-0.90 for better scannability")
        except ImportError:
            print("\n⚠️  QR validation tools not available. Install pyzbar and opencv-python for validation.")
        except Exception as e:
            print(f"\n⚠️  Validation error: {str(e)}")
        
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

