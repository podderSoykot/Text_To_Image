"""
Example Usage of Artistic QR Code Pipeline
==========================================
Demonstrates various ways to use the pipeline
"""

from artistic_qr_pipeline import ArtisticQRPipeline

def example_basic():
    """Basic usage example."""
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    pipeline = ArtisticQRPipeline()
    
    pipeline.process(
        prompt="A beautiful sunset over mountains, cinematic lighting",
        qr_data="https://example.com",
        output_path="example_basic.png"
    )


def example_custom_parameters():
    """Example with custom parameters."""
    print("\n" + "="*60)
    print("Example 2: Custom Parameters")
    print("="*60)
    
    pipeline = ArtisticQRPipeline()
    
    pipeline.process(
        prompt="Futuristic cityscape at night, neon lights, cyberpunk style",
        qr_data="https://mywebsite.com",
        output_path="example_custom.png",
        image_size=1024,
        subtlety=0.90,
        num_inference_steps=75,
        guidance_scale=8.0,
        seed=42
    )


def example_step_by_step():
    """Step-by-step control example."""
    print("\n" + "="*60)
    print("Example 3: Step-by-Step Control")
    print("="*60)
    
    pipeline = ArtisticQRPipeline()
    
    # Step 1: Load model
    pipeline.load_model()
    
    # Step 2: Create QR code
    print("\nCreating QR code...")
    qr_code = pipeline.create_qr_code(
        data="Contact: info@example.com",
        size=512,
        error_correction="H"
    )
    
    # Step 3: Generate image
    print("\nGenerating image...")
    image = pipeline.generate_image(
        prompt="Abstract art, vibrant colors, modern design",
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=123
    )
    
    # Step 4: Embed QR code
    print("\nEmbedding QR code...")
    result = pipeline.embed_qr_artistically(
        image,
        qr_code,
        subtlety=0.92,
        contrast_boost=0.05
    )
    
    # Step 5: Save
    result.save("example_step_by_step.png")
    print("\nSaved to: example_step_by_step.png")


def example_different_subtleties():
    """Example showing different subtlety levels."""
    print("\n" + "="*60)
    print("Example 4: Different Subtlety Levels")
    print("="*60)
    
    pipeline = ArtisticQRPipeline()
    pipeline.load_model()
    
    prompt = "A serene landscape with mountains and lake"
    qr_data = "https://example.com"
    
    # Generate base image once
    base_image = pipeline.generate_image(prompt, seed=42)
    base_image = base_image.resize((512, 512))
    
    # Create QR code
    qr_code = pipeline.create_qr_code(qr_data, size=512)
    
    # Test different subtlety levels
    subtlety_levels = [0.85, 0.90, 0.92, 0.95]
    
    for subtlety in subtlety_levels:
        result = pipeline.embed_qr_artistically(
            base_image.copy(),
            qr_code,
            subtlety=subtlety
        )
        output_path = f"example_subtlety_{subtlety}.png"
        result.save(output_path)
        print(f"Saved: {output_path} (subtlety: {subtlety})")


if __name__ == "__main__":
    # Run examples
    print("Artistic QR Code Pipeline - Examples")
    print("="*60)
    
    # Uncomment the example you want to run:
    
    # example_basic()
    # example_custom_parameters()
    # example_step_by_step()
    example_different_subtleties()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)

