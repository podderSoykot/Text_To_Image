"""
Fix QR code scannability in existing generated images.
This script re-embeds the QR code with improved contrast settings.
"""
import sys
from pathlib import Path
from PIL import Image
from artistic_qr_pipeline import ArtisticQRPipeline
from qr_validator import QRCodeValidator

def fix_qr_image(image_path: str, output_path: str = None, subtlety: float = 0.88):
    """
    Fix QR code scannability by re-embedding with better contrast.
    
    Args:
        image_path: Path to the generated image with QR code
        output_path: Path to save fixed image (default: adds '_fixed' to filename)
        subtlety: Subtlety level (lower = more visible QR, better scannability)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return False
    
    # Find original QR code
    qr_ref_path = image_path.parent / f"{image_path.stem}_original_qr.png"
    if not qr_ref_path.exists():
        print(f"Error: Original QR code not found: {qr_ref_path}")
        print("   Looking for: {qr_ref_path.name}")
        return False
    
    # Decode QR to get data
    print("Decoding original QR code to get data...")
    validator = QRCodeValidator()
    qr_result = validator.validate_qr_code(str(qr_ref_path))
    
    if not qr_result['scannable'] or not qr_result['data_decoded']:
        print(f"Error: Could not decode original QR code")
        return False
    
    qr_data = qr_result['data_decoded']
    print(f"QR Data: {qr_data[:50]}...")
    
    # Load the generated image (we'll extract the base image)
    # Actually, we need to re-embed, so we'll use the original QR and a base image
    # For now, let's try to enhance the existing image directly
    print("\nLoading images...")
    generated_image = Image.open(image_path)
    original_qr = Image.open(qr_ref_path)
    
    # Create pipeline
    print("Initializing pipeline...")
    pipeline = ArtisticQRPipeline()
    
    # Re-embed QR with better settings
    print(f"Re-embedding QR code with subtlety={subtlety} (lower = better scannability)...")
    fixed_image = pipeline.embed_qr_artistically(
        generated_image,
        original_qr,
        subtlety=subtlety,
        contrast_boost=0.08  # Increased contrast boost
    )
    
    # Save fixed image
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_fixed.png"
    else:
        output_path = Path(output_path)
    
    fixed_image.save(output_path)
    print(f"\nFixed image saved: {output_path}")
    
    # Validate the fixed image
    print("\nValidating fixed QR code...")
    fixed_result = validator.validate_qr_code(str(output_path), qr_data)
    
    print("\n" + "="*60)
    print("Validation Results")
    print("="*60)
    print(f"Original Image Scannable: {'YES' if validator.validate_qr_code(str(image_path))['scannable'] else 'NO'}")
    print(f"Fixed Image Scannable: {'YES' if fixed_result['scannable'] else 'NO'}")
    if fixed_result['scannable']:
        print(f"Decoded Data: {fixed_result['data_decoded']}")
        print(f"Matches Expected: {fixed_result.get('matches_expected', 'N/A')}")
    print("="*60)
    
    return fixed_result['scannable']


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_qr_scannability.py <image_path> [output_path] [subtlety]")
        print("\nExample:")
        print("  python fix_qr_scannability.py outputs/qr_xxx.png")
        print("  python fix_qr_scannability.py outputs/qr_xxx.png outputs/qr_xxx_fixed.png 0.88")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    subtlety = float(sys.argv[3]) if len(sys.argv) > 3 else 0.88
    
    success = fix_qr_image(image_path, output_path, subtlety)
    sys.exit(0 if success else 1)


