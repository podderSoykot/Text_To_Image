"""Verify that generated QR code matches the original QR code structure"""
import qrcode
from PIL import Image
import numpy as np
from artistic_qr_pipeline import ArtisticQRPipeline
from qr_validator import validate_qr_image

def verify_qr_structure_match(original_qr_path, embedded_image_path, expected_data):
    """Verify that embedded QR code matches original structure"""
    
    print("\n" + "="*60)
    print("QR Code Structure Verification")
    print("="*60)
    
    # Load images
    original_qr = Image.open(original_qr_path)
    embedded = Image.open(embedded_image_path)
    
    # Convert to grayscale for comparison
    original_gray = np.array(original_qr.convert('L'))
    embedded_gray = np.array(embedded.convert('L'))
    
    # Threshold both to binary
    original_binary = (original_gray < 127).astype(int)
    embedded_binary = (embedded_gray < 127).astype(int)
    
    # Resize to same size if needed
    if original_binary.shape != embedded_binary.shape:
        original_binary = np.array(Image.fromarray((original_binary * 255).astype(np.uint8)).resize((embedded_binary.shape[1], embedded_binary.shape[0]))).astype(float) / 255.0
        original_binary = (original_binary < 0.5).astype(int)
    
    # Calculate structure similarity
    # Compare QR code pattern (corner markers and structure)
    # Focus on key QR code features: corner squares and alignment patterns
    
    # Extract corner regions (QR codes have 3 corner squares)
    size = min(original_binary.shape)
    corner_size = size // 10  # Approximate corner square size
    
    # Top-left corner
    orig_corner_tl = original_binary[:corner_size, :corner_size]
    embed_corner_tl = embedded_binary[:corner_size, :corner_size]
    
    # Top-right corner
    orig_corner_tr = original_binary[:corner_size, -corner_size:]
    embed_corner_tr = embedded_binary[:corner_size, -corner_size:]
    
    # Bottom-left corner
    orig_corner_bl = original_binary[-corner_size:, :corner_size]
    embed_corner_bl = embedded_binary[-corner_size:, :corner_size]
    
    # Calculate similarity for each corner
    similarity_tl = np.mean(orig_corner_tl == embed_corner_tl)
    similarity_tr = np.mean(orig_corner_tr == embed_corner_tr)
    similarity_bl = np.mean(orig_corner_bl == embed_corner_bl)
    
    avg_similarity = (similarity_tl + similarity_tr + similarity_bl) / 3
    
    print(f"\nOriginal QR: {original_qr_path}")
    print(f"Embedded Image: {embedded_image_path}")
    print(f"\nStructure Similarity:")
    print(f"  Top-left corner: {similarity_tl*100:.1f}%")
    print(f"  Top-right corner: {similarity_tr*100:.1f}%")
    print(f"  Bottom-left corner: {similarity_bl*100:.1f}%")
    print(f"  Average: {avg_similarity*100:.1f}%")
    
    # Validate scannability
    print(f"\nScannability Test:")
    result = validate_qr_image(embedded_image_path, expected_data)
    print(f"  Scannable: {result['scannable']}")
    if result.get('data_decoded'):
        print(f"  Decoded Data: {result['data_decoded']}")
        print(f"  Matches Expected: {result.get('matches_expected', False)}")
    
    print("\n" + "="*60)
    
    return avg_similarity > 0.7 and result['scannable']

if __name__ == "__main__":
    # Test with generated images
    verify_qr_structure_match(
        "original_qr.png",
        "test_qr_subtlety_0.90.png",
        "https://test.com"
    )

