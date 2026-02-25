# QR Code Scannability Fix

## Issue
The generated QR code images were not scannable due to insufficient contrast in the embedding algorithm.

## Root Cause
The embedding algorithm was too subtle, especially with the default `subtlety=0.92` parameter:
- Only 8% darkening for black QR modules
- Only 8% lightening for white QR modules  
- Insufficient contrast boost (only 2.5 pixels)

QR scanners require at least 15% contrast between black and white modules to reliably decode QR codes.

## Fixes Applied

### 1. Improved Contrast Calculation
- Ensures minimum 15% contrast even at high subtlety (0.92-0.95)
- Better contrast mapping based on subtlety parameter:
  - `subtlety=0.85` → 20% contrast
  - `subtlety=0.90` → 17.5% contrast  
  - `subtlety=0.92` → 15% contrast (minimum)
  - `subtlety=0.95` → 15% contrast (minimum)

### 2. Increased Contrast Boost
- Default `contrast_boost` increased from `0.05` to `0.08`
- Contrast strength increased from 50 to 80 pixels
- Adaptive contrast boost: higher for subtlety >= 0.90

### 3. Edge Enhancement
- Added optional edge enhancement to preserve QR module boundaries
- Helps QR scanners detect module edges more clearly

## Recommendations

### For Better Scannability
1. **Use lower subtlety values** (0.85-0.90) for maximum scannability
2. **Regenerate images** with the fixed algorithm
3. **Test with multiple QR scanners** (phone cameras, dedicated scanners)

### Subtlety Guidelines
- **0.85-0.88**: More visible QR code, easiest to scan (recommended for production)
- **0.90-0.92**: Balanced - image prominent, QR scannable (good for most use cases)
- **0.93-0.95**: Very subtle QR code, may have scanning issues

### Regenerating Images
To regenerate an image with better scannability:

```python
from artistic_qr_pipeline import ArtisticQRPipeline

pipeline = ArtisticQRPipeline()
pipeline.load_model()

# Use lower subtlety for better scannability
pipeline.process(
    prompt="Your prompt here",
    qr_data="Your QR data here",
    output_path="output.png",
    subtlety=0.88,  # Lower = better scannability
    image_size=512
)
```

Or via API:
```json
{
    "prompt": "Your prompt",
    "qr_data": "Your QR data",
    "subtlety": 0.88,
    "validate_qr": true
}
```

## Testing
After regeneration, validate the QR code:
```python
from qr_validator import QRCodeValidator

validator = QRCodeValidator()
result = validator.validate_qr_code("output.png", "expected_data")
print(f"Scannable: {result['scannable']}")
```

## Notes
- The original QR code (`*_original_qr.png`) is always scannable
- The issue was in the artistic embedding process
- Fixed images will have better contrast while maintaining artistic quality


