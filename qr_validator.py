"""
QR Code Validation and Testing Utilities
========================================
Tools to validate and test QR code scannability in generated images.
"""

import os
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict
import qrcode
from qrcode import QRCode

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("Warning: pyzbar not available. QR validation will be limited.")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: opencv-python not available. Some validation features disabled.")


class QRCodeValidator:
    """
    Validates QR code scannability in images.
    """
    
    def __init__(self):
        self.validation_results = []
    
    def validate_qr_code(self, 
                        image_path: str, 
                        expected_data: Optional[str] = None) -> Dict:
        """
        Validate if QR code in image is scannable.
        
        Args:
            image_path: Path to image with QR code
            expected_data: Expected data in QR code (for verification)
        
        Returns:
            Dictionary with validation results
        """
        result = {
            'image_path': image_path,
            'scannable': False,
            'data_decoded': None,
            'matches_expected': False,
            'confidence': 0.0,
            'methods_tried': [],
            'errors': []
        }
        
        if not os.path.exists(image_path):
            result['errors'].append(f"Image file not found: {image_path}")
            return result
        
        # Try multiple decoding methods
        image = Image.open(image_path)
        
        # Method 1: Direct QR code library
        try:
            decoded = self._decode_with_qrcode_lib(image)
            if decoded:
                result['scannable'] = True
                result['data_decoded'] = decoded
                result['methods_tried'].append('qrcode_library')
                result['confidence'] = 1.0
        except Exception as e:
            result['errors'].append(f"QRCode library error: {str(e)}")
        
        # Method 2: PyZBar (if available)
        if PYZBAR_AVAILABLE and not result['scannable']:
            try:
                decoded = self._decode_with_pyzbar(image)
                if decoded:
                    result['scannable'] = True
                    result['data_decoded'] = decoded
                    result['methods_tried'].append('pyzbar')
                    result['confidence'] = 0.9
            except Exception as e:
                result['errors'].append(f"PyZBar error: {str(e)}")
        
        # Method 3: OpenCV + QRCodeDetector (if available)
        if OPENCV_AVAILABLE and not result['scannable']:
            try:
                decoded = self._decode_with_opencv(image_path)
                if decoded:
                    result['scannable'] = True
                    result['data_decoded'] = decoded
                    result['methods_tried'].append('opencv')
                    result['confidence'] = 0.8
            except Exception as e:
                result['errors'].append(f"OpenCV error: {str(e)}")
        
        # Verify if decoded data matches expected
        if expected_data and result['data_decoded']:
            result['matches_expected'] = (result['data_decoded'] == expected_data)
        
        return result
    
    def _decode_with_qrcode_lib(self, image: Image.Image) -> Optional[str]:
        """Try to decode using qrcode library (limited - mainly for validation)."""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # This is a basic check - qrcode library doesn't have a decoder
        # We'll use it to verify structure
        return None
    
    def _decode_with_pyzbar(self, image: Image.Image) -> Optional[str]:
        """Decode QR code using PyZBar."""
        if not PYZBAR_AVAILABLE:
            return None
        
        # Convert PIL to numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Decode
        decoded_objects = pyzbar_decode(img_array)
        
        if decoded_objects:
            # Get first QR code found
            for obj in decoded_objects:
                if obj.type == 'QRCODE':
                    return obj.data.decode('utf-8')
        
        return None
    
    def _decode_with_opencv(self, image_path: str) -> Optional[str]:
        """Decode QR code using OpenCV QRCodeDetector."""
        if not OPENCV_AVAILABLE:
            return None
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Initialize QR code detector
        detector = cv2.QRCodeDetector()
        
        # Detect and decode
        retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(img)
        
        if retval and decoded_info:
            # Return first decoded string
            for info in decoded_info:
                if info:
                    return info
        
        return None
    
    def test_scannability(self, 
                         image_path: str, 
                         expected_data: str,
                         verbose: bool = True) -> bool:
        """
        Test if QR code is scannable and returns expected data.
        
        Args:
            image_path: Path to image
            expected_data: Expected QR code data
            verbose: Print detailed results
        
        Returns:
            True if scannable and matches expected data
        """
        result = self.validate_qr_code(image_path, expected_data)
        
        if verbose:
            self._print_validation_result(result)
        
        return result['scannable'] and result['matches_expected']
    
    def _print_validation_result(self, result: Dict):
        """Print validation results in readable format."""
        print("\n" + "="*60)
        print("QR Code Validation Results")
        print("="*60)
        print(f"Image: {result['image_path']}")
        print(f"Scannable: {'✅ YES' if result['scannable'] else '❌ NO'}")
        
        if result['data_decoded']:
            print(f"Decoded Data: {result['data_decoded']}")
        
        if result.get('expected_data'):
            match_status = "✅ MATCHES" if result['matches_expected'] else "❌ DOES NOT MATCH"
            print(f"Expected Data Match: {match_status}")
        
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Methods Used: {', '.join(result['methods_tried']) if result['methods_tried'] else 'None'}")
        
        if result['errors']:
            print(f"\nErrors:")
            for error in result['errors']:
                print(f"  - {error}")
        
        print("="*60 + "\n")
    
    def batch_validate(self, 
                      image_paths: List[str], 
                      expected_data: Optional[str] = None) -> List[Dict]:
        """
        Validate multiple images.
        
        Args:
            image_paths: List of image paths
            expected_data: Expected data (same for all)
        
        Returns:
            List of validation results
        """
        results = []
        for path in image_paths:
            result = self.validate_qr_code(path, expected_data)
            results.append(result)
            self.validation_results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get statistics from validation results."""
        if not self.validation_results:
            return {'total': 0, 'scannable': 0, 'success_rate': 0.0}
        
        total = len(self.validation_results)
        scannable = sum(1 for r in self.validation_results if r['scannable'])
        success_rate = (scannable / total) * 100 if total > 0 else 0.0
        
        return {
            'total': total,
            'scannable': scannable,
            'not_scannable': total - scannable,
            'success_rate': success_rate
        }
    
    def assess_scannability_level(self, image_path: str) -> str:
        """
        Assess QR code scannability level (like QR Code AI).
        
        Returns:
            'No Scannable', 'Low Scannability', 'Medium Scannability', or 'High Scannability'
        """
        result = self.validate_qr_code(image_path)
        
        if result['scannable']:
            confidence = result.get('confidence', 0.0)
            methods_count = len(result.get('methods_tried', []))
            
            if confidence >= 0.9 and methods_count >= 2:
                return 'High Scannability'
            elif confidence >= 0.7 or methods_count >= 1:
                return 'Medium Scannability'
            else:
                return 'Low Scannability'
        else:
            return 'No Scannable'


def validate_qr_image(image_path: str, expected_data: Optional[str] = None) -> Dict:
    """
    Convenience function to validate a QR code image.
    
    Args:
        image_path: Path to image
        expected_data: Expected QR code data
    
    Returns:
        Validation result dictionary
    """
    validator = QRCodeValidator()
    return validator.validate_qr_code(image_path, expected_data)


def test_qr_scannability(image_path: str, expected_data: str) -> bool:
    """
    Convenience function to test QR code scannability.
    
    Args:
        image_path: Path to image
        expected_data: Expected QR code data
    
    Returns:
        True if scannable and matches expected
    """
    validator = QRCodeValidator()
    return validator.test_scannability(image_path, expected_data, verbose=True)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python qr_validator.py <image_path> [expected_data]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    expected_data = sys.argv[2] if len(sys.argv) > 2 else None
    
    validator = QRCodeValidator()
    result = validator.validate_qr_code(image_path, expected_data)
    validator._print_validation_result(result)

