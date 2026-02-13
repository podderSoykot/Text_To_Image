"""Detailed QR code validation with image preprocessing"""
import cv2
import numpy as np
from PIL import Image

def preprocess_for_qr(image_path):
    """Preprocess image to improve QR code detection"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to enhance contrast
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Try adaptive thresholding
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    
    return gray, thresh, adaptive

def test_qr_detection(image_path):
    """Test QR code detection with multiple methods"""
    print(f"\nTesting QR code detection for: {image_path}")
    print("="*60)
    
    # Method 1: OpenCV QRCodeDetector
    try:
        img = cv2.imread(image_path)
        detector = cv2.QRCodeDetector()
        retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(img)
        
        if retval and decoded_info:
            print("[SUCCESS] OpenCV Detection: SUCCESS")
            for i, info in enumerate(decoded_info):
                if info:
                    print(f"   QR Code {i+1}: {info}")
            return True
        else:
            print("[FAILED] OpenCV Detection: FAILED")
    except Exception as e:
        print(f"[ERROR] OpenCV Error: {e}")
    
    # Method 2: Preprocessed images
    print("\nTrying with preprocessed images...")
    try:
        gray, thresh, adaptive = preprocess_for_qr(image_path)
        
        # Test on grayscale
        detector = cv2.QRCodeDetector()
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(gray)
        if retval and decoded_info:
            print("[SUCCESS] Grayscale Detection: SUCCESS")
            for info in decoded_info:
                if info:
                    print(f"   Data: {info}")
            return True
        
        # Test on thresholded
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(thresh)
        if retval and decoded_info:
            print("[SUCCESS] Threshold Detection: SUCCESS")
            for info in decoded_info:
                if info:
                    print(f"   Data: {info}")
            return True
        
        # Test on adaptive threshold
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(adaptive)
        if retval and decoded_info:
            print("[SUCCESS] Adaptive Threshold Detection: SUCCESS")
            for info in decoded_info:
                if info:
                    print(f"   Data: {info}")
            return True
        
        print("[FAILED] All preprocessing methods failed")
    except Exception as e:
        print(f"❌ Preprocessing Error: {e}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("The QR code may be too subtle. Try:")
    print("1. Reduce subtlety parameter (0.85-0.90 instead of 0.92)")
    print("2. Increase contrast_boost in embedding function")
    print("3. Test with a physical QR scanner app")
    print("="*60)
    
    return False

if __name__ == "__main__":
    test_qr_detection("artistic_qr_output.png")

