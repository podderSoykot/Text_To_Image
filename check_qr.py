"""Quick QR code validation check"""
from qr_validator import validate_qr_image

result = validate_qr_image('artistic_qr_output.png')

print('\n' + '='*60)
print('QR Code Validation Results')
print('='*60)
print(f'Image: artistic_qr_output.png')
print(f'Scannable: {"YES" if result["scannable"] else "NO"}')
if result.get('data_decoded'):
    print(f'Decoded Data: {result["data_decoded"]}')
print(f'Methods Used: {", ".join(result.get("methods_tried", [])) or "None"}')
print(f'Confidence: {result.get("confidence", 0):.2f}')
if result.get('errors'):
    print(f'\nErrors:')
    for error in result['errors']:
        print(f'  - {error}')
print('='*60)

