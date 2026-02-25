# Artistic QR Code Generator - FastAPI

REST API for generating AI-artistic images with embedded scannable QR codes.

## Installation

```bash
pip install -r requirements.txt
```

## Running the API

### Development Server

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Production Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### 1. Root Endpoint
```
GET /
```
Returns API information.

### 2. Health Check
```
GET /health
```
Check API health and pipeline status.

### 3. Generate QR Code
```
POST /generate
```

**Request Body:**
```json
{
  "prompt": "A beautiful sunset over mountains",
  "qr_data": "https://example.com",
  "image_size": 512,
  "subtlety": 0.92,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "seed": null,
  "validate_qr": true
}
```

**Response:**
```json
{
  "success": true,
  "image_path": "/download/qr_12345.png",
  "qr_reference_path": "/download/qr_12345_original_qr.png",
  "scannability_level": "High Scannability",
  "scannable": true,
  "decoded_data": "https://example.com",
  "message": "QR code generated successfully",
  "job_id": "12345-67890-abcde"
}
```

### 4. Download Generated Image
```
GET /download/{filename}
```
Download a generated QR code image.

### 5. Validate QR Code
```
GET /validate/{filename}?expected_data=your_data
```
Validate QR code scannability.

### 6. Statistics
```
GET /stats
```
Get API statistics (file count, total size).

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Example Usage

### Using Python Requests

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "Futuristic cityscape, neon lights",
    "qr_data": "https://mywebsite.com",
    "subtlety": 0.90
})

result = response.json()
print(f"Image: http://localhost:8000{result['image_path']}")
```

### Using cURL

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A dog in the sky with clouds",
    "qr_data": "https://example.com",
    "subtlety": 0.92
  }'
```

### Using the Example Client

```python
from api_client_example import generate_qr_code

result = generate_qr_code(
    prompt="Beautiful landscape",
    qr_data="https://example.com"
)
```

## Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text prompt for image generation |
| `qr_data` | string | required | Data to encode in QR code |
| `image_size` | integer | 512 | Image size (256-1024) |
| `subtlety` | float | 0.92 | QR subtlety (0.85-0.95) |
| `num_inference_steps` | integer | 50 | Inference steps (20-100) |
| `guidance_scale` | float | 7.5 | Guidance scale (1.0-20.0) |
| `seed` | integer | null | Random seed |
| `validate_qr` | boolean | true | Validate QR scannability |

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether generation succeeded |
| `image_path` | string | Path to download generated image |
| `qr_reference_path` | string | Path to original QR code |
| `scannability_level` | string | Scannability level (High/Medium/Low/No) |
| `scannable` | boolean | Whether QR is scannable |
| `decoded_data` | string | Decoded QR code data |
| `message` | string | Status message |
| `job_id` | string | Unique job identifier |

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: File not found
- `500`: Server error

## File Management

- Generated files are stored in `outputs/` directory
- Files are automatically cleaned up after 24 hours
- Original QR codes are saved for comparison

## Performance

- **First Request**: ~30-60 seconds (model loading)
- **Subsequent Requests**: ~10-30 seconds (CPU) or ~5-10 seconds (GPU)
- **Concurrent Requests**: Supported (use workers in production)

## Security Notes

- CORS is enabled for all origins (adjust for production)
- Files are stored locally (consider cloud storage for production)
- No authentication (add for production use)

## Production Deployment

1. Use multiple workers: `--workers 4`
2. Add authentication/API keys
3. Use reverse proxy (nginx)
4. Configure CORS properly
5. Use cloud storage for files
6. Add rate limiting
7. Monitor with logging





