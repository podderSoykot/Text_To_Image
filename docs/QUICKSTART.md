# Quick Start Guide - Artistic QR Code Pipeline

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Basic Example
```bash
python artistic_qr_pipeline.py \
    --prompt "A beautiful sunset over mountains" \
    --qr-data "https://example.com" \
    --output "my_artistic_qr.png"
```

### 3. Scan the Result!
Open `my_artistic_qr.png` and scan it with any QR code scanner app.

## 📝 Common Use Cases

### Product Promotion
```bash
python artistic_qr_pipeline.py \
    --prompt "Modern product on white background, professional photography" \
    --qr-data "https://myshop.com/product" \
    --output "product.png"
```

### Art Portfolio
```bash
python artistic_qr_pipeline.py \
    --prompt "Abstract digital art, vibrant colors" \
    --qr-data "https://myportfolio.com" \
    --output "portfolio.png" \
    --subtlety 0.93
```

### Event Poster
```bash
python artistic_qr_pipeline.py \
    --prompt "Concert poster, neon lights, urban style" \
    --qr-data "https://eventbrite.com/event" \
    --output "poster.png" \
    --subtlety 0.88
```

## ⚙️ Key Parameters

- `--subtlety`: How visible the QR code is
  - `0.85-0.88`: More visible, easier to scan
  - `0.90-0.92`: Balanced (recommended) ⭐
  - `0.93-0.95`: Very subtle, image is main focus

- `--size`: Image dimensions (512, 768, 1024)
- `--steps`: Quality vs speed (20-100, default: 50)
- `--seed`: For reproducible results

## 💡 Tips

1. **First Run**: Model will download (~4GB) - be patient!
2. **Memory**: If you get errors, use `--size 512` or smaller
3. **Quality**: More steps = better quality but slower
4. **Testing**: Start with `--steps 30` for faster testing

## 📚 Full Documentation

See `README_ARTISTIC_QR.md` for complete documentation.




