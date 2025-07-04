# FENIX Document Translation Pipeline - Dependency Installation Guide

## 🚀 Quick Installation

To install all required dependencies for the FENIX document translation pipeline:

```bash
pip install -r requirements.txt
```

## 📋 Critical Dependencies

### Core Translation and PDF Processing
- `google-generativeai>=0.3.0` - Gemini API for translation
- `pymupdf>=1.23.0` - PDF content extraction
- `python-dotenv>=1.0.0` - Environment configuration

### Document Generation and Conversion ⚠️ IMPORTANT
- `python-docx>=0.8.11` - Word document creation
- **`docx2pdf>=0.1.8`** - Word to PDF conversion (**NOT `doc2pdf`**)
- `docx2txt>=0.8` - Alternative text extraction

### Computer Vision and Layout Analysis
- `torch>=2.0.0` - PyTorch for YOLO
- `torchvision>=0.15.0` - Vision utilities
- `ultralytics>=8.0.0` - YOLO implementation

## 🔧 Common Installation Issues

### Issue: "docx2pdf not found" Error
**Problem:** You may have installed `doc2pdf` instead of `docx2pdf`

**Solution:**
```bash
# WRONG (don't install this)
pip uninstall doc2pdf

# CORRECT (install this)
pip install docx2pdf
```

### Issue: PDF Conversion Fails
**Problem:** Missing PDF conversion libraries

**Solution:**
```bash
pip install docx2pdf docx2txt reportlab
```

### Issue: YOLO Model Loading Fails
**Problem:** Missing PyTorch or CUDA dependencies

**Solution:**
```bash
# For CPU-only
pip install torch torchvision ultralytics

# For CUDA (if you have NVIDIA GPU)
pip install torch torchvision ultralytics --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 Manual Dependency Installation

If you prefer to install dependencies individually:

```bash
# Core translation
pip install google-generativeai pymupdf python-dotenv

# Document processing (CRITICAL: note the exact package names)
pip install python-docx docx2pdf docx2txt

# Computer vision
pip install torch torchvision ultralytics

# Data processing
pip install numpy Pillow pydantic

# Additional utilities
pip install networkx pandas tqdm

# Optional web interface
pip install fastapi uvicorn
```

## ✅ Verification

To verify your installation is correct, run:

```python
# Test critical imports
try:
    import docx2pdf
    print("✅ docx2pdf installed correctly")
except ImportError:
    print("❌ docx2pdf missing - run: pip install docx2pdf")

try:
    from docx import Document
    print("✅ python-docx installed correctly")
except ImportError:
    print("❌ python-docx missing - run: pip install python-docx")

try:
    import fitz  # PyMuPDF
    print("✅ PyMuPDF installed correctly")
except ImportError:
    print("❌ PyMuPDF missing - run: pip install pymupdf")
```

## 🚨 Important Notes

1. **Package Name Confusion:** Make sure to install `docx2pdf` (with 'x'), not `doc2pdf`
2. **Version Compatibility:** Use the version constraints in `requirements.txt` to avoid conflicts
3. **CUDA Support:** If you have an NVIDIA GPU, install the CUDA version of PyTorch for better performance
4. **Virtual Environment:** Always use a virtual environment to avoid dependency conflicts

## 🔍 Troubleshooting

If you encounter any dependency issues:

1. **Check the exact error message** in the logs
2. **Verify package names** - common mistakes include `doc2pdf` vs `docx2pdf`
3. **Update pip** before installing: `pip install --upgrade pip`
4. **Use virtual environment** to avoid conflicts
5. **Check system requirements** for CUDA/GPU dependencies

For additional help, check the error messages which now provide specific installation instructions. 