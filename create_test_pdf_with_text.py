#!/usr/bin/env python3
"""
Create a simple test PDF with extractable text content for testing translation
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import os

def create_test_pdf_with_text(filename="test_document_with_text.pdf"):
    """Create a test PDF with actual text content"""
    
    # Create the PDF document
    doc = SimpleDocTemplate(filename, pagesize=A4)
    
    # Get sample styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12,
        alignment=0  # Left alignment
    )
    
    # Create content
    story = []
    
    # Title
    title = Paragraph("Document Translation Test", title_style)
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Introduction
    intro = Paragraph("""
    This is a test document created to verify the translation functionality 
    of the PyMuPDF-YOLO integrated document processing pipeline. The document 
    contains various types of content that should be extracted and translated 
    correctly.
    """, body_style)
    story.append(intro)
    story.append(Spacer(1, 12))
    
    # Section 1
    section1_title = Paragraph("Section 1: Technical Overview", styles['Heading2'])
    story.append(section1_title)
    
    section1_content = Paragraph("""
    The enhanced document processing system combines PyMuPDF for high-fidelity 
    text extraction with YOLO for layout analysis. This integration provides 
    intelligent content mapping and processing routing based on document 
    characteristics. Pure text documents use fast PyMuPDF-only processing, 
    while mixed content documents use coordinate-based extraction.
    """, body_style)
    story.append(section1_content)
    story.append(Spacer(1, 12))
    
    # Section 2
    section2_title = Paragraph("Section 2: Translation Features", styles['Heading2'])
    story.append(section2_title)
    
    section2_content = Paragraph("""
    The translation service supports multiple languages and includes advanced 
    features such as context-aware translation, glossary management, and 
    markdown-aware processing. The system preserves document structure and 
    formatting while providing accurate translations.
    """, body_style)
    story.append(section2_content)
    story.append(Spacer(1, 12))
    
    # List example
    list_title = Paragraph("Key Features:", styles['Heading3'])
    story.append(list_title)
    
    features = [
        "High-fidelity text extraction using PyMuPDF",
        "Layout analysis with YOLO neural networks", 
        "Intelligent processing strategy routing",
        "Context-aware translation services",
        "Markdown structure preservation",
        "Multi-format output generation"
    ]
    
    for feature in features:
        bullet_point = Paragraph(f"• {feature}", body_style)
        story.append(bullet_point)
    
    story.append(Spacer(1, 12))
    
    # Conclusion
    conclusion_title = Paragraph("Conclusion", styles['Heading2'])
    story.append(conclusion_title)
    
    conclusion_content = Paragraph("""
    This document demonstrates the capabilities of the integrated translation 
    system. When processed, it should generate translated output in the target 
    language while preserving the original structure and formatting. The system 
    should create a Word document, processing report, and extract any non-text 
    elements to a separate images folder.
    """, body_style)
    story.append(conclusion_content)
    
    # Build the PDF
    doc.build(story)
    
    print(f"✅ Test PDF created: {filename}")
    return filename

if __name__ == "__main__":
    create_test_pdf_with_text() 