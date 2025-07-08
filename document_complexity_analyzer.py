"""
Document Complexity Analyzer for Fenix PDF Translation Pipeline

This module analyzes PDF documents to determine their complexity level,
enabling optimized processing strategies for different document types.
"""

import os
import logging
import fitz  # PyMuPDF
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DocumentComplexity(Enum):
    """Document complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

@dataclass
class ComplexityAnalysis:
    """Results of document complexity analysis"""
    complexity_level: DocumentComplexity
    confidence_score: float  # 0.0 to 1.0
    page_count: int
    has_images: bool
    has_tables: bool
    has_complex_formatting: bool
    has_toc: bool
    has_footnotes: bool
    has_equations: bool
    has_multi_column: bool
    has_bibliography: bool
    font_diversity: int
    processing_recommendation: str
    analysis_details: Dict[str, Any]

class DocumentComplexityAnalyzer:
    """
    Analyzes PDF documents to determine complexity level and optimal processing strategy.
    
    Simple Document Criteria:
    - No images/figures
    - No tables
    - No complex formatting (bold/italic only)
    - No footnotes/endnotes
    - No mathematical equations
    - No multi-column layouts
    - No headers/footers with complex content
    - Single font family throughout
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Complexity thresholds
        self.SIMPLE_FONT_DIVERSITY_THRESHOLD = 3
        self.SIMPLE_IMAGE_THRESHOLD = 0
        self.SIMPLE_TABLE_THRESHOLD = 0
        
        # Regex patterns for complexity detection
        self.bibliography_patterns = [
            r'(?i)^(bibliography|references|works cited|literatura|βιβλιογραφία)',
            r'(?i)^(sources|citations|further reading)',
            r'(?i)^(suggested reading|recommended reading)'
        ]
        
        self.footnote_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\d+\)',  # (1), (2), etc.
            r'^\d+\.',   # 1., 2., etc. at start of line
            r'^\d+\s',   # 1 , 2 , etc. at start of line
        ]
        
        self.equation_patterns = [
            r'[∑∫∂∆√π∞≤≥≠±×÷]',  # Mathematical symbols
            r'\$.*\$',  # LaTeX inline math
            r'\\[a-zA-Z]+\{',  # LaTeX commands
            r'[a-zA-Z]\s*=\s*[a-zA-Z0-9]',  # Variable assignments
        ]
        
        self.toc_patterns = [
            r'(?i)^(table of contents|contents|index)',
            r'(?i)^(πίνακας περιεχομένων|περιεχόμενα)',
            r'(?i)^(índice|tabla de contenidos)',
            r'\.{3,}\s*\d+$',  # Dot leaders with page numbers
        ]
    
    def analyze_document(self, pdf_path: str) -> ComplexityAnalysis:
        """
        Analyze a PDF document and return complexity assessment.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ComplexityAnalysis object with detailed assessment
        """
        try:
            self.logger.info(f"🔍 Analyzing document complexity: {os.path.basename(pdf_path)}")
            
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            # Initialize analysis results
            analysis_details = {
                'total_pages': len(doc),
                'images_found': 0,
                'tables_found': 0,
                'fonts_detected': set(),
                'has_multi_column': False,
                'has_complex_headers': False,
                'has_embedded_objects': False,
                'text_complexity_score': 0.0,
                'layout_complexity_score': 0.0,
                'content_complexity_score': 0.0
            }
            
            # Analyze each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                self._analyze_page(page, page_num, analysis_details)
            
            # Perform overall document analysis
            complexity_assessment = self._assess_overall_complexity(analysis_details)
            
            # Generate processing recommendation
            recommendation = self._generate_processing_recommendation(complexity_assessment)
            
            # Create final analysis result
            result = ComplexityAnalysis(
                complexity_level=complexity_assessment['level'],
                confidence_score=complexity_assessment['confidence'],
                page_count=analysis_details['total_pages'],
                has_images=analysis_details['images_found'] > 0,
                has_tables=analysis_details['tables_found'] > 0,
                has_complex_formatting=analysis_details['text_complexity_score'] > 0.3,
                has_toc=analysis_details.get('has_toc', False),
                has_footnotes=analysis_details.get('has_footnotes', False),
                has_equations=analysis_details.get('has_equations', False),
                has_multi_column=analysis_details['has_multi_column'],
                has_bibliography=analysis_details.get('has_bibliography', False),
                font_diversity=len(analysis_details['fonts_detected']),
                processing_recommendation=recommendation,
                analysis_details=analysis_details
            )
            
            self.logger.info(f"📊 Complexity Analysis Complete:")
            self.logger.info(f"   Level: {result.complexity_level.value.upper()}")
            self.logger.info(f"   Confidence: {result.confidence_score:.2f}")
            self.logger.info(f"   Recommendation: {result.processing_recommendation}")
            
            doc.close()
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing document complexity: {e}")
            # Return conservative complex assessment on error
            return ComplexityAnalysis(
                complexity_level=DocumentComplexity.COMPLEX,
                confidence_score=0.5,
                page_count=0,
                has_images=True,  # Conservative assumption
                has_tables=True,
                has_complex_formatting=True,
                has_toc=True,
                has_footnotes=True,
                has_equations=True,
                has_multi_column=True,
                has_bibliography=True,
                font_diversity=10,
                processing_recommendation="FULL_DIGITAL_TWIN",
                analysis_details={'error': str(e)}
            )
    
    def _analyze_page(self, page: fitz.Page, page_num: int, analysis_details: Dict) -> None:
        """Analyze a single page for complexity indicators"""
        try:
            # Get page text and blocks
            text = page.get_text()
            blocks = page.get_text("dict")["blocks"]
            
            # Analyze text content
            self._analyze_text_content(text, analysis_details)
            
            # Analyze layout structure
            self._analyze_layout_structure(blocks, analysis_details)
            
            # Analyze images and drawings
            self._analyze_visual_content(page, analysis_details)
            
            # Analyze font usage
            self._analyze_fonts(blocks, analysis_details)
            
        except Exception as e:
            self.logger.warning(f"Error analyzing page {page_num}: {e}")
    
    def _analyze_text_content(self, text: str, analysis_details: Dict) -> None:
        """Analyze text content for complexity indicators"""
        # Note: Bibliography detection removed - not relevant for simple document classification
        
        # Check for footnotes
        footnote_count = 0
        for pattern in self.footnote_patterns:
            footnote_count += len(re.findall(pattern, text))
        
        if footnote_count > 3:  # More than 3 footnote references
            analysis_details['has_footnotes'] = True
        
        # Check for equations
        equation_count = 0
        for pattern in self.equation_patterns:
            equation_count += len(re.findall(pattern, text))
        
        if equation_count > 2:  # More than 2 mathematical expressions
            analysis_details['has_equations'] = True
        
        # Check for TOC
        for pattern in self.toc_patterns:
            if re.search(pattern, text, re.MULTILINE):
                analysis_details['has_toc'] = True
                break
        
        # Calculate text complexity score
        complexity_indicators = [
            len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0,  # Special characters ratio
            len(re.findall(r'[A-Z]{2,}', text)) / len(text.split()) if text else 0,  # All-caps words ratio
            len(re.findall(r'\d+', text)) / len(text.split()) if text else 0,  # Numbers ratio
        ]
        
        analysis_details['text_complexity_score'] = sum(complexity_indicators) / len(complexity_indicators)
    
    def _analyze_layout_structure(self, blocks: List, analysis_details: Dict) -> None:
        """Analyze layout structure for complexity indicators"""
        if not blocks:
            return
        
        # Check for multi-column layout
        text_blocks = [block for block in blocks if "lines" in block]
        if len(text_blocks) > 1:
            # Simple heuristic: if text blocks are side by side, it's multi-column
            x_positions = [block["bbox"][0] for block in text_blocks]
            if len(set(x_positions)) > 1:
                analysis_details['has_multi_column'] = True
        
        # Check for tables (simple heuristic)
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = "".join(span["text"] for span in line["spans"])
                    # Look for table-like patterns
                    if re.search(r'\|\s*\w+\s*\|', line_text) or \
                       re.search(r'\t.*\t.*\t', line_text) or \
                       line_text.count('|') > 2:
                        analysis_details['tables_found'] += 1
                        break
        
        # Calculate layout complexity score
        layout_indicators = [
            1.0 if analysis_details['has_multi_column'] else 0.0,
            min(analysis_details['tables_found'] / 5.0, 1.0),  # Normalize table count
            len(text_blocks) / 20.0 if text_blocks else 0.0,  # Block density
        ]
        
        analysis_details['layout_complexity_score'] = sum(layout_indicators) / len(layout_indicators)
    
    def _analyze_visual_content(self, page: fitz.Page, analysis_details: Dict) -> None:
        """Analyze visual content (images, drawings)"""
        try:
            # Get image list
            image_list = page.get_images()
            analysis_details['images_found'] += len(image_list)
            
            # Check for drawings/vector graphics
            drawings = page.get_drawings()
            if drawings:
                analysis_details['images_found'] += len(drawings)
            
        except Exception as e:
            self.logger.debug(f"Error analyzing visual content: {e}")
    
    def _analyze_fonts(self, blocks: List, analysis_details: Dict) -> None:
        """Analyze font usage for complexity indicators"""
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_name = span.get("font", "")
                        if font_name:
                            analysis_details['fonts_detected'].add(font_name)
    
    def _assess_overall_complexity(self, analysis_details: Dict) -> Dict[str, Any]:
        """Assess overall document complexity based on analysis"""
        
        # Calculate complexity scores
        scores = {
            'visual_content': min((analysis_details['images_found'] + analysis_details['tables_found']) / 5.0, 1.0),
            'font_diversity': min(len(analysis_details['fonts_detected']) / self.SIMPLE_FONT_DIVERSITY_THRESHOLD, 1.0),
            'text_complexity': analysis_details['text_complexity_score'],
            'layout_complexity': analysis_details['layout_complexity_score'],
            'special_features': sum([
                analysis_details.get('has_toc', False),
                analysis_details.get('has_footnotes', False),
                analysis_details.get('has_equations', False),
                analysis_details['has_multi_column'],
            ]) / 4.0
        }
        
        # Calculate weighted overall complexity score
        weights = {
            'visual_content': 0.30,
            'font_diversity': 0.20,
            'text_complexity': 0.20,
            'layout_complexity': 0.15,
            'special_features': 0.15
        }
        
        overall_score = sum(scores[key] * weights[key] for key in scores)
        
        # Determine complexity level
        if overall_score < 0.3:
            level = DocumentComplexity.SIMPLE
            confidence = 0.9 - (overall_score * 0.5)  # Higher confidence for clearly simple docs
        elif overall_score < 0.7:
            level = DocumentComplexity.MODERATE
            confidence = 0.7  # Moderate confidence for middle ground
        else:
            level = DocumentComplexity.COMPLEX
            confidence = 0.8 + (overall_score - 0.7) * 0.5  # Higher confidence for clearly complex docs
        
        return {
            'level': level,
            'confidence': min(confidence, 1.0),
            'overall_score': overall_score,
            'component_scores': scores
        }
    
    def _generate_processing_recommendation(self, complexity_assessment: Dict) -> str:
        """Generate processing recommendation based on complexity assessment"""
        
        level = complexity_assessment['level']
        confidence = complexity_assessment['confidence']
        
        if level == DocumentComplexity.SIMPLE and confidence > 0.8:
            return "FAST_TRACK_SIMPLE"
        elif level == DocumentComplexity.SIMPLE:
            return "FAST_TRACK_WITH_VALIDATION"
        elif level == DocumentComplexity.MODERATE:
            return "HYBRID_PROCESSING"
        else:
            return "FULL_DIGITAL_TWIN"
    
    def should_use_fast_track(self, analysis: ComplexityAnalysis) -> bool:
        """
        Determine if document should use fast-track processing.
        
        Args:
            analysis: ComplexityAnalysis result
            
        Returns:
            True if fast-track processing is recommended
        """
        return (
            analysis.complexity_level == DocumentComplexity.SIMPLE and
            analysis.confidence_score > 0.7 and
            not analysis.has_images and
            not analysis.has_tables and
            analysis.font_diversity <= self.SIMPLE_FONT_DIVERSITY_THRESHOLD
        )
    
    def should_skip_toc_generation(self, analysis: ComplexityAnalysis) -> bool:
        """
        Determine if TOC generation should be skipped.
        
        Args:
            analysis: ComplexityAnalysis result
            
        Returns:
            True if TOC generation should be skipped
        """
        return not analysis.has_toc
    
    def get_processing_strategy(self, analysis: ComplexityAnalysis) -> str:
        """
        Get recommended processing strategy based on analysis.
        
        Args:
            analysis: ComplexityAnalysis result
            
        Returns:
            Processing strategy identifier
        """
        if self.should_use_fast_track(analysis):
            return "simple_text_extraction"
        elif analysis.complexity_level == DocumentComplexity.MODERATE:
            return "hybrid_processing"
        else:
            return "full_digital_twin"


# Global instance for easy access
document_complexity_analyzer = DocumentComplexityAnalyzer() 