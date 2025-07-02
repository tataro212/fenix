"""
PDF Parser Module for Ultimate PDF Translator

Handles PDF parsing, structure extraction, and content organization
"""

import os
import re
import logging
import fitz  # PyMuPDF
import math
from collections import defaultdict
from config_manager import config_manager
from utils import LIST_MARKER_REGEX, CHAPTER_TITLE_PATTERNS
from structured_document_model import (
    Document, ContentBlock, ContentType, Heading, Paragraph, ImagePlaceholder,
    Table, CodeBlock, ListItem, Footnote, Equation, Caption, Metadata,
    convert_legacy_structured_content_to_document, create_content_block_from_legacy
)
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class PDFParser:
    """Main PDF parsing functionality"""
    
    def __init__(self):
        self.settings = config_manager.pdf_processing_settings
        self.toc_compiled_patterns = [
            re.compile(r'^Table of Contents$', re.IGNORECASE),
            re.compile(r'^Contents$', re.IGNORECASE),
            re.compile(r'^Index$', re.IGNORECASE),
            re.compile(r'^Table des matières$', re.IGNORECASE),
            re.compile(r'^Inhalt$', re.IGNORECASE),
            re.compile(r'^Sommaire$', re.IGNORECASE)
        ]
        self.toc_structure_patterns = [
            re.compile(r'\.{3,}\s*\d+$'),  # Dots followed by page number
            re.compile(r'^\d+\.\d+\s+[A-Z]'),  # Numbered sections
            re.compile(r'^[A-Z][a-z]+\s+\d+$'),  # Chapter names with numbers
            re.compile(r'^\d+\.\s+[A-Z]'),  # Simple numbered sections
            re.compile(r'^[IVX]+\.\s+[A-Z]')  # Roman numerals
        ]
        self.min_image_size = (8, 8)  # Updated minimum image size threshold
        self.figure_pattern = re.compile(r'^(?:Figure|Fig\.?|Diagram|Schema)\s+\d+(?:\.\d+)?', re.IGNORECASE)

    def extract_images_from_pdf(self, pdf_filepath, output_image_folder):
        """Extract images from PDF and save to folder"""
        if not self.settings['extract_images']:
            logger.info("Image extraction is disabled in config.ini.")
            return []
        
        logger.info(f"--- Extracting Images from PDF: {os.path.basename(pdf_filepath)} ---")
        
        try:
            os.makedirs(output_image_folder, exist_ok=True)
        except Exception as e:
            logger.error(f"Could not create image output folder {output_image_folder}: {e}")
            return []
        
        all_extracted_image_refs = []
        
        try:
            doc = fitz.open(pdf_filepath)

            # First extract regular images
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Filter small or low-quality images with enhanced criteria
                        min_width = self.settings.get('min_image_width_px', 50)
                        min_height = self.settings.get('min_image_height_px', 50)

                        if (pix.width < min_width or pix.height < min_height):
                            logger.debug(f"Skipping small image: {pix.width}x{pix.height} (min: {min_width}x{min_height})")
                            pix = None
                            continue

                        # Filter out very thin images (likely lines or decorative elements)
                        aspect_ratio = max(pix.width, pix.height) / min(pix.width, pix.height)
                        if aspect_ratio > 20:  # Very thin images
                            logger.debug(f"Skipping thin image: {pix.width}x{pix.height} (aspect ratio: {aspect_ratio:.1f})")
                            pix = None
                            continue
                        
                        # Convert to RGB if necessary
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
                            img_filepath = os.path.join(output_image_folder, img_filename)
                            pix.save(img_filepath)
                            
                            # Get image position on page
                            img_rect = page.get_image_rects(xref)
                            if img_rect:
                                rect = img_rect[0]
                                image_ref = {
                                    'filename': img_filename,
                                    'filepath': img_filepath,
                                    'page_num': page_num + 1,
                                    'x0': rect.x0,
                                    'y0': rect.y0,
                                    'x1': rect.x1,
                                    'y1': rect.y1,
                                    'width': pix.width,
                                    'height': pix.height
                                }
                                all_extracted_image_refs.append(image_ref)
                                logger.debug(f"Extracted image: {img_filename}")
                        else:
                            # Convert CMYK to RGB
                            pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                            img_filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
                            img_filepath = os.path.join(output_image_folder, img_filename)
                            pix_rgb.save(img_filepath)
                            
                            # Get image position
                            img_rect = page.get_image_rects(xref)
                            if img_rect:
                                rect = img_rect[0]
                                image_ref = {
                                    'filename': img_filename,
                                    'filepath': img_filepath,
                                    'page_num': page_num + 1,
                                    'x0': rect.x0,
                                    'y0': rect.y0,
                                    'x1': rect.x1,
                                    'y1': rect.y1,
                                    'width': pix_rgb.width,
                                    'height': pix_rgb.height
                                }
                                all_extracted_image_refs.append(image_ref)
                                logger.debug(f"Extracted image: {img_filename}")
                            
                            pix_rgb = None
                        
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Could not extract image {img_index + 1} from page {page_num + 1}: {e}")
                        continue
            
            # Extract tables as images if enabled
            if self.settings.get('extract_tables_as_images', False):
                logger.info("Extracting tables as images...")
                table_images = self.extract_tables_as_images(doc, output_image_folder)
                all_extracted_image_refs.extend(table_images)

            # Extract equations as images if enabled
            if self.settings.get('extract_equations_as_images', False):
                logger.info("Extracting equations as images...")
                equation_images = self.extract_equations_as_images(doc, output_image_folder)
                all_extracted_image_refs.extend(equation_images)

            # Extract visual content areas if enabled
            if self.settings.get('extract_figures_by_caption', False):
                logger.info("Extracting visual content areas...")
                visual_images = self.extract_visual_content_areas(doc, output_image_folder)
                all_extracted_image_refs.extend(visual_images)

            doc.close()
            logger.info(f"Successfully extracted {len(all_extracted_image_refs)} images")
            return all_extracted_image_refs

        except Exception as e:
            logger.error(f"Error extracting images from PDF: {e}")
            return []
    
    def extract_cover_page_from_pdf(self, pdf_filepath, output_folder):
        """Extract the first page of PDF as cover page image"""
        logger.info(f"--- Extracting Cover Page from PDF: {os.path.basename(pdf_filepath)} ---")
        
        try:
            os.makedirs(output_folder, exist_ok=True)
        except Exception as e:
            logger.error(f"Could not create output folder {output_folder}: {e}")
            return None
        
        try:
            doc = fitz.open(pdf_filepath)
            
            if len(doc) == 0:
                logger.warning("PDF has no pages")
                doc.close()
                return None
            
            # Get first page
            first_page = doc[0]
            
            # Render page as image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = first_page.get_pixmap(matrix=mat)
            
            # Save cover page
            cover_filename = f"cover_page_{os.path.splitext(os.path.basename(pdf_filepath))[0]}.png"
            cover_filepath = os.path.join(output_folder, cover_filename)
            pix.save(cover_filepath)
            
            doc.close()
            
            cover_data = {
                'filename': cover_filename,
                'filepath': cover_filepath,
                'width': pix.width,
                'height': pix.height
            }
            
            logger.info(f"Cover page extracted: {cover_filename}")
            return cover_data
            
        except Exception as e:
            logger.error(f"Error extracting cover page: {e}")
            return None
    
    def groupby_images_by_page(self, all_extracted_image_refs):
        """Group extracted images by page number"""
        if not all_extracted_image_refs:
            return {}
        
        sorted_images = sorted(all_extracted_image_refs, key=lambda x: x['page_num'])
        images_by_page = defaultdict(list)
        
        for img_ref in sorted_images:
            images_by_page[img_ref['page_num']].append(img_ref)
        
        return dict(images_by_page)

    def extract_tables_as_images(self, doc, output_image_folder):
        """Extract table areas as images"""
        logger.info("--- Extracting Tables as Images ---")

        table_images = []
        min_columns = self.settings.get('min_table_columns', 2)
        min_rows = self.settings.get('min_table_rows', 2)
        min_width = self.settings.get('min_table_width_points', 100)
        min_height = self.settings.get('min_table_height_points', 50)

        for page_num in range(len(doc)):
            page = doc[page_num]

            try:
                # Detect table areas using text analysis
                table_areas = self._detect_table_areas(page, min_columns, min_rows)

                for table_index, table_area in enumerate(table_areas):
                    bbox = table_area['bbox']
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]

                    # Check minimum size requirements
                    if width >= min_width and height >= min_height:
                        # Render table area as image
                        table_image = self._render_area_as_image(
                            page, bbox, page_num + 1, f"table_{table_index + 1}",
                            output_image_folder, "table"
                        )

                        if table_image:
                            table_images.append(table_image)
                            logger.debug(f"Extracted table: {table_image['filename']}")

            except Exception as e:
                logger.warning(f"Could not extract tables from page {page_num + 1}: {e}")

        logger.info(f"Successfully extracted {len(table_images)} table images")
        return table_images

    def extract_equations_as_images(self, doc, output_image_folder):
        """Extract equation areas as images"""
        logger.info("--- Extracting Equations as Images ---")

        equation_images = []
        min_width = self.settings.get('min_equation_width_points', 30)
        min_height = self.settings.get('min_equation_height_points', 15)
        detect_math_symbols = self.settings.get('detect_math_symbols', True)

        for page_num in range(len(doc)):
            page = doc[page_num]

            try:
                # Detect equation areas using text and symbol analysis
                equation_areas = self._detect_equation_areas(page, detect_math_symbols)

                for eq_index, eq_area in enumerate(equation_areas):
                    bbox = eq_area['bbox']
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]

                    # Check minimum size requirements
                    if width >= min_width and height >= min_height:
                        # Render equation area as image
                        equation_image = self._render_area_as_image(
                            page, bbox, page_num + 1, f"equation_{eq_index + 1}",
                            output_image_folder, "equation"
                        )

                        if equation_image:
                            equation_images.append(equation_image)
                            logger.debug(f"Extracted equation: {equation_image['filename']}")

            except Exception as e:
                logger.warning(f"Could not extract equations from page {page_num + 1}: {e}")

        logger.info(f"Successfully extracted {len(equation_images)} equation images")
        return equation_images

    def extract_visual_content_areas(self, doc, output_image_folder):
        """Extract areas with visual content - simple and effective approach"""
        logger.info("--- Extracting Visual Content Areas ---")

        visual_images = []
        min_width = self.settings.get('min_figure_width_points', 50)
        min_height = self.settings.get('min_figure_height_points', 50)

        for page_num in range(len(doc)):
            page = doc[page_num]

            try:
                # Simple approach: if page has drawings or images, capture generous areas
                visual_areas = self._simple_visual_detection(page, page_num)

                for vis_index, visual_area in enumerate(visual_areas):
                    bbox = visual_area['bbox']
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]

                    # Check minimum size requirements
                    if width >= min_width and height >= min_height:
                        # Final filter: check if this area actually contains visual content
                        if self._area_has_visual_content(page, bbox, visual_area['content_type']):
                            # Render visual area as image
                            visual_image = self._render_area_as_image(
                                page, bbox, page_num + 1, f"visual_{vis_index + 1}",
                                output_image_folder, "visual"
                            )

                            if visual_image:
                                # Add visual content information
                                visual_image['content_type'] = visual_area.get('content_type', 'diagram')
                                visual_image['confidence'] = visual_area.get('confidence', 0.8)
                                visual_images.append(visual_image)
                                logger.debug(f"Extracted visual content: {visual_image['filename']} - Page {page_num + 1}")
                        else:
                            logger.debug(f"Page {page_num + 1}: Filtered out text-only area")

            except Exception as e:
                logger.warning(f"Could not extract visual content from page {page_num + 1}: {e}")

        # Phase 2: Validate images against original text context
        validated_images = self._validate_images_against_text_context(doc, visual_images)

        logger.info(f"Successfully extracted {len(validated_images)} visual content images (validated against text context)")
        return validated_images

    def _detect_table_areas(self, page, min_columns, min_rows):
        """Detect table areas on a page using text alignment analysis"""
        table_areas = []

        try:
            blocks = page.get_text("dict")["blocks"]

            # Group text blocks by vertical position to find potential table rows
            text_lines = []
            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    line_text = ""
                    line_bbox = line["bbox"]
                    spans_info = []

                    for span in line["spans"]:
                        line_text += span["text"]
                        spans_info.append({
                            'text': span["text"],
                            'bbox': span["bbox"],
                            'x0': span["bbox"][0],
                            'x1': span["bbox"][2]
                        })

                    if line_text.strip():
                        text_lines.append({
                            'text': line_text.strip(),
                            'bbox': line_bbox,
                            'y0': line_bbox[1],
                            'y1': line_bbox[3],
                            'spans': spans_info
                        })

            # Sort lines by vertical position
            text_lines.sort(key=lambda x: x['y0'])

            # Group lines into potential table rows
            table_candidates = self._group_lines_into_tables(text_lines, min_columns, min_rows)

            for candidate in table_candidates:
                if self._validate_table_structure(candidate, min_columns, min_rows):
                    # ENHANCED: Filter out assessment text that looks like tables
                    if not self._is_assessment_text_misclassified_as_table(candidate, page):
                        table_areas.append(candidate)
                    else:
                        logger.debug(f"Filtered out assessment text misclassified as table: {candidate.get('sample_text', '')[:50]}...")

        except Exception as e:
            logger.warning(f"Error detecting table areas: {e}")

        return table_areas

    def _is_assessment_text_misclassified_as_table(self, table_candidate, page):
        """Check if a table candidate is actually assessment text that should not be extracted as an image"""
        try:
            # Get all text from the table area
            bbox = table_candidate['bbox']
            clip_rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            table_text = page.get_text(clip=clip_rect)

            if not table_text or len(table_text.strip()) < 20:
                return False

            text_lower = table_text.lower()

            # Strong indicators that this is assessment text, not a table
            assessment_patterns = [
                'first point', 'second point', 'third point', 'fourth point', 'fifth point',
                'first assessment', 'second assessment', 'third assessment',
                'point of assessment', 'assessment point', 'evaluation point',
                'first criterion', 'second criterion', 'third criterion',
                'first step', 'second step', 'third step',
                'first stage', 'second stage', 'third stage'
            ]

            # Count assessment pattern matches
            assessment_matches = sum(1 for pattern in assessment_patterns if pattern in text_lower)

            if assessment_matches >= 1:  # Even one match is strong indicator
                logger.debug(f"Assessment text detected in table candidate: {assessment_matches} patterns found")
                return True

            # Check for numbered list patterns that are common in assessment text
            lines = table_text.split('\n')
            numbered_assessment_lines = 0

            for line in lines:
                line_stripped = line.strip().lower()
                if line_stripped:
                    # Look for assessment-style numbered items
                    if (re.match(r'^\d+[\.\)]\s+.*(?:point|assessment|criterion|step|stage|evaluation)', line_stripped) or
                        re.match(r'^(?:first|second|third|fourth|fifth).*(?:point|assessment|criterion)', line_stripped)):
                        numbered_assessment_lines += 1

            # If significant portion of lines are assessment-style, it's likely assessment text
            if len(lines) > 2 and numbered_assessment_lines / len(lines) > 0.3:
                logger.debug(f"Assessment-style numbered list detected: {numbered_assessment_lines}/{len(lines)} lines")
                return True

            # Check for academic/evaluation keywords that suggest this is text content
            evaluation_keywords = [
                'evaluation', 'assessment', 'criteria', 'methodology', 'analysis',
                'conclusion', 'findings', 'results', 'discussion', 'recommendation',
                'objective', 'goal', 'purpose', 'approach', 'strategy'
            ]

            evaluation_count = sum(1 for keyword in evaluation_keywords if keyword in text_lower)

            # If many evaluation keywords and looks like structured text, likely assessment content
            if evaluation_count >= 3 and len(table_text.split()) > 30:
                logger.debug(f"Academic evaluation text detected: {evaluation_count} keywords found")
                return True

            # Check if this looks more like paragraph text than tabular data
            words = table_text.split()
            if len(words) > 50:  # Substantial text content
                # Calculate text characteristics
                avg_word_length = sum(len(word) for word in words) / len(words)
                sentence_count = table_text.count('.') + table_text.count('!') + table_text.count('?')

                # If it has characteristics of prose text, likely not a table
                if (avg_word_length > 4 and  # Longer words suggest prose
                    sentence_count > 2 and   # Multiple sentences suggest prose
                    len(words) / max(1, sentence_count) > 10):  # Long sentences suggest prose
                    logger.debug(f"Prose-like text characteristics detected in table candidate")
                    return True

            return False

        except Exception as e:
            logger.debug(f"Error checking assessment text classification: {e}")
            return False  # Conservative: assume it's a valid table

    def _detect_equation_areas(self, page, detect_math_symbols):
        """Detect equation areas on a page using mathematical symbol analysis"""
        equation_areas = []

        # Mathematical symbols and patterns to look for
        math_symbols = ['∑', '∫', '∂', '∆', '∇', '∞', '±', '≤', '≥', '≠', '≈', '∝', '∈', '∉', '⊂', '⊃', '∪', '∩']
        math_patterns = [
            r'[a-zA-Z]\s*[=]\s*[a-zA-Z0-9\+\-\*/\(\)]+',  # Basic equations
            r'\d+\s*[+\-*/]\s*\d+\s*[=]\s*\d+',  # Arithmetic
            r'[a-zA-Z]\^[0-9]+',  # Exponents
            r'[a-zA-Z]_[0-9]+',   # Subscripts
            r'\([^)]*\)\s*[=]\s*',  # Function definitions
            r'\\[a-zA-Z]+\{[^}]*\}',  # LaTeX commands
        ]

        try:
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" not in block:
                    continue

                block_text = ""
                block_bbox = block["bbox"]
                has_math_content = False

                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        span_text = span["text"]
                        line_text += span_text

                        # Check for mathematical symbols
                        if detect_math_symbols:
                            if any(symbol in span_text for symbol in math_symbols):
                                has_math_content = True

                    block_text += line_text + " "

                # Check for mathematical patterns
                if not has_math_content and detect_math_symbols:
                    for pattern in math_patterns:
                        if re.search(pattern, block_text):
                            has_math_content = True
                            break

                # If this block contains mathematical content, consider it an equation area
                if has_math_content:
                    equation_areas.append({
                        'bbox': block_bbox,
                        'text': block_text.strip(),
                        'confidence': 0.8 if any(symbol in block_text for symbol in math_symbols) else 0.6
                    })

        except Exception as e:
            logger.warning(f"Error detecting equation areas: {e}")

        return equation_areas

    def _simple_visual_detection(self, page, page_num):
        """Simple and effective visual detection - identify areas to exclude from translation"""
        visual_areas = []

        try:
            page_rect = page.rect

            # Method 1: Check for vector drawings
            drawings = page.get_drawings()
            if len(drawings) >= 3:  # Lower threshold since we're excluding these areas
                # Create a large area covering most of the page
                padding = 30
                bbox = [
                    padding,
                    padding,
                    page_rect.width - padding,
                    page_rect.height - padding
                ]

                visual_areas.append({
                    'bbox': bbox,
                    'content_type': 'page_with_drawings',
                    'exclude_from_translation': True,  # Flag to exclude from translation
                    'confidence': 0.9
                })
                logger.debug(f"Page {page_num + 1}: Found {len(drawings)} drawings, excluding from translation")

            # Method 2: Check for raster images
            images = page.get_images(full=True)
            if images:
                for img in images:
                    try:
                        xref = img[0]
                        img_rects = page.get_image_rects(xref)

                        if img_rects:
                            for rect in img_rects:
                                # Create generous area around image
                                padding = 150  # Very generous padding
                                bbox = [
                                    max(0, rect.x0 - padding),
                                    max(0, rect.y0 - padding),
                                    min(page_rect.width, rect.x1 + padding),
                                    min(page_rect.height, rect.y1 + padding)
                                ]

                                visual_areas.append({
                                    'bbox': bbox,
                                    'content_type': 'image_area',
                                    'exclude_from_translation': True,  # Flag to exclude from translation
                                    'confidence': 0.95
                                })
                                logger.debug(f"Page {page_num + 1}: Found raster image, excluding from translation")
                    except:
                        continue

            # Method 3: Check for pages with low text density (might have visual content)
            if not visual_areas:  # Only if we haven't found anything else
                text_blocks = page.get_text("dict")["blocks"]
                text_blocks = [b for b in text_blocks if "lines" in b]

                if text_blocks:
                    total_text_area = 0
                    for block in text_blocks:
                        bbox = block["bbox"]
                        block_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        total_text_area += block_area

                    page_area = page_rect.width * page_rect.height
                    text_coverage = total_text_area / page_area

                    # If very low text coverage, might have visual content
                    if text_coverage < 0.2:  # Less than 20% text
                        padding = 50
                        bbox = [
                            padding,
                            padding,
                            page_rect.width - padding,
                            page_rect.height - padding
                        ]

                        visual_areas.append({
                            'bbox': bbox,
                            'content_type': 'sparse_text_page',
                            'exclude_from_translation': True,  # Flag to exclude from translation
                            'confidence': 0.8
                        })
                        logger.debug(f"Page {page_num + 1}: Low text coverage ({text_coverage:.2f}), excluding from translation")

        except Exception as e:
            logger.warning(f"Error in simple visual detection: {e}")

        return visual_areas

    def _has_substantial_visual_content(self, page, drawings):
        """Check if drawings represent substantial visual content, not just text formatting"""
        try:
            # Get all text from the page
            page_text = page.get_text()

            # If page is mostly text with common patterns, likely just formatted text
            if self._is_likely_formatted_text_page(page_text):
                return False

            # Analyze drawing complexity
            complex_drawings = 0
            total_drawing_area = 0

            for drawing in drawings:
                try:
                    if hasattr(drawing, 'rect'):
                        rect = drawing.rect
                        area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
                        total_drawing_area += area

                        # Consider drawings with substantial area as complex
                        if area > 1000:  # Minimum area threshold
                            complex_drawings += 1
                except:
                    continue

            # Need multiple complex drawings for substantial visual content
            if complex_drawings >= 3:
                return True

            # Check if drawings cover significant portion of page
            page_area = page.rect.width * page.rect.height
            drawing_coverage = total_drawing_area / page_area if page_area > 0 else 0

            # If drawings cover substantial area, likely visual content
            if drawing_coverage > 0.1:  # More than 10% of page
                return True

            return False

        except Exception as e:
            logger.debug(f"Error checking substantial visual content: {e}")
            return False  # Conservative: assume it's text formatting

    def _is_likely_formatted_text_page(self, page_text):
        """Check if page appears to be formatted text rather than visual content"""
        if not page_text or len(page_text.strip()) < 50:
            return False

        text_lower = page_text.lower()

        # Strong indicators of formatted text content
        formatted_text_indicators = [
            # Assessment/evaluation content
            'assessment', 'evaluation', 'point', 'criteria', 'first', 'second', 'third',
            'step', 'phase', 'stage', 'level', 'grade', 'score', 'rating',
            # List/enumeration patterns
            'following', 'include', 'such as', 'for example', 'namely',
            # Academic/formal text
            'according to', 'research', 'study', 'analysis', 'conclusion',
            'introduction', 'methodology', 'results', 'discussion'
        ]

        indicator_count = sum(1 for indicator in formatted_text_indicators if indicator in text_lower)

        # If we have multiple text formatting indicators, likely formatted text
        if indicator_count >= 3:
            return True

        # Check for numbered/bulleted lists (common in formatted text)
        lines = page_text.split('\n')
        numbered_lines = sum(1 for line in lines if re.match(r'^\s*\d+[\.\)]\s+', line.strip()))
        bulleted_lines = sum(1 for line in lines if re.match(r'^\s*[•\-\*]\s+', line.strip()))

        if len(lines) > 5 and (numbered_lines + bulleted_lines) / len(lines) > 0.3:
            return True

        # Check for bold/emphasis patterns (common in formatted text)
        if len(page_text.split()) > 20:
            # Look for short emphasized phrases (typical of headings/bold text)
            words = page_text.split()
            short_phrases = sum(1 for i in range(len(words)-1) if len(words[i]) < 15 and words[i].istitle())

            if short_phrases / len(words) > 0.2:  # Many title-case words
                return True

        return False

    def _area_has_visual_content(self, page, bbox, content_type):
        """Final check to ensure area actually contains visual content, not just text"""
        try:
            # Always keep areas that were detected based on actual drawings or images
            if content_type in ['page_with_drawings', 'image_area']:
                return True

            # For sparse text pages, do additional validation
            if content_type == 'sparse_text_page':
                # Get text in this area
                clip_rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                text_in_area = page.get_text(clip=clip_rect)

                # Filter out obvious text-only pages
                if self._is_mostly_plain_text(text_in_area):
                    return False

                # Check if area has any drawings within it
                drawings = page.get_drawings()
                if drawings:
                    drawings_in_area = 0
                    for drawing in drawings:
                        try:
                            if hasattr(drawing, 'rect'):
                                draw_rect = drawing.rect
                                # Check if drawing overlaps with our area
                                if (draw_rect.x0 >= bbox[0] and draw_rect.x1 <= bbox[2] and
                                    draw_rect.y0 >= bbox[1] and draw_rect.y1 <= bbox[3]):
                                    drawings_in_area += 1
                        except:
                            continue

                    # If we have drawings in this area, keep it
                    if drawings_in_area >= 2:
                        return True

                # If no drawings and mostly text, filter it out
                return False

            return True

        except Exception as e:
            logger.warning(f"Error checking area visual content: {e}")
            return True  # When in doubt, keep it

    def _is_mostly_plain_text(self, text):
        """Check if text is mostly plain paragraphs (not diagrams) - ENHANCED for better text detection"""
        if not text or len(text) < 50:
            return True

        text_lower = text.lower()

        # ENHANCED: Check for assessment/evaluation content patterns
        assessment_indicators = [
            'assessment', 'evaluation', 'point', 'criteria', 'first', 'second', 'third',
            'step', 'phase', 'stage', 'level', 'grade', 'score', 'rating',
            'following', 'include', 'such as', 'for example', 'namely'
        ]
        assessment_count = sum(1 for indicator in assessment_indicators if indicator in text_lower)

        if assessment_count >= 2:  # Strong indicator of formatted text content
            return True

        # Check for TOC/bibliography indicators (more aggressive)
        toc_indicators = ['contents', 'chapter', 'bibliography', 'references', 'index', 'page', 'introduction', 'preface']
        toc_count = sum(1 for indicator in toc_indicators if indicator in text_lower)

        if toc_count >= 1:  # Even one indicator is enough
            return True

        # ENHANCED: Check for numbered/bulleted lists (common in formatted text)
        lines = text.split('\n')
        if len(lines) > 3:
            # Count lines with numbers (page numbers, chapter numbers)
            numbered_lines = sum(1 for line in lines if re.search(r'\d+', line.strip()))
            bulleted_lines = sum(1 for line in lines if re.match(r'^\s*[•\-\*]\s+', line.strip()))
            list_lines = numbered_lines + bulleted_lines

            if list_lines / len(lines) > 0.3:  # 30% of lines are list items
                return True

            if numbered_lines / len(lines) > 0.4:  # Lower threshold for numbered content
                return True

        # Check for very regular paragraph text (more aggressive)
        sentences = text.split('.')
        if len(sentences) > 5:  # Lower threshold
            avg_sentence_length = sum(len(s.strip()) for s in sentences if s.strip()) / max(1, len([s for s in sentences if s.strip()]))
            # If sentences are very uniform in length, likely regular text
            if 30 <= avg_sentence_length <= 300:  # Broader range
                return True

        # ENHANCED: Check for academic/formal text patterns
        academic_indicators = [
            'according to', 'research', 'study', 'analysis', 'conclusion',
            'methodology', 'results', 'discussion', 'however', 'therefore',
            'furthermore', 'moreover', 'consequently', 'in addition'
        ]
        academic_count = sum(1 for indicator in academic_indicators if indicator in text_lower)

        if academic_count >= 2:  # Academic text patterns
            return True

        # Check for common text patterns
        words = text.split()
        if len(words) > 50:
            # If mostly common English words, likely regular text
            common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            common_count = sum(1 for word in words if word.lower() in common_words)
            if common_count / len(words) > 0.15:  # High ratio of common words
                return True

        # ENHANCED: Check for title case patterns (common in formatted headings)
        if len(words) > 10:
            title_case_words = sum(1 for word in words if word.istitle() and len(word) > 2)
            if title_case_words / len(words) > 0.3:  # Many title case words
                return True

        return False

    def _validate_images_against_text_context(self, doc, visual_images):
        """Phase 2: Validate that extracted images have meaningful placement in original text"""
        validated_images = []

        for image in visual_images:
            page_num = image['page_num']

            # Get the original text from this page
            page = doc[page_num - 1]  # Convert to 0-based index
            page_text = page.get_text()

            # Check if this page has meaningful text context for image placement
            if self._page_has_image_placement_context(page_text, page_num):
                validated_images.append(image)
                logger.debug(f"✅ Validated: {image['filename']} - Page {page_num} has image placement context")
            else:
                logger.debug(f"❌ Filtered: {image['filename']} - Page {page_num} lacks image placement context")

        logger.info(f"Image validation: {len(visual_images)} → {len(validated_images)} (filtered {len(visual_images) - len(validated_images)} text-only images)")

        # Phase 3: Remove duplicate images (same page, similar content)
        deduplicated_images = self._remove_duplicate_images(validated_images)
        logger.info(f"Duplicate removal: {len(validated_images)} → {len(deduplicated_images)} (removed {len(validated_images) - len(deduplicated_images)} duplicates)")

        return deduplicated_images

    def _remove_duplicate_images(self, images):
        """Remove competing extractions from the same page, keeping the best quality one"""
        if not images:
            return images

        # Group images by page
        images_by_page = defaultdict(list)
        for img in images:
            images_by_page[img['page_num']].append(img)

        filtered_images = []

        for page_num, page_images in images_by_page.items():
            if len(page_images) <= 1:
                # No competing extractions on this page
                filtered_images.extend(page_images)
                continue

            # For pages with multiple extractions, apply intelligent filtering
            best_images = self._select_best_page_extractions(page_images, page_num)
            filtered_images.extend(best_images)

            if len(page_images) > len(best_images):
                logger.debug(f"Page {page_num}: Kept {len(best_images)} best extractions out of {len(page_images)}")

        return filtered_images

    def _select_best_page_extractions(self, page_images, page_num):
        """Select the best extractions from a page with multiple candidates"""
        if len(page_images) <= 1:
            return page_images

        # Separate by extraction type
        by_type = defaultdict(list)
        for img in page_images:
            extraction_type = self._get_extraction_type(img['filename'])
            by_type[extraction_type].append(img)

        best_images = []

        # For each type, keep only the best one(s)
        for extraction_type, type_images in by_type.items():
            if extraction_type == 'visual':
                # For visual content, we might have overlapping extractions
                # Keep the one with the best quality indicators
                best_visual = self._select_best_visual_extraction(type_images)
                if best_visual:
                    best_images.append(best_visual)
            else:
                # For tables, equations, etc., keep all if they're different enough
                # or the best one if they're too similar
                filtered_type_images = self._filter_similar_extractions(type_images)
                best_images.extend(filtered_type_images)

        return best_images

    def _get_extraction_type(self, filename):
        """Determine extraction type from filename"""
        if '_visual_' in filename:
            return 'visual'
        elif '_table_' in filename:
            return 'table'
        elif '_equation_' in filename:
            return 'equation'
        elif '_img_' in filename:
            return 'image'
        else:
            return 'unknown'

    def _select_best_visual_extraction(self, visual_images):
        """Select the best visual extraction from competing candidates"""
        if len(visual_images) <= 1:
            return visual_images[0] if visual_images else None

        # Score each image based on quality indicators
        scored_images = []

        for img in visual_images:
            score = self._calculate_image_quality_score(img)
            scored_images.append((score, img))

        # Sort by score (highest first)
        scored_images.sort(key=lambda x: x[0], reverse=True)

        # Return the best one
        best_score, best_img = scored_images[0]

        logger.debug(f"Selected best visual extraction: {best_img['filename']} (score: {best_score:.2f})")
        return best_img

    def _calculate_image_quality_score(self, img):
        """Calculate a quality score for an image based on various factors"""
        score = 0.0

        try:
            # Factor 1: File size (larger usually means more content)
            if os.path.exists(img['filepath']):
                file_size = os.path.getsize(img['filepath'])
                # Normalize file size (1MB = 1.0 point)
                size_score = min(file_size / (1024 * 1024), 5.0)  # Cap at 5 points
                score += size_score

            # Factor 2: Bounding box area (larger area usually better)
            if 'bbox' in img:
                bbox = img['bbox']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                # Normalize area (100,000 square points = 1.0 point)
                area_score = min(area / 100000, 3.0)  # Cap at 3 points
                score += area_score

            # Factor 3: Content type confidence
            confidence = img.get('confidence', 0.5)
            score += confidence * 2.0  # Up to 2 points for confidence

            # Factor 4: Filename indicators (prefer comprehensive extractions)
            filename = img['filename'].lower()
            if 'visual' in filename:
                score += 1.0  # Visual extractions are usually more comprehensive

            # Factor 5: Penalize very small files (likely text-only)
            if os.path.exists(img['filepath']):
                file_size = os.path.getsize(img['filepath'])
                if file_size < 50000:  # Less than 50KB
                    score -= 2.0  # Penalty for very small files

        except Exception as e:
            logger.debug(f"Error calculating quality score for {img['filename']}: {e}")
            score = 1.0  # Default score

        return max(score, 0.1)  # Minimum score of 0.1

    def _filter_similar_extractions(self, type_images):
        """Filter out very similar extractions of the same type"""
        if len(type_images) <= 1:
            return type_images

        # For now, keep all different type extractions
        # In the future, we could add more sophisticated similarity detection
        return type_images

    def _are_images_similar(self, img1, img2):
        """Check if two images are competing extractions from the same content area"""
        try:
            # Check if they're the same extraction type
            type1 = self._get_extraction_type(img1['filename'])
            type2 = self._get_extraction_type(img2['filename'])

            # Only compare images of the same type
            if type1 != type2:
                return False

            # For visual extractions, check for significant overlap
            if type1 == 'visual' and 'bbox' in img1 and 'bbox' in img2:
                bbox1 = img1['bbox']
                bbox2 = img2['bbox']

                # Calculate overlap
                overlap_area = self._calculate_bbox_overlap(bbox1, bbox2)
                area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

                # If there's significant overlap (>20%), they're competing extractions
                min_area = min(area1, area2)
                if min_area > 0 and overlap_area > 0.2 * min_area:
                    return True

            # Check file size similarity for potential competing extractions
            if os.path.exists(img1['filepath']) and os.path.exists(img2['filepath']):
                size1 = os.path.getsize(img1['filepath'])
                size2 = os.path.getsize(img2['filepath'])

                # If one is much larger, they might be competing versions
                # (good extraction vs text-only extraction)
                size_ratio = max(size1, size2) / max(min(size1, size2), 1)
                if size_ratio > 3:  # One is 3x larger - likely competing extractions
                    return True

                # Check if they have similar dimensions
                if 'width' in img1 and 'height' in img1 and 'width' in img2 and 'height' in img2:
                    width_ratio = max(img1['width'], img2['width']) / max(min(img1['width'], img2['width']), 1)
                    height_ratio = max(img1['height'], img2['height']) / max(min(img1['height'], img2['height']), 1)
                    if width_ratio < 1.2 and height_ratio < 1.2:  # Very similar dimensions
                        return True

        except Exception as e:
            logger.debug(f"Error checking image similarity: {e}")

        return False

    def _calculate_bbox_overlap(self, bbox1, bbox2):
        """Calculate the overlap area between two bounding boxes"""
        x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
        y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
        return x_overlap * y_overlap

    def _is_better_image(self, img1, img2):
        """Determine which image is better quality using comprehensive scoring"""
        score1 = self._calculate_image_quality_score(img1)
        score2 = self._calculate_image_quality_score(img2)
        return score1 > score2

    def _page_has_image_placement_context(self, page_text, page_num):
        """Check if a page has meaningful context where an image could be placed"""
        if not page_text or len(page_text.strip()) < 100:
            return False  # Too little text to have meaningful context

        text_lower = page_text.lower()

        # Strong indicators that this page should have images
        strong_image_indicators = [
            'figure', 'diagram', 'chart', 'graph', 'illustration', 'image', 'picture',
            'schema', 'model', 'flowchart', 'plot', 'drawing', 'sketch'
        ]

        # Look for explicit figure references
        figure_patterns = [
            r'figure\s+\d+',
            r'fig\.\s*\d+',
            r'diagram\s+\d+',
            r'chart\s+\d+',
            r'see\s+figure',
            r'shown\s+in\s+figure',
            r'as\s+illustrated',
            r'as\s+shown'
        ]

        # Count strong indicators
        strong_count = sum(1 for indicator in strong_image_indicators if indicator in text_lower)

        # Count figure reference patterns
        pattern_matches = sum(1 for pattern in figure_patterns if re.search(pattern, text_lower))

        # If we have explicit figure references, definitely keep
        if pattern_matches >= 1:
            return True

        # If we have multiple strong indicators, likely has visual content
        if strong_count >= 2:
            return True

        # Check for scientific/technical content that often has diagrams
        technical_indicators = [
            'equation', 'formula', 'algorithm', 'method', 'process', 'system',
            'structure', 'analysis', 'result', 'data', 'experiment', 'study'
        ]

        technical_count = sum(1 for indicator in technical_indicators if indicator in text_lower)

        # Technical content with some visual indicators
        if technical_count >= 3 and strong_count >= 1:
            return True

        # Filter out obvious text-only pages (enhanced)
        text_only_indicators = [
            'table of contents', 'bibliography', 'references', 'index',
            'acknowledgments', 'preface', 'introduction', 'conclusion',
            'abstract', 'summary', 'appendix', 'glossary', 'notes'
        ]

        text_only_count = sum(1 for indicator in text_only_indicators if indicator in text_lower)

        if text_only_count >= 1:
            return False  # Definitely text-only

        # Enhanced text-only detection: check for paragraph-heavy content
        # If the page is mostly continuous prose, it's likely text-only
        if self._is_mostly_continuous_prose(page_text):
            return False

        # Check text structure - if very paragraph-heavy, likely text-only
        sentences = page_text.split('.')
        if len(sentences) > 20:  # Many sentences
            avg_sentence_length = sum(len(s.strip()) for s in sentences if s.strip()) / max(1, len([s for s in sentences if s.strip()]))
            if 40 <= avg_sentence_length <= 150:  # Regular prose
                return False

        # Default: if we have some content and no clear text-only indicators, allow it
        return len(page_text.strip()) >= 200  # Reasonable amount of content

    def _is_mostly_continuous_prose(self, page_text):
        """Check if page text is mostly continuous prose (indicating text-only content)"""
        if not page_text or len(page_text.strip()) < 150:  # Reduced threshold
            return False

        # Split into sentences and analyze structure
        sentences = [s.strip() for s in page_text.split('.') if s.strip()]
        if len(sentences) < 3:  # Reduced threshold
            return False

        # Calculate average sentence length
        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)

        # Check for typical prose characteristics
        long_sentences = sum(1 for s in sentences if len(s) > 40)  # Reduced threshold
        long_sentence_ratio = long_sentences / len(sentences)

        # Check for paragraph structure (multiple line breaks)
        paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)

        # More lenient criteria for prose detection
        if (avg_sentence_length > 40 and  # Reduced from 60
            long_sentence_ratio > 0.3 and  # Reduced from 0.4
            paragraph_count >= 2):  # Reduced from 3
            return True

        # Check for academic/technical writing patterns
        academic_indicators = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'in addition', 'on the other hand', 'for example', 'such as',
            'according to', 'research shows', 'studies indicate', 'demonstrates',
            'analysis', 'methodology', 'implementation', 'significant'
        ]

        text_lower = page_text.lower()
        academic_count = sum(1 for indicator in academic_indicators if indicator in text_lower)

        # More lenient academic language detection
        if academic_count >= 2 and len(sentences) > 5:  # Reduced thresholds
            return True

        # Additional check: if text has many connecting words, it's likely prose
        connecting_words = [
            'and', 'but', 'or', 'because', 'since', 'while', 'although',
            'when', 'where', 'which', 'that', 'this', 'these', 'those'
        ]

        connecting_count = sum(1 for word in connecting_words if word in text_lower)
        word_count = len(page_text.split())

        if word_count > 50 and connecting_count / word_count > 0.1:  # 10% connecting words
            return True

        return False

    def _detect_complex_layout_areas(self, page):
        """Detect areas with complex text layouts that are actually diagrams (not regular text)"""
        layout_areas = []

        try:
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" not in block:
                    continue

                block_bbox = block["bbox"]
                block_text = ""

                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"]
                    block_text += "\n"

                # More strict criteria for text diagrams
                if self._is_text_diagram(block_text):
                    # Add padding around the text block
                    padding = 30
                    bbox = [
                        max(0, block_bbox[0] - padding),
                        max(0, block_bbox[1] - padding),
                        min(page.rect.width, block_bbox[2] + padding),
                        min(page.rect.height, block_bbox[3] + padding)
                    ]

                    layout_areas.append({
                        'bbox': bbox,
                        'content_type': 'text_diagram',
                        'confidence': 0.7
                    })

        except Exception as e:
            logger.warning(f"Error detecting complex layout areas: {e}")

        return layout_areas

    def _is_text_diagram(self, text):
        """Determine if text block is actually a diagram (more lenient criteria)"""
        if not text or len(text) < 30:  # Reduced minimum length
            return False

        # Look for diagram-like patterns
        diagram_chars = ['|', '-', '+', '/', '\\', '→', '←', '↑', '↓', '○', '●', '□', '■', '△', '▲', '◆', '◇', '=', '*']
        diagram_char_count = sum(text.count(char) for char in diagram_chars)

        # Look for structured spacing (ASCII art patterns)
        lines = text.split('\n')
        if len(lines) < 2:  # Reduced minimum lines
            return False

        # Count lines with significant spacing patterns
        structured_lines = 0
        for line in lines:
            # Lines with multiple spaces and short content (typical of diagrams)
            if line.count(' ') >= 3 and 2 <= len(line.strip()) <= 30:  # More lenient
                structured_lines += 1

        # Calculate ratios
        diagram_char_ratio = diagram_char_count / len(text)
        structured_line_ratio = structured_lines / len(lines)

        # More lenient criteria
        if diagram_char_count >= 8 and diagram_char_ratio >= 0.02:  # Reduced thresholds
            return True

        # OR high structured line ratio (ASCII art)
        if structured_line_ratio >= 0.4 and structured_lines >= 3:  # More lenient
            return True

        # Check for table-like structures (but not regular paragraphs)
        if self._looks_like_ascii_table(lines):
            return True

        # Check for mathematical/chemical formulas with special characters
        if self._looks_like_formula_diagram(text):
            return True

        return False

    def _looks_like_formula_diagram(self, text):
        """Check if text contains mathematical or chemical formula diagrams"""
        formula_chars = ['→', '←', '↑', '↓', '⇌', '≡', '≈', '∆', '∇', '∑', '∫', '∂']
        formula_count = sum(text.count(char) for char in formula_chars)

        # Chemical/mathematical notation patterns
        chem_patterns = [
            r'[A-Z][a-z]?\d*',  # Chemical formulas like H2O, CO2
            r'\d+\s*[+\-]\s*\d+',  # Mathematical expressions
            r'[A-Z]\s*[=]\s*[A-Z]',  # Simple equations
        ]

        pattern_matches = sum(1 for pattern in chem_patterns if re.search(pattern, text))

        return formula_count >= 2 or pattern_matches >= 2

    def _looks_like_ascii_table(self, lines):
        """Check if lines form an ASCII table structure"""
        if len(lines) < 3:
            return False

        # Look for horizontal separators
        separator_lines = sum(1 for line in lines if line.count('-') >= 5 or line.count('=') >= 5)

        # Look for vertical separators
        vertical_lines = sum(1 for line in lines if line.count('|') >= 2)

        # Need both horizontal and vertical structure
        return separator_lines >= 1 and vertical_lines >= 2

    def _detect_visual_space_areas(self, page):
        """Detect areas with sparse text that might contain visual content, but filter out regular text pages"""
        space_areas = []

        try:
            # Get all text blocks
            blocks = page.get_text("dict")["blocks"]
            text_blocks = [b for b in blocks if "lines" in b]

            if not text_blocks:
                return space_areas

            page_rect = page.rect
            page_area = page_rect.width * page_rect.height

            # Calculate text coverage and analyze content
            total_text_area = 0
            all_text = ""

            for block in text_blocks:
                bbox = block["bbox"]
                block_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                total_text_area += block_area

                # Collect all text for analysis
                for line in block["lines"]:
                    for span in line["spans"]:
                        all_text += span["text"] + " "

            text_coverage = total_text_area / page_area

            # Only consider pages with very low text coverage AND signs of visual content
            if text_coverage < 0.15:  # Very sparse text (less than 15%)
                # Additional checks to ensure this isn't just a regular text page
                if self._has_visual_indicators(all_text, page):
                    # Create a large area covering most of the page
                    padding = 50
                    bbox = [
                        padding,
                        padding,
                        page_rect.width - padding,
                        page_rect.height - padding
                    ]

                    space_areas.append({
                        'bbox': bbox,
                        'content_type': 'sparse_layout',
                        'confidence': 0.3  # Lower confidence for sparse areas
                    })

        except Exception as e:
            logger.warning(f"Error detecting visual space areas: {e}")

        return space_areas

    def _has_visual_indicators(self, text, page):
        """Check if a page has indicators of visual content beyond just sparse text"""
        try:
            # First, filter out TOC/bibliography/index pages
            if self._is_toc_or_bibliography_page(text):
                return False

            # Check for vector drawings (be less restrictive)
            drawings = page.get_drawings()
            if len(drawings) >= 5:  # Reduced from 10
                return True

            # Check for embedded images
            images = page.get_images(full=True)
            if len(images) >= 1:
                return True

            # Check for visual-related keywords in context
            visual_keywords = ['figure', 'diagram', 'chart', 'graph', 'illustration', 'image', 'picture', 'schema', 'model']
            text_lower = text.lower()

            # Look for figure references (Figure X.X, etc.)
            figure_patterns = [r'figure\s+\d+', r'fig\.\s*\d+', r'diagram\s+\d+', r'chart\s+\d+']
            has_figure_refs = any(re.search(pattern, text_lower) for pattern in figure_patterns)

            if has_figure_refs:
                return True

            # Check for diagram-like characters (be more lenient)
            diagram_chars = ['|', '-', '+', '/', '\\', '→', '←', '↑', '↓', '○', '●', '□', '■', '△', '▲']
            diagram_char_count = sum(text.count(char) for char in diagram_chars)

            # Lower threshold for diagram characters
            if diagram_char_count >= 8:
                return True

            return False

        except Exception as e:
            logger.warning(f"Error checking visual indicators: {e}")
            return False

    def _is_toc_or_bibliography_page(self, text):
        """Check if a page is table of contents, bibliography, or index"""
        text_lower = text.lower()

        # Strong TOC indicators
        strong_toc_indicators = [
            'table of contents', 'contents', 'chapter',
            'bibliography', 'references', 'index', 'appendix'
        ]

        # Weak TOC indicators (need multiple)
        weak_toc_indicators = [
            'introduction', 'preface', 'acknowledgments', 'about the author',
            'part i', 'part ii', 'part iii', 'section'
        ]

        # Check for strong indicators (any one is enough)
        strong_count = sum(1 for indicator in strong_toc_indicators if indicator in text_lower)
        if strong_count >= 1:
            return True

        # Check for multiple weak indicators
        weak_count = sum(1 for indicator in weak_toc_indicators if indicator in text_lower)
        if weak_count >= 2:
            return True

        # Check for page number patterns (common in TOC)
        page_number_patterns = [
            r'\.\.\.\s*\d+',  # dots followed by page numbers
            r'\d+\s*\.\.\.',  # page numbers followed by dots
            r'^\s*\d+\s*$',   # standalone page numbers
            r'\d+\s*\n',      # numbers at end of lines
        ]

        page_number_matches = sum(1 for pattern in page_number_patterns if re.search(pattern, text_lower, re.MULTILINE))

        # If many page number patterns, likely TOC
        if page_number_matches >= 2:
            return True

        # Check for repetitive short lines (typical of TOC structure)
        lines = text.split('\n')
        short_lines = sum(1 for line in lines if 3 <= len(line.strip()) <= 60)

        # More aggressive TOC detection based on structure
        if len(lines) > 5 and short_lines / len(lines) > 0.6:
            return True

        # Check for numbered sections/chapters
        numbered_sections = sum(1 for line in lines if re.match(r'^\s*\d+\.?\s+[A-Z]', line.strip()))
        if numbered_sections >= 3:
            return True

        return False



    def _bboxes_overlap(self, bbox1, bbox2, tolerance=10):
        """Check if two bounding boxes overlap (with tolerance)"""
        return not (bbox1[2] + tolerance < bbox2[0] or  # bbox1 is to the left of bbox2
                   bbox2[2] + tolerance < bbox1[0] or   # bbox2 is to the left of bbox1
                   bbox1[3] + tolerance < bbox2[1] or   # bbox1 is above bbox2
                   bbox2[3] + tolerance < bbox1[1])     # bbox2 is above bbox1

    def _combine_bboxes(self, bbox1, bbox2):
        """Combine two bounding boxes into one that encompasses both"""
        return [
            min(bbox1[0], bbox2[0]),  # min x
            min(bbox1[1], bbox2[1]),  # min y
            max(bbox1[2], bbox2[2]),  # max x
            max(bbox1[3], bbox2[3])   # max y
        ]

    def _group_lines_into_tables(self, text_lines, min_columns, min_rows):
        """Group text lines into potential table structures"""
        table_candidates = []

        if len(text_lines) < min_rows:
            return table_candidates

        # Look for groups of lines with similar column structure
        i = 0
        while i < len(text_lines) - min_rows + 1:
            potential_table = []
            current_line = text_lines[i]

            # Check if this line could be a table header (has multiple columns)
            if len(current_line['spans']) >= min_columns:
                potential_table.append(current_line)

                # Look for subsequent lines with similar column structure
                j = i + 1
                while j < len(text_lines) and len(potential_table) < 20:  # Max 20 rows per table
                    next_line = text_lines[j]

                    # Check if lines are close vertically (part of same table)
                    vertical_gap = next_line['y0'] - current_line['y1']
                    if vertical_gap > 50:  # Too far apart
                        break

                    # Check if column structure is similar
                    if self._lines_have_similar_columns(current_line, next_line):
                        potential_table.append(next_line)
                        current_line = next_line
                    else:
                        break

                    j += 1

                # If we found enough rows, consider it a table
                if len(potential_table) >= min_rows:
                    table_bbox = self._calculate_table_bbox(potential_table)
                    table_candidates.append({
                        'bbox': table_bbox,
                        'rows': potential_table,
                        'row_count': len(potential_table),
                        'column_count': max(len(row['spans']) for row in potential_table)
                    })
                    i = j  # Skip processed lines
                else:
                    i += 1
            else:
                i += 1

        return table_candidates

    def _lines_have_similar_columns(self, line1, line2, tolerance=20):
        """Check if two lines have similar column structure"""
        spans1 = line1['spans']
        spans2 = line2['spans']

        # Must have similar number of columns
        if abs(len(spans1) - len(spans2)) > 1:
            return False

        # Check if column positions are similar
        min_spans = min(len(spans1), len(spans2))
        for i in range(min_spans):
            x1_start = spans1[i]['x0']
            x2_start = spans2[i]['x0']

            if abs(x1_start - x2_start) > tolerance:
                return False

        return True

    def _calculate_table_bbox(self, table_rows):
        """Calculate bounding box for a table"""
        if not table_rows:
            return [0, 0, 0, 0]

        min_x = min(row['bbox'][0] for row in table_rows)
        min_y = min(row['bbox'][1] for row in table_rows)
        max_x = max(row['bbox'][2] for row in table_rows)
        max_y = max(row['bbox'][3] for row in table_rows)

        return [min_x, min_y, max_x, max_y]

    def _validate_table_structure(self, table_candidate, min_columns, min_rows):
        """Validate that a table candidate is actually a table"""
        if table_candidate['row_count'] < min_rows:
            return False

        if table_candidate['column_count'] < min_columns:
            return False

        # Check that most rows have similar column counts
        row_column_counts = [len(row['spans']) for row in table_candidate['rows']]
        most_common_count = max(set(row_column_counts), key=row_column_counts.count)
        consistent_rows = sum(1 for count in row_column_counts if abs(count - most_common_count) <= 1)

        # At least 70% of rows should have consistent column structure
        consistency_ratio = consistent_rows / len(row_column_counts)
        return consistency_ratio >= 0.7

    def _render_area_as_image(self, page, bbox, page_num, area_id, output_folder, area_type):
        """Render a specific area of the page as an image"""
        try:
            # Create clip rectangle with some padding
            padding = 5
            clip_rect = fitz.Rect(
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                bbox[2] + padding,
                bbox[3] + padding
            )

            # Render the clipped area with high resolution
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)

            # Save the image
            filename = f"page_{page_num}_{area_type}_{area_id}.png"
            filepath = os.path.join(output_folder, filename)
            pix.save(filepath)

            # Create image reference
            image_ref = {
                'type': area_type,
                'filename': filename,
                'filepath': filepath,
                'page_num': page_num,
                'x0': bbox[0],
                'y0': bbox[1],
                'x1': bbox[2],
                'y1': bbox[3],
                'width': pix.width,
                'height': pix.height
            }

            pix = None
            return image_ref

        except Exception as e:
            logger.warning(f"Could not render {area_type} area: {e}")
            return None

    def detect_document_structure(self, doc):
        """
        Enhanced document structure detection with multi-factor analysis.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            dict: Document structure information including:
                - toc_pages: List of page numbers containing TOC
                - content_start_page: First page of main content
                - heading_hierarchy: Dict mapping heading levels to styles
                - section_boundaries: List of section start/end pages
                - document_parts: Dict identifying different document parts
        """
        structure_info = {
            'toc_pages': [],
            'content_start_page': 0,
            'heading_hierarchy': {},
            'section_boundaries': [],
            'document_parts': {
                'front_matter': [],
                'main_content': [],
                'back_matter': []
            },
            'font_analysis': self._analyze_fonts(doc),
            'spatial_analysis': self._analyze_spatial_layout(doc)
        }
        
        # First pass: Identify document parts and TOC
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            
            # Check for TOC
            if self._is_toc_page(page, page_text):
                structure_info['toc_pages'].append(page_num)
                structure_info['document_parts']['front_matter'].append(page_num)
                continue
                
            # Check for content start
            if not structure_info['content_start_page'] and self._is_content_start(page, page_text):
                structure_info['content_start_page'] = page_num
                structure_info['document_parts']['main_content'].append(page_num)
                continue
                
            # Check for back matter
            if self._is_back_matter(page, page_text):
                structure_info['document_parts']['back_matter'].append(page_num)
                continue
                
            # Default to main content
            structure_info['document_parts']['main_content'].append(page_num)
            
        # Second pass: Analyze heading hierarchy
        structure_info['heading_hierarchy'] = self._analyze_heading_hierarchy(doc, structure_info)
        
        # Third pass: Identify section boundaries
        structure_info['section_boundaries'] = self._identify_section_boundaries(doc, structure_info)
        
        return structure_info
        
    def _is_toc_page(self, page, page_text):
        """
        Enhanced TOC page detection with multiple indicators and scoring system.
        Returns True if the page is likely a Table of Contents page.
        """
        if not page_text:
            return False
            
        text_lower = page_text.lower()
        lines = page_text.split('\n')
        toc_score = 0
        
        # 1. Check for TOC title patterns
        for pattern in self.toc_compiled_patterns:
            if pattern.search(page_text):
                toc_score += 3  # Strong indicator
                break
                
        # 2. Check for TOC-like structure
        structure_indicators = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check each structure pattern
            for pattern in self.toc_structure_patterns:
                if pattern.search(line):
                    structure_indicators += 1
                    break
                    
            # Check for dot leaders with page numbers
            if re.search(r'\.{3,}\s*\d+$', line):
                structure_indicators += 1
                
        # Add structure score (capped at 3 points)
        toc_score += min(structure_indicators, 3)
        
        # 3. Check for page number patterns
        page_number_patterns = [
            r'\.{3,}\s*\d+',  # dots followed by page numbers
            r'\d+\s*\.{3,}',  # page numbers followed by dots
            r'^\s*\d+\s*$',   # standalone page numbers
            r'\d+\s*\n'       # numbers at end of lines
        ]
        
        page_number_matches = sum(1 for pattern in page_number_patterns 
                                if re.search(pattern, text_lower, re.MULTILINE))
        toc_score += min(page_number_matches, 2)  # Cap at 2 points
        
        # 4. Check for repetitive short lines (typical of TOC structure)
        short_lines = sum(1 for line in lines if 3 <= len(line.strip()) <= 60)
        if len(lines) > 5 and short_lines / len(lines) > 0.6:
            toc_score += 2
            
        # 5. Check for common TOC keywords
        toc_keywords = [
            'chapter', 'section', 'part', 'appendix',
            'introduction', 'conclusion', 'references'
        ]
        keyword_matches = sum(1 for keyword in toc_keywords if keyword in text_lower)
        toc_score += min(keyword_matches, 2)  # Cap at 2 points
        
        # Determine if this is a TOC page (threshold of 5 points)
        is_toc = toc_score >= 5
        
        if is_toc:
            logger.info(f"Page identified as TOC (score: {toc_score})")
            
        return is_toc
        
    def _is_content_start(self, page, page_text):
        """Detect the start of main content"""
        # Common indicators of content start
        content_start_indicators = [
            r'^Chapter\s+\d+',
            r'^Introduction',
            r'^Abstract',
            r'^Executive Summary',
            r'^1\.\s+[A-Z]'  # First numbered section
        ]
        
        for pattern in content_start_indicators:
            if re.search(pattern, page_text, re.MULTILINE):
                return True
                
        return False
        
    def _is_back_matter(self, page, page_text):
        """Detect back matter sections"""
        back_matter_indicators = [
            r'^References',
            r'^Bibliography',
            r'^Appendix',
            r'^Index',
            r'^Glossary'
        ]
        
        for pattern in back_matter_indicators:
            if re.search(pattern, page_text, re.MULTILINE):
                return True
                
        return False
        
    def _analyze_fonts(self, doc):
        """Analyze font usage across the document"""
        font_info = {
            'font_sizes': defaultdict(int),
            'font_families': defaultdict(int),
            'font_styles': defaultdict(int)
        }
        
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_info['font_sizes'][span.get("size", 0)] += 1
                            font_info['font_families'][span.get("font", "")] += 1
                            font_info['font_styles'][span.get("flags", 0)] += 1
                            
        return font_info
        
    def _analyze_spatial_layout(self, doc):
        """Analyze spatial layout patterns"""
        layout_info = {
            'column_count': defaultdict(int),
            'margins': [],
            'content_areas': []
        }
        
        for page in doc:
            page_rect = page.rect
            blocks = page.get_text("dict")["blocks"]
            
            # Analyze column structure
            x_positions = set()
            for block in blocks:
                if "bbox" in block:
                    x_positions.add(block["bbox"][0])  # Left edge
                    x_positions.add(block["bbox"][2])  # Right edge
                    
            layout_info['column_count'][len(x_positions)] += 1
            
            # Record content areas
            for block in blocks:
                if "bbox" in block:
                    layout_info['content_areas'].append(block["bbox"])
                    
        return layout_info
        
    def _analyze_heading_hierarchy(self, doc, structure_info):
        """Analyze and establish heading hierarchy with improved context awareness"""
        heading_styles = defaultdict(list)
        font_families = defaultdict(int)
        
        # First pass: collect all potential headings and font statistics
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            if self._is_heading_candidate(text, span):
                                font = span.get("font", "")
                                font_families[font] += 1
                                heading_styles[span.get("size", 0)].append({
                                    'text': text,
                                    'font': font,
                                    'flags': span.get("flags", 0),
                                    'page': page_num,
                                    'bbox': span.get("bbox", (0, 0, 0, 0))
                                })
                                
        # Determine dominant font family for headings
        dominant_font = max(font_families.items(), key=lambda x: x[1])[0] if font_families else None
        
        # Establish heading levels based on font size, style, and context
        heading_hierarchy = {}
        sorted_sizes = sorted(heading_styles.keys(), reverse=True)
        
        # Ensure we have at least 3 heading levels
        if len(sorted_sizes) < 3:
            # Create artificial heading levels if needed
            base_size = max(sorted_sizes) if sorted_sizes else 16
            sorted_sizes = [base_size, base_size * 0.8, base_size * 0.6]
        
        # Analyze heading patterns for each size
        for i, size in enumerate(sorted_sizes):
            examples = heading_styles.get(size, [])
            patterns = defaultdict(int)
            
            # Analyze patterns in examples
            for example in examples:
                text = example['text']
                # Count different heading patterns
                if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', text):
                    patterns['title_case'] += 1
                elif re.match(r'^\d+\.', text):
                    patterns['numbered'] += 1
                elif re.match(r'^[IVX]+\.', text):
                    patterns['roman'] += 1
                elif re.match(r'^[A-Z]\.', text):
                    patterns['lettered'] += 1
                elif re.match(r'^Chapter\s+\d+', text):
                    patterns['chapter'] += 1
            
            # Determine most common pattern
            dominant_pattern = max(patterns.items(), key=lambda x: x[1])[0] if patterns else None
            
            heading_hierarchy[f'h{i+1}'] = {
                'font_size': size,
                'font_family': dominant_font,
                'pattern': dominant_pattern,
                'examples': examples[:3],  # Keep first 3 examples
                'confidence': min(1.0, len(examples) / 5)  # Confidence based on number of examples
            }
            
        return heading_hierarchy
        
    def _is_heading_candidate(self, text, span):
        """Determine if text is likely a heading"""
        if not text or len(text) > 200:  # Too long for a heading
            return False
            
        # Check font characteristics
        is_bold = bool(span.get("flags", 0) & 16)
        font_size = span.get("size", 0)
        
        # Check text patterns
        heading_patterns = [
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title case
            r'^\d+\.\s+[A-Z]',  # Numbered sections
            r'^[IVX]+\.\s+[A-Z]',  # Roman numerals
            r'^[A-Z]\.\s+[A-Z]',  # Lettered sections
            r'^Chapter\s+\d+',  # Chapter headings
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\d+$'  # Title with number
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text):
                return True
                
        # Also consider font size and style
        if is_bold and font_size > 12:  # Bold and larger than normal text
            return True
                
        return False
        
    def _identify_section_boundaries(self, doc, structure_info):
        """Identify section boundaries based on heading patterns"""
        section_boundaries = []
        current_section = None
        
        for page_num in structure_info['document_parts']['main_content']:
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            if self._is_heading_candidate(text, span):
                                if current_section:
                                    current_section['end_page'] = page_num - 1
                                    section_boundaries.append(current_section)
                                    
                                current_section = {
                                    'start_page': page_num,
                                    'heading': text,
                                    'level': self._determine_heading_level(span, structure_info)
                                }
                                
        # Add the last section
        if current_section:
            current_section['end_page'] = len(doc) - 1
            section_boundaries.append(current_section)
            
        return section_boundaries
        
    def _determine_heading_level(self, span, structure_info):
        """Determine heading level based on font characteristics"""
        font_size = span.get("size", 0)
        
        for level, info in structure_info['heading_hierarchy'].items():
            if abs(font_size - info['font_size']) < 0.5:  # Allow small font size differences
                return int(level[1])  # Extract number from 'h1', 'h2', etc.
                
        return 1  # Default to level 1 if no match found

    def extract_toc_from_content_two_pass(self, doc=None, structure_info=None, content=None) -> List[Dict[str, Any]]:
        """
        Robust two-pass ToC extraction that combines document structure analysis with content parsing.
        
        Args:
            doc: PyMuPDF document object (optional)
            structure_info: Pre-computed document structure (optional)
            content: Raw text content (optional)
            
        Returns:
            List of TOC entries with structure: {
                'title': str,
                'page': int,
                'level': int,
                'source': str,
                'confidence': float
            }
        """
        toc_entries = []
        
        # --- Pass 1: Extract from document structure if available ---
        if doc is not None:
            if structure_info is None:
                structure_info = self.detect_document_structure(doc)
                
            # 1A. Extract from explicit TOC pages
            toc_page_numbers = structure_info.get('toc_pages', [])
            for page_num in toc_page_numbers:
                page = doc[page_num]
                page_text = page.get_text()
                lines = page_text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Pattern: "Title .......... 5"
                    m = re.match(r'^(.+?)\s*\.{3,}\s*(\d+)$', line)
                    if m:
                        title = m.group(1).strip()
                        page_ref = int(m.group(2))
                        toc_entries.append({
                            'title': title,
                            'page': page_ref,
                            'level': 1,  # Default, will refine in pass 2
                            'source': 'toc_page_chapter',  # Updated source field
                            'confidence': 0.9
                        })
                        continue
                        
                    # Pattern: "1.1 Section Title   7"
                    m2 = re.match(r'^(\d+(?:\.\d+)*)(?:\s+)(.+?)\s+(\d+)$', line)
                    if m2:
                        number = m2.group(1)
                        title = m2.group(2).strip()
                        page_ref = int(m2.group(3))
                        level = number.count('.') + 1
                        toc_entries.append({
                            'title': f"{number} {title}",
                            'page': page_ref,
                            'level': level,
                            'source': 'toc_page_numbered',
                            'confidence': 0.95
                        })
                        continue
                        
                    # Pattern: "Chapter 1: Title"
                    m3 = re.match(r'^(?:Chapter|Section)\s+(\d+)(?::\s*)(.+)$', line, re.IGNORECASE)
                    if m3:
                        number = m3.group(1)
                        title = m3.group(2).strip()
                        toc_entries.append({
                            'title': f"Chapter {number}: {title}",
                            'page': None,  # Will be filled in pass 2
                            'level': 1,
                            'source': 'toc_page_chapter',
                            'confidence': 0.85
                        })
            
            # 1B. Extract from heading structure
            section_boundaries = structure_info.get('section_boundaries', [])
            for section in section_boundaries:
                heading = section.get('heading')
                page = section.get('start_page')
                level = section.get('level', 1)
                if heading and page is not None:
                    toc_entries.append({
                        'title': heading,
                        'page': page + 1,  # Convert to 1-based page
                        'level': level,
                        'source': 'heading_structure',
                        'confidence': 0.8
                    })
        
        # --- Pass 2: Extract from content if available ---
        if content:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                
                # Skip lines that are clearly not TOC entries
                if line.startswith('Chapter') or line.startswith('Section'):
                    continue
                
                # Skip lines that are only Markdown underlines (e.g., '===', '---')
                if re.match(r'^[=\-]+$', line):
                    continue
                
                # Try to detect heading level and text
                level = 1
                text = line
                
                # Check for common heading patterns
                if line.startswith('#'):
                    level = len(re.match(r'^#+', line).group())
                    text = line[level:].strip()
                elif re.match(r'^\d+\.', line):
                    level = 2
                    text = re.sub(r'^\d+\.\s*', '', line)
                elif re.match(r'^[A-Z]\.', line):
                    level = 3
                    text = re.sub(r'^[A-Z]\.\s*', '', line)
                
                # Skip if text is too short after cleaning
                if len(text) < 3:
                    continue
                
                # Skip if too similar to existing entries
                if any(self._is_similar_to_previous(text, entry['title']) for entry in toc_entries):
                    continue
                
                toc_entries.append({
                    'title': text,
                    'page': None,  # Will be filled in pass 3
                    'level': level,
                    'source': 'content_analysis',
                    'confidence': 0.7
                })
        
        # --- Pass 3: Reconcile and refine entries ---
        # 3A. Normalize and deduplicate by canonical headings
        canonical_headings = {'introduction', 'background', 'methods', 'history', 'current state'}
        seen_titles = set()
        refined_entries = []
        
        for entry in sorted(toc_entries, key=lambda x: (len(x['title']), -x['confidence'])):
            title_lower = entry['title'].lower()
            # Extract canonical heading if present
            canonical = None
            for heading in canonical_headings:
                if heading in title_lower:
                    canonical = heading
                    break
            if canonical:
                title_lower = canonical
            # Limit the title to a maximum of 16 words
            words = title_lower.split()
            if len(words) > 16:
                title_lower = ' '.join(words[:16])
            # Skip if this is a substring of any seen title
            if any(title_lower in seen_title for seen_title in seen_titles):
                continue
            # If any seen title is a substring of this, remove the longer one
            to_remove = [seen_title for seen_title in seen_titles if seen_title in title_lower]
            for rem in to_remove:
                seen_titles.remove(rem)
                refined_entries = [e for e in refined_entries if e['title'].lower() != rem]
            seen_titles.add(title_lower)
            refined_entries.append(entry)
        # 3B. Sort by page number (if available) and level
        refined_entries.sort(key=lambda x: (
            x['page'] if x['page'] is not None else float('inf'),
            x['level']
        ))
        # 3C. Fill in missing page numbers based on order
        current_page = 1
        for entry in refined_entries:
            if entry['page'] is None:
                entry['page'] = current_page
            current_page = entry['page'] + 1
        return refined_entries

    def _is_similar_to_previous(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar using Levenshtein distance."""
        return SequenceMatcher(None, text1, text2).ratio() > threshold

    def extract_structured_content_from_pdf(self, pdf_path: str, extracted_images: Optional[List[Dict[str, Any]]] = None) -> Document:
        """
        Extract structured content from PDF with enhanced TOC handling.
        """
        try:
            # Extract raw content first
            raw_content = self.extract_text_from_pdf(pdf_path)
            
            # Extract TOC using two-pass approach
            toc_entries = self.extract_toc_from_content_two_pass(raw_content)
            
            # Create document with TOC entries
            document = Document(
                title=os.path.basename(pdf_path),
                content_blocks=[],
                metadata={'toc_entries': toc_entries}
            )
            
            # Process content blocks
            # ... existing content block processing code ...
            
            return document
            
        except Exception as e:
            logger.error(f"Error extracting structured content: {e}")
            raise

class StructuredContentExtractor:
    """Extract structured content from PDF with enhanced parsing"""
    
    def __init__(self):
        self.settings = config_manager.pdf_processing_settings
        self.parser = PDFParser()
        
    def extract_structured_content_from_pdf(self, filepath, all_extracted_image_refs):
        """Main function to extract structured content from PDF - returns Document object"""
        logger.info(f"--- Structuring Content from PDF: {os.path.basename(filepath)} ---")

        images_by_page = self.parser.groupby_images_by_page(all_extracted_image_refs)

        try:
            doc = fitz.open(filepath)

            # Analyze document structure
            structure_analysis = self.parser.detect_document_structure(doc)

            # Extract content with structure as Document object
            document = self._extract_content_as_document(doc, images_by_page, structure_analysis, filepath)

            doc.close()

            logger.info(f"Extracted Document with {len(document.content_blocks)} content blocks across {document.total_pages} pages")
            return document

        except Exception as e:
            logger.error(f"Error extracting structured content: {e}")
            # Return empty document on error
            return Document(
                title=f"Error Processing {os.path.basename(filepath)}",
                source_filepath=filepath,
                metadata={'error': str(e)}
            )

    def extract_structured_content_from_pdf_legacy(self, filepath, all_extracted_image_refs):
        """Legacy method that returns old format for backward compatibility"""
        document = self.extract_structured_content_from_pdf(filepath, all_extracted_image_refs)

        # Convert Document back to legacy format
        from structured_document_model import convert_document_to_legacy_format
        return convert_document_to_legacy_format(document)
    
    def _analyze_document_structure(self, doc):
        """Analyze document structure with global font analysis and spatial understanding"""
        structure_info = {
            'total_pages': len(doc),
            'font_analysis': {},
            'toc_pages': [],
            'content_start_page': 0,
            'bibliography_start_page': None,
            'dominant_font_size': 12.0,
            'heading_font_sizes': set(),
            'font_hierarchy': {},  # Enhanced font hierarchy mapping
            'body_text_style': None,  # Dominant body text style
            'heading_styles': {}  # Mapping of heading levels to styles
        }

        # Perform global font analysis across entire document
        font_profile = self._perform_global_font_analysis(doc)
        structure_info.update(font_profile)

        # Analyze fonts and structure in first few pages for TOC detection
        for page_num in range(min(10, len(doc))):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # Check for TOC indicators
                            text = span.get("text", "").strip().lower()
                            if any(keyword in text for keyword in self.settings['toc_detection_keywords']):
                                if page_num not in structure_info['toc_pages']:
                                    structure_info['toc_pages'].append(page_num)

        return structure_info

    def _perform_global_font_analysis(self, doc):
        """
        Enhanced global font analysis with statistical methods for adaptive heading detection.
        
        Uses multiple statistical measures to identify:
        1. Body text characteristics (mode, mean, standard deviation)
        2. Heading hierarchy based on statistical clustering
        3. Font style patterns and their distribution
        """
        logger.info("🔍 Performing enhanced statistical font analysis...")

        font_styles = {}  # font_key -> count
        font_sizes = []
        font_weights = defaultdict(int)  # Track font weights
        font_families = defaultdict(int)  # Track font families

        # First pass: collect all font information
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_size = span.get("size", 12.0)
                            font_name = span.get("font", "Arial")
                            font_flags = span.get("flags", 0)

                            # Determine font style
                            is_bold = bool(font_flags & 2**4)  # Bold flag
                            is_italic = bool(font_flags & 2**1)  # Italic flag

                            # Create font style key
                            style_key = f"{font_name}, {font_size:.1f}pt"
                            if is_bold:
                                style_key += ", bold"
                                font_weights['bold'] += 1
                            if is_italic:
                                style_key += ", italic"
                                font_weights['italic'] += 1

                            # Count occurrences
                            font_styles[style_key] = font_styles.get(style_key, 0) + 1
                            font_sizes.append(font_size)
                            font_families[font_name] += 1

        # Statistical analysis of font sizes
        from collections import Counter
        import numpy as np
        from scipy import stats

        # Convert to numpy array for statistical analysis
        font_sizes_array = np.array(font_sizes)
        
        # Calculate basic statistics
        mean_size = np.mean(font_sizes_array)
        median_size = np.median(font_sizes_array)
        std_size = np.std(font_sizes_array)
        
        # Use mode for body text size (more robust than mean for discrete values)
        size_counter = Counter(font_sizes)
        body_text_size = size_counter.most_common(1)[0][0]
        
        # Calculate size clusters using KDE (Kernel Density Estimation)
        kde = stats.gaussian_kde(font_sizes_array)
        x_range = np.linspace(min(font_sizes_array), max(font_sizes_array), 100)
        density = kde(x_range)
        
        # Find peaks in the density plot (potential heading sizes)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(density, height=0.1*np.max(density))
        potential_heading_sizes = sorted(x_range[peaks], reverse=True)
        
        # Filter heading sizes based on statistical significance
        heading_sizes = []
        for size in potential_heading_sizes:
            if size > body_text_size + std_size:  # Must be significantly larger than body text
                heading_sizes.append(size)
        
        # Limit to top 6 heading sizes
        heading_sizes = heading_sizes[:6]
        
        # Build heading hierarchy
        heading_styles = {}
        font_hierarchy = {}
        
        # Map heading levels to font sizes
        for i, size in enumerate(heading_sizes):
            level = i + 1
            heading_styles[f"h{level}"] = size
            font_hierarchy[size] = level
            
            # Calculate confidence score for this heading level
            size_frequency = size_counter[size] / len(font_sizes)
            confidence = min(1.0, size_frequency * 10)  # Scale frequency to 0-1 range
            
            heading_styles[f"h{level}_confidence"] = confidence

        # Identify dominant font family
        dominant_font = max(font_families.items(), key=lambda x: x[1])[0]
        
        # Create body text style string
        body_text_style = f"{dominant_font}, {body_text_size:.1f}pt"
        if font_weights['bold'] > font_weights['italic']:
            body_text_style += ", bold"
        elif font_weights['italic'] > font_weights['bold']:
            body_text_style += ", italic"

        logger.info(f"📊 Enhanced font analysis complete:")
        logger.info(f"   • Body text: {body_text_style} (size: {body_text_size:.1f}pt)")
        logger.info(f"   • Font statistics: mean={mean_size:.1f}, median={median_size:.1f}, std={std_size:.1f}")
        logger.info(f"   • Heading hierarchy: {heading_styles}")
        logger.info(f"   • Font families: {dict(font_families)}")

        return {
            'font_analysis': font_styles,
            'dominant_font_size': body_text_size,
            'heading_font_sizes': set(heading_sizes),
            'font_hierarchy': font_hierarchy,
            'body_text_style': body_text_style,
            'heading_styles': heading_styles,
            'font_statistics': {
                'mean': mean_size,
                'median': median_size,
                'std': std_size,
                'dominant_font': dominant_font
            }
        }

    def _extract_spatial_elements(self, page, page_num, structure_analysis):
        """
        Proposition 1: Extract all text and image elements with spatial coordinates.

        Returns a list of elements with bounding boxes for spatial analysis.
        """
        elements = []

        # Extract text blocks with spatial information
        blocks = page.get_text("dict")["blocks"]

        for block_num, block in enumerate(blocks):
            if "lines" not in block:
                continue

            # Extract text content and formatting
            block_text = ""
            block_formatting = {
                'font_sizes': [],
                'font_names': [],
                'is_bold': False,
                'is_italic': False
            }

            for line in block["lines"]:
                for span in line["spans"]:
                    span_text = span["text"]
                    block_text += span_text

                    # Collect formatting information
                    block_formatting['font_sizes'].append(span.get("size", 12.0))
                    block_formatting['font_names'].append(span.get("font", "Arial"))

                    font_flags = span.get("flags", 0)
                    if font_flags & 2**4:  # Bold
                        block_formatting['is_bold'] = True
                    if font_flags & 2**1:  # Italic
                        block_formatting['is_italic'] = True

            block_text = block_text.strip()
            if not block_text:
                continue

            # Create spatial element
            element = {
                'type': 'text',
                'content': block_text,
                'bbox': block.get("bbox", [0, 0, 0, 0]),
                'page_num': page_num,
                'block_num': block_num,
                'formatting': block_formatting,
                'element_id': f"text_{page_num}_{block_num}"
            }

            elements.append(element)

        return elements

    def _apply_spatial_reading_order(self, elements):
        """
        Enhanced spatial reading order with multi-column layout detection and complex document structure support.

        Addresses detection issues by:
        1. Detecting multi-column layouts
        2. Handling sidebars and footnotes
        3. Using adaptive vertical tolerance
        4. Supporting right-to-left reading patterns
        """
        if not elements:
            return elements

        # Analyze page layout to detect columns
        logger.debug("Analyzing page layout for reading order...")
        layout_analysis = self._analyze_page_layout(elements)

        if layout_analysis['is_multi_column']:
            logger.info(f"Multi-column layout detected. Columns: {len(layout_analysis.get('columns', []))}")
            sorted_elements = self._sort_multi_column_layout(elements, layout_analysis)
        else:
            logger.info("Single-column layout detected.")
            sorted_elements = self._sort_single_column_layout(elements)

        # Assign reading order positions
        for i, element in enumerate(sorted_elements):
            element['reading_order_position'] = i

        logger.info(f"Applied spatial reading order. Total elements: {len(sorted_elements)}. Multi-column: {layout_analysis['is_multi_column']}.")
        if sorted_elements:
            logger.debug("First 5 elements in reading order:")
            for idx, elem in enumerate(sorted_elements[:5]):
                content_preview = elem.get('content', 'N/A')[:50].replace('\n', ' ')
                logger.debug(f"  {idx+1}. BBox: {elem['bbox']}, Content: '{content_preview}...'")
        return sorted_elements

    def _analyze_page_layout(self, elements):
        """Analyze page layout to detect columns, sidebars, and other structural elements"""
        if not elements:
            return {'is_multi_column': False, 'columns': []}

        # Extract x-coordinates of all elements
        x_positions = []
        for element in elements:
            bbox = element['bbox']
            x_left, x_right = bbox[0], bbox[2]
            x_positions.extend([x_left, x_right])

        # Find potential column boundaries using clustering
        x_positions.sort()

        # Simple column detection: look for gaps in x-positions
        gaps = []
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            if gap > 50:  # Significant gap threshold
                gaps.append({
                    'position': (x_positions[i-1] + x_positions[i]) / 2,
                    'size': gap
                })

        # Determine if multi-column based on significant gaps
        is_multi_column = len(gaps) >= 1 and any(gap['size'] > 100 for gap in gaps)

        # Define column boundaries
        columns = []
        if is_multi_column and gaps:
            # Sort gaps by position
            gaps.sort(key=lambda x: x['position'])

            # Create column definitions
            page_left = min(x_positions)
            page_right = max(x_positions)

            prev_boundary = page_left
            for gap in gaps:
                if gap['size'] > 100:  # Only consider significant gaps
                    columns.append({
                        'left': prev_boundary,
                        'right': gap['position'],
                        'center': (prev_boundary + gap['position']) / 2
                    })
                    prev_boundary = gap['position']

            # Add final column
            columns.append({
                'left': prev_boundary,
                'right': page_right,
                'center': (prev_boundary + page_right) / 2
            })

        return {
            'is_multi_column': is_multi_column,
            'columns': columns,
            'gaps': gaps
        }

    def _sort_multi_column_layout(self, elements, layout_analysis):
        """
        Sort elements in multi-column layout using reading bands approach.

        Primary sort: Vertical position (Y-coordinate), then Horizontal position (X-coordinate) within each column.
        Columns are then concatenated from left to right.
        """
        if not elements:
            return []

        columns_data = layout_analysis.get('columns', [])
        if not columns_data: # Fallback if column detection failed to define columns
            logger.warning("Multi-column layout detected but no column boundaries found. Using single column sort as fallback.")
            return self._sort_single_column_layout(elements)

        # Assign elements to columns
        elements_in_columns = [[] for _ in columns_data]
        for element in elements:
            bbox = element['bbox']
            element_center_x = (bbox[0] + bbox[2]) / 2
            assigned_to_column = False
            for i, col in enumerate(columns_data):
                if col['left'] <= element_center_x < col['right']:
                    elements_in_columns[i].append(element)
                    assigned_to_column = True
                    break
            if not assigned_to_column: # If element doesn't fit neatly, assign to nearest column or handle as outlier
                # For now, assign to the last column if it's to the right, or first if to the left, or skip
                if element_center_x >= columns_data[-1]['right']:
                    elements_in_columns[-1].append(element)
                elif element_center_x < columns_data[0]['left']:
                     elements_in_columns[0].append(element)
                else: # Could be an element spanning columns or in a gutter; log and assign based on proximity
                    logger.debug(f"Element {element.get('element_id', 'N/A')} at {bbox} not fitting neatly into columns. Assigning to closest.")
                    # Simplified: find closest column by center distance
                    closest_col_idx = 0
                    min_dist = float('inf')
                    for i, col in enumerate(columns_data):
                        dist = abs(element_center_x - col['center'])
                        if dist < min_dist:
                            min_dist = dist
                            closest_col_idx = i
                    elements_in_columns[closest_col_idx].append(element)


        # Sort elements within each column (top-to-bottom, then left-to-right)
        for i in range(len(elements_in_columns)):
            elements_in_columns[i].sort(key=lambda el: (el['bbox'][1], el['bbox'][0]))

        # Concatenate sorted elements from columns
        sorted_elements = []
        for col_elements in elements_in_columns:
            sorted_elements.extend(col_elements)

        logger.debug(f"Sorted {len(elements)} elements into {len(columns_data)} columns.")
        return sorted_elements

    def _sort_single_column_layout(self, elements):
        """Sort elements in single-column layout with enhanced vertical tolerance"""
        # Calculate adaptive vertical tolerance based on element sizes
        element_heights = [abs(e['bbox'][3] - e['bbox'][1]) for e in elements if e['bbox']]
        avg_height = sum(element_heights) / len(element_heights) if element_heights else 20

        # Use adaptive tolerance: smaller for dense layouts, larger for sparse layouts
        vertical_tolerance = max(10, min(30, avg_height * 0.5))

        def enhanced_spatial_sort_key(element):
            bbox = element['bbox']
            y0, x0 = bbox[1], bbox[0]  # y0 (top), x0 (left)

            # Group elements by approximate vertical bands with adaptive tolerance
            vertical_band = round(y0 / vertical_tolerance) * vertical_tolerance

            return (vertical_band, x0)  # Positive y for top-to-bottom reading (smaller y = higher on page)

        return sorted(elements, key=enhanced_spatial_sort_key)

    def _create_content_block_from_element(self, element, structure_analysis):
        """Create ContentBlock from spatial element with enhanced classification"""
        if element['type'] != 'text':
            return None

        text = element['content']
        formatting = element['formatting']
        bbox = element['bbox']
        page_num = element['page_num']
        block_num = element['block_num']

        # Filter out unwanted content
        if self._should_filter_content(text, bbox, page_num, structure_analysis):
            return None

        # Use adaptive heading detection based on global font analysis
        content_type = self._classify_content_type_adaptive(text, formatting, structure_analysis)

        # Create appropriate ContentBlock
        if content_type.startswith('h'):
            level = int(content_type[1])
            return Heading(
                block_type=ContentType.HEADING,
                original_text=text,
                page_num=page_num,
                bbox=tuple(bbox),
                block_num=block_num,
                formatting=formatting,
                level=level,
                content=text
            )
        elif content_type == 'paragraph':
            return Paragraph(
                block_type=ContentType.PARAGRAPH,
                original_text=text,
                page_num=page_num,
                bbox=tuple(bbox),
                block_num=block_num,
                formatting=formatting,
                content=text
            )
        # Add other content types as needed

        return None

    def _classify_content_type_adaptive(self, text, formatting, structure_analysis):
        """
        Enhanced adaptive content classification with statistical analysis.

        Uses multiple factors to classify content:
        1. Statistical font analysis (size, weight, family)
        2. Text characteristics (length, patterns)
        3. Contextual clues (position, surrounding content)
        4. Confidence scoring for each classification
        """
        import re
        import numpy as np

        # Get document structure information
        font_hierarchy = structure_analysis.get('font_hierarchy', {})
        font_stats = structure_analysis.get('font_statistics', {})
        dominant_font_size = structure_analysis.get('dominant_font_size', 12.0)
        body_text_style = structure_analysis.get('body_text_style', '')

        # Extract formatting attributes
        font_sizes = formatting.get('font_sizes', [12.0])
        primary_font_size = max(font_sizes) if font_sizes else 12.0
        font_names = formatting.get('font_names', [])
        primary_font = font_names[0] if font_names else ''

        # Enhanced formatting detection
        is_bold = formatting.get('is_bold', False) or self._detect_bold_from_flags(formatting)
        is_italic = formatting.get('is_italic', False) or self._detect_italic_from_flags(formatting)
        font_color = formatting.get('color', 0)

        # Text analysis
        text_clean = text.strip()
        text_length = len(text_clean)
        word_count = len(text_clean.split())

        # Early filtering: very long text is unlikely to be a heading
        if word_count > 20 or text_length > 150:
            return 'paragraph'

        # Statistical heading detection
        heading_score = 0.0
        score_details = []
        confidence = 0.0

        # 1. Font size analysis (40% weight)
        size_ratio = primary_font_size / dominant_font_size if dominant_font_size > 0 else 1.0
        std_dev = font_stats.get('std', 1.0)
        
        # Calculate z-score for font size
        z_score = (primary_font_size - font_stats.get('mean', dominant_font_size)) / std_dev
        
        if z_score > 2.0:  # More than 2 standard deviations above mean
            heading_score += 0.4; score_details.append(f"ZScore>2.0 ({z_score:.2f})")
        elif z_score > 1.5:
            heading_score += 0.3; score_details.append(f"ZScore>1.5 ({z_score:.2f})")
        elif z_score > 1.0:
            heading_score += 0.2; score_details.append(f"ZScore>1.0 ({z_score:.2f})")

        # 2. Font weight and style analysis (30% weight)
        if is_bold:
            heading_score += 0.3; score_details.append("Bold")
        elif is_italic:
            heading_score += 0.1; score_details.append("ItalicOnly")

        # 3. Font family analysis (10% weight)
        if primary_font and body_text_style:
            if primary_font.lower() not in body_text_style.lower():
                heading_score += 0.1; score_details.append("DiffFontFamily")

        # 4. Text characteristics (20% weight)
        # Length-based scoring
        if text_length <= 50:
            heading_score += 0.1; score_details.append(f"Len<=50 ({text_length})")
        elif text_length <= 100:
            heading_score += 0.05; score_details.append(f"Len<=100 ({text_length})")

        # Pattern-based scoring
        has_patterns = self._has_heading_patterns(text_clean)
        if has_patterns:
            heading_score += 0.1; score_details.append("HasPatterns")

        # 5. Position-based analysis
        appears_section_start = self._appears_to_be_section_start(text_clean)
        if appears_section_start:
            heading_score += 0.1; score_details.append("SectionStart")

        # Calculate final confidence
        confidence = min(1.0, heading_score)

        # Determine heading level based on font size and hierarchy
        heading_level = None
        for h_font_size, h_level in font_hierarchy.items():
            if abs(primary_font_size - h_font_size) <= 0.5:  # 0.5pt tolerance
                heading_level = h_level
                break

        # If no exact match, estimate level based on size ratio
        if heading_level is None and heading_score >= 0.6:
            # Estimate level based on how many standard deviations above mean
            estimated_level = min(6, max(1, int(z_score)))
            heading_level = estimated_level

        # Log classification details
        if heading_score >= 0.6:
            logger.debug(
                f"[Adaptive Classifier] Classified as h{heading_level} "
                f"(score: {heading_score:.2f}, confidence: {confidence:.2f}): "
                f"'{text_clean[:100]}...' "
                f"Details: {', '.join(score_details)}"
            )
            return f'h{heading_level}'

        # Check for list items
        list_patterns = [
            r'^\s*[•\-\*]\s+',  # Bullet points
            r'^\s*\d+[\.\)]\s+',  # Numbered lists
            r'^\s*[a-zA-Z][\.\)]\s+',  # Lettered lists
        ]

        for pattern in list_patterns:
            if re.match(pattern, text_clean):
                return 'list_item'

        # Default to paragraph
        return 'paragraph'

    def _detect_bold_from_flags(self, formatting):
        """Detect bold formatting from font flags"""
        flags = formatting.get('flags', 0)
        # Font flag 16 typically indicates bold in PyMuPDF
        return bool(flags & 16)

    def _detect_italic_from_flags(self, formatting):
        """Detect italic formatting from font flags"""
        flags = formatting.get('flags', 0)
        # Font flag 2 typically indicates italic in PyMuPDF
        return bool(flags & 2)

    def _has_heading_patterns(self, text):
        """Check for common heading patterns"""
        import re

        heading_patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 Introduction"
            # ALL CAPS headings (max 5 words) - Constrained to avoid long capitalized sentences.
            r'^([A-Z]+\s){0,4}[A-Z]+$',
            # Title Case (max 7 words) - Constrained to avoid long title-cased sentences.
            r'^([A-Z][a-z]+\s){0,6}[A-Z][a-z]+$',
            r'^(Chapter|Section|Part|Appendix)\s+\d+',  # Chapter/Section numbers
            r'^\d+\.\d+',  # Numbered sections like "2.1"
            r'^[IVX]+\.\s+[A-Z]',  # Roman numerals
            r'^[A-Z]\.\s+[A-Z]',  # Single letter sections "A."
        ]
        # Check if the text matches any of the defined heading patterns.
        # These patterns contribute to the heading_score in _classify_content_type_adaptive.
        for pattern in heading_patterns:
            if re.search(pattern, text):
                logger.debug(f"Found heading pattern ({pattern}) in text: '{text[:100]}...'")
                return True
        return False

    def _appears_to_be_section_start(self, text):
        """Check if text appears to start a new section"""
        import re

        section_indicators = [
            r'^(introduction|conclusion|abstract|summary|overview)',
            r'^(background|methodology|results|discussion)',
            r'^(references|bibliography|appendix|glossary)',
            r'^\d+\.\s*(introduction|background|method)',
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in section_indicators)

    def _is_list_item_pattern(self, text):
        """Check if text follows list item patterns"""
        import re

        list_patterns = [
            r'^[•·▪▫‣⁃]\s+',  # Bullet points
            r'^[-*+]\s+',     # Dash/asterisk bullets
            r'^\d+[.)]\s+',   # Numbered lists
            r'^[a-z][.)]\s+', # Lettered lists
            r'^\([a-z0-9]+\)\s+',  # Parenthetical lists
        ]

        return any(re.search(pattern, text) for pattern in list_patterns)

    def _associate_images_with_text_spatial(self, content_blocks, extracted_images, page_num):
        """
        Proposition 1: Associate images with text based on spatial relationships.

        Determines if an image should be placed before, after, or alongside text blocks
        based on spatial positioning and implements explicit caption detection.
        """
        if not extracted_images:
            return content_blocks

        logger.debug(f"Associating {len(extracted_images)} images with {len(content_blocks)} text blocks on page {page_num}")

        # Create image placeholders with spatial analysis
        image_placeholders = []

        for img_idx, image in enumerate(extracted_images):
            # Create basic image placeholder
            placeholder = ImagePlaceholder(
                block_type=ContentType.IMAGE_PLACEHOLDER,
                original_text="",
                page_num=page_num,
                bbox=tuple(image.get('position', {}).get('bbox', [0, 0, 0, 0])),
                block_num=len(content_blocks) + img_idx,
                image_path=image.get('filepath', ''),
                width=image.get('width'),
                height=image.get('height')
            )

            # Determine spatial relationship with text blocks
            spatial_info = self._analyze_image_text_spatial_relationship(placeholder, content_blocks)

            placeholder.spatial_relationship = spatial_info['relationship']
            placeholder.reading_order_position = spatial_info['reading_order_position']

            # Detect and link captions
            caption_info = self._detect_and_link_caption(placeholder, content_blocks)
            if caption_info:
                placeholder.caption_block_id = caption_info['caption_block_id']
                placeholder.caption = caption_info['caption_text']

            image_placeholders.append(placeholder)

        # Insert image placeholders into content blocks at appropriate positions
        enhanced_content_blocks = self._insert_images_by_spatial_order(content_blocks, image_placeholders)

        return enhanced_content_blocks

    def _analyze_image_text_spatial_relationship(self, image_placeholder, text_blocks):
        """
        Analyze spatial relationship between an image and surrounding text blocks.

        Returns relationship type and suggested reading order position.
        """
        image_bbox = image_placeholder.bbox
        image_center_y = (image_bbox[1] + image_bbox[3]) / 2
        image_center_x = (image_bbox[0] + image_bbox[2]) / 2

        # Find closest text blocks
        distances = []
        for block in text_blocks:
            if not hasattr(block, 'bbox') or not block.bbox:
                continue

            text_bbox = block.bbox
            text_center_y = (text_bbox[1] + text_bbox[3]) / 2
            text_center_x = (text_bbox[0] + text_bbox[2]) / 2

            # Calculate distance
            distance = ((image_center_x - text_center_x) ** 2 + (image_center_y - text_center_y) ** 2) ** 0.5

            distances.append({
                'block': block,
                'distance': distance,
                'relative_position': self._get_relative_position(image_bbox, text_bbox)
            })

        # Sort by distance
        distances.sort(key=lambda x: x['distance'])

        # Determine relationship based on closest blocks
        if not distances:
            return {'relationship': 'standalone', 'reading_order_position': None}

        closest = distances[0]
        relationship = self._determine_spatial_relationship(closest['relative_position'])

        # Determine reading order position
        reading_order_pos = getattr(closest['block'], 'reading_order_position', None)
        if reading_order_pos is not None:
            if relationship == 'before':
                reading_order_pos = reading_order_pos
            elif relationship == 'after':
                reading_order_pos = reading_order_pos + 1
            else:  # alongside or wrapped
                reading_order_pos = reading_order_pos + 0.5  # Insert between blocks

        return {
            'relationship': relationship,
            'reading_order_position': reading_order_pos
        }

    def _get_relative_position(self, image_bbox, text_bbox):
        """Determine relative position of image to text block"""
        img_center_x = (image_bbox[0] + image_bbox[2]) / 2
        img_center_y = (image_bbox[1] + image_bbox[3]) / 2
        text_center_x = (text_bbox[0] + text_bbox[2]) / 2
        text_center_y = (text_bbox[1] + text_bbox[3]) / 2

        # Determine primary direction
        dx = img_center_x - text_center_x
        dy = img_center_y - text_center_y

        if abs(dy) > abs(dx):
            return 'above' if dy < 0 else 'below'
        else:
            return 'left' if dx < 0 else 'right'

    def _determine_spatial_relationship(self, relative_position):
        """Map relative position to spatial relationship type"""
        if relative_position in ['above']:
            return 'before'
        elif relative_position in ['below']:
            return 'after'
        elif relative_position in ['left', 'right']:
            return 'alongside'
        else:
            return 'wrapped'

    def _detect_and_link_caption(self, image_placeholder, text_blocks):
        """
        Enhanced caption detection with flexible proximity rules and multi-directional search.

        Addresses detection issues by:
        1. Expanding search directions (below, above, adjacent)
        2. Using adaptive distance thresholds based on image size
        3. Implementing fuzzy alignment matching
        4. Adding confidence scoring for better candidate selection
        """
        image_bbox = image_placeholder.bbox
        image_width = image_bbox[2] - image_bbox[0]
        image_height = image_bbox[3] - image_bbox[1]

        # Adaptive distance thresholds based on image size
        max_vertical_distance = min(100, max(50, image_height * 0.5))
        max_horizontal_distance = min(150, max(75, image_width * 0.3))

        caption_candidates = []

        for block in text_blocks:
            if not hasattr(block, 'bbox') or not block.bbox:
                continue

            text_bbox = block.bbox
            text_content = getattr(block, 'content', '') or getattr(block, 'original_text', '')

            # Debug: Log all text blocks being evaluated
            logger.debug(f"Evaluating text block: '{text_content[:50]}...' at bbox {text_bbox}")

            # Skip if not caption-like
            is_caption_like = self._is_likely_caption(text_content)
            logger.debug(f"Is caption-like: {is_caption_like}")
            if not is_caption_like:
                continue

            # Calculate spatial relationship
            spatial_analysis = self._analyze_caption_spatial_relationship(image_bbox, text_bbox)

            # Debug: Log spatial analysis results
            logger.debug(f"Spatial analysis for '{text_content[:30]}...': {spatial_analysis}")
            logger.debug(f"Max distances: vertical={max_vertical_distance}, horizontal={max_horizontal_distance}")

            # Check if within acceptable proximity (use proximity_distance for more accurate edge-to-edge measurement)
            proximity_ok = spatial_analysis['proximity_distance'] <= max_vertical_distance
            alignment_ok = spatial_analysis['horizontal_alignment'] > 0.2  # At least 20% overlap

            logger.debug(f"Proximity check: {spatial_analysis['proximity_distance']} <= {max_vertical_distance} = {proximity_ok}")
            logger.debug(f"Alignment check: {spatial_analysis['horizontal_alignment']} > 0.2 = {alignment_ok}")

            if proximity_ok and alignment_ok:

                # Calculate confidence score
                confidence = self._calculate_caption_confidence(
                    text_content, spatial_analysis, image_bbox, text_bbox
                )

                if confidence > 0.01:  # Minimum confidence threshold (lowered for debugging)
                    caption_candidates.append({
                        'block': block,
                        'text': text_content,
                        'confidence': confidence,
                        'spatial_analysis': spatial_analysis
                    })

        # Select best caption candidate based on confidence
        if caption_candidates:
            # Sort by confidence (highest first)
            caption_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            best_caption = caption_candidates[0]

            return {
                'caption_block_id': getattr(best_caption['block'], 'block_id', None),
                'caption_text': best_caption['text'],
                'confidence': best_caption['confidence'],
                'spatial_relationship': best_caption['spatial_analysis']['relationship']
            }

        return None

    def _analyze_caption_spatial_relationship(self, image_bbox, text_bbox):
        """Analyze spatial relationship between image and potential caption"""
        img_left, img_top, img_right, img_bottom = image_bbox
        text_left, text_top, text_right, text_bottom = text_bbox

        # Calculate centers
        img_center_x = (img_left + img_right) / 2
        img_center_y = (img_top + img_bottom) / 2
        text_center_x = (text_left + text_right) / 2
        text_center_y = (text_top + text_bottom) / 2

        # Calculate distances
        vertical_distance = abs(text_center_y - img_center_y)
        horizontal_distance = abs(text_center_x - img_center_x)

        # Determine relationship
        if text_top > img_bottom:  # Text below image
            relationship = 'below'
            proximity_distance = text_top - img_bottom
        elif text_bottom < img_top:  # Text above image
            relationship = 'above'
            proximity_distance = img_top - text_bottom
        elif text_left > img_right:  # Text to the right
            relationship = 'right'
            proximity_distance = text_left - img_right
        elif text_right < img_left:  # Text to the left
            relationship = 'left'
            proximity_distance = img_left - text_right
        else:  # Overlapping
            relationship = 'overlapping'
            proximity_distance = 0

        # Calculate alignment scores
        horizontal_alignment = self._calculate_alignment_score(
            img_left, img_right, text_left, text_right
        )
        vertical_alignment = self._calculate_alignment_score(
            img_top, img_bottom, text_top, text_bottom
        )

        return {
            'relationship': relationship,
            'vertical_distance': vertical_distance,
            'horizontal_distance': horizontal_distance,
            'proximity_distance': proximity_distance,
            'horizontal_alignment': horizontal_alignment,
            'vertical_alignment': vertical_alignment
        }

    def _calculate_alignment_score(self, start1, end1, start2, end2):
        """Calculate alignment score between two ranges (0-1, where 1 is perfect alignment)"""
        overlap = max(0, min(end1, end2) - max(start1, start2))
        range1_size = end1 - start1
        range2_size = end2 - start2

        if range1_size == 0 or range2_size == 0:
            return 0

        # Alignment score based on overlap relative to smaller range
        smaller_range = min(range1_size, range2_size)
        return overlap / smaller_range if smaller_range > 0 else 0

    def _calculate_caption_confidence(self, text_content, spatial_analysis, image_bbox, text_bbox):
        """Calculate confidence score for caption candidate"""
        confidence = 0.0

        # Base confidence from caption pattern matching
        if self._has_explicit_caption_markers(text_content):
            confidence += 0.5
        else:
            confidence += 0.2  # Base score for passing _is_likely_caption

        # Spatial relationship bonus
        relationship = spatial_analysis['relationship']
        if relationship == 'below':
            confidence += 0.3  # Preferred position
        elif relationship == 'above':
            confidence += 0.2
        elif relationship in ['left', 'right']:
            confidence += 0.1

        # Proximity bonus (closer is better)
        proximity = spatial_analysis['proximity_distance']
        if proximity <= 10:
            confidence += 0.2
        elif proximity <= 30:
            confidence += 0.1
        elif proximity <= 50:
            confidence += 0.05

        # Alignment bonus
        if relationship in ['below', 'above']:
            confidence += spatial_analysis['horizontal_alignment'] * 0.2
        else:
            confidence += spatial_analysis['vertical_alignment'] * 0.2

        # Text length penalty for very long text (likely not a caption)
        text_length = len(text_content.strip())
        if text_length > 300:
            confidence *= 0.5
        elif text_length > 200:
            confidence *= 0.8

        return min(1.0, confidence)

    def _has_explicit_caption_markers(self, text):
        """Check for explicit caption markers (stronger indicators)"""
        import re
        text_lower = text.lower().strip()

        explicit_patterns = [
            r'^(figure|fig|table|chart|diagram)\s*\d+',
            r'^(source|credit)[:.]',
            r'^\([a-z]\)',  # (a), (b), etc.
            r'^\d+\.',      # 1., 2., etc.
        ]

        return any(re.search(pattern, text_lower) for pattern in explicit_patterns)

    def _is_likely_caption(self, text):
        """
        Enhanced caption detection with improved pattern recognition and context analysis.

        Addresses detection issues by expanding pattern recognition and adding
        contextual analysis for better accuracy.
        """
        import re

        if not text or len(text.strip()) < 5:
            logger.debug(f"Caption check failed: text too short ({len(text.strip()) if text else 0} chars)")
            return False

        text_lower = text.lower().strip()
        text_clean = text.strip()
        logger.debug(f"Checking caption patterns for: '{text_clean[:50]}...'")  # Debug log

        # Enhanced caption patterns - more comprehensive
        caption_patterns = [
            # Standard figure/table references
            r'^(figure|fig|image|diagram|chart|graph|table|schema|plate|exhibit)\s*\d*[:\.\-\s]',
            r'^(fig\.|figure\.|table\.|chart\.|diagram\.)?\s*\d+[:\.\-\s]',

            # Source and attribution patterns
            r'^(source|credit|copyright|adapted from|modified from)[:.\-\s]',
            r'^(photo|image)\s+(by|from|courtesy)',

            # Parenthetical captions
            r'^\([^)]+\)$',

            # Academic/technical patterns
            r'^(step|phase|stage|example|case)\s*\d*[:\.]',
            r'^(above|below|left|right)[:.\-\s]',

            # Multi-language support
            r'^(figura|tabela|esquema|gráfico)\s*\d*[:\.]',  # Spanish/Portuguese
            r'^(abbildung|tabelle|diagramm)\s*\d*[:\.]',     # German
            r'^(図|表|グラフ)\s*\d*[:\.]',                    # Japanese
        ]

        # Check explicit caption patterns
        for pattern in caption_patterns:
            if re.search(pattern, text_lower):
                return True

        # Enhanced heuristic analysis
        word_count = len(text_clean.split())

        # Short descriptive text without sentence endings
        if (word_count <= 15 and
            not text_clean.endswith('.') and
            not text_clean.endswith('!') and
            not text_clean.endswith('?')):

            # Check for descriptive keywords
            descriptive_keywords = [
                'showing', 'depicting', 'illustrating', 'displaying',
                'overview', 'comparison', 'analysis', 'structure',
                'process', 'workflow', 'system', 'model', 'framework'
            ]

            if any(keyword in text_lower for keyword in descriptive_keywords):
                return True

        # Check for numbered/lettered items (a), (b), (1), (2)
        if re.match(r'^\([a-z0-9]+\)', text_lower):
            return True

        # Check for colon-separated descriptive text
        if ':' in text_clean and word_count <= 20:
            return True

        # Italic or emphasized text patterns (common in captions)
        if (word_count <= 25 and
            ('*' in text_clean or '_' in text_clean or
             text_clean.isupper() or text_clean.islower())):
            return True

        return False

    def _insert_images_by_spatial_order(self, content_blocks, image_placeholders):
        """Insert image placeholders into content blocks based on spatial reading order"""
        if not image_placeholders:
            return content_blocks

        # Combine all blocks and sort by reading order
        all_blocks = content_blocks + image_placeholders

        # Sort by reading order position
        def sort_key(block):
            pos = getattr(block, 'reading_order_position', None)
            return pos if pos is not None else float('inf')

        sorted_blocks = sorted(all_blocks, key=sort_key)

        logger.debug(f"Inserted {len(image_placeholders)} images into reading order")
        return sorted_blocks

    def _extract_content_as_document(self, doc, images_by_page, structure_analysis, filepath):
        """Extract content as a structured Document object"""
        content_blocks = []

        # Extract document title from first page or filename
        document_title = self._extract_document_title(doc) or os.path.splitext(os.path.basename(filepath))[0]

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text blocks with formatting
            page_content_blocks = self._extract_page_content_as_blocks(page, page_num + 1, structure_analysis)
            content_blocks.extend(page_content_blocks)

            # Add images for this page as ImagePlaceholder blocks
            if page_num + 1 in images_by_page:
                page_images = images_by_page[page_num + 1]
                logger.debug(f"Adding {len(page_images)} images for page {page_num + 1}")

                for img_ref in page_images:
                    image_block = ImagePlaceholder(
                        block_type=ContentType.IMAGE_PLACEHOLDER,
                        original_text=img_ref.get('ocr_text', ''),
                        page_num=page_num + 1,
                        bbox=(img_ref['x0'], img_ref['y0'], img_ref['x1'], img_ref['y1']),
                        image_path=img_ref['filepath'],
                        width=img_ref.get('width'),
                        height=img_ref.get('height'),
                        ocr_text=img_ref.get('ocr_text'),
                        translation_needed=img_ref.get('translation_needed', False)
                    )
                    content_blocks.append(image_block)
                    logger.debug(f"Added image block: {img_ref['filename']} at position ({img_ref['x0']}, {img_ref['y0']})")
            else:
                logger.debug(f"No images found for page {page_num + 1}")

        # Improve image placement by associating images with nearby text
        self._associate_images_with_text_blocks(content_blocks)

        # Create and return Document object
        document = Document(
            title=document_title,
            content_blocks=content_blocks,
            source_filepath=filepath,
            total_pages=len(doc),
            metadata={
                'structure_analysis': structure_analysis,
                'extraction_method': 'structured_document_model'
            }
        )

        return document

    def _extract_document_title(self, doc):
        """Extract document title from metadata or first page"""
        try:
            # Try to get title from PDF metadata
            metadata = doc.metadata
            if metadata and metadata.get('title'):
                return metadata['title'].strip()

            # Try to extract from first page
            if len(doc) > 0:
                first_page = doc[0]
                blocks = first_page.get_text("dict")["blocks"]

                for block in blocks[:3]:  # Check first 3 blocks
                    if "lines" not in block:
                        continue

                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span.get("text", "")

                    block_text = block_text.strip()
                    if block_text and len(block_text) < 200:  # Reasonable title length
                        return block_text

        except Exception as e:
            logger.debug(f"Could not extract document title: {e}")

        return None

    def _extract_page_content_as_blocks(self, page, page_num, structure_analysis):
        """
        Extract content from a single page as ContentBlock objects with spatial layout analysis.

        Proposition 1: Implements 2D spatial analysis instead of linear block reading.
        """
        content_blocks = []

        try:
            # Proposition 1: Extract all elements with spatial coordinates first
            all_elements = self._extract_spatial_elements(page, page_num, structure_analysis)

            # Proposition 1: Sort elements by spatial reading order
            ordered_elements = self._apply_spatial_reading_order(all_elements)

            # Convert ordered elements to ContentBlocks
            for element in ordered_elements:
                content_block = self._create_content_block_from_element(element, structure_analysis)
                if content_block:
                    content_blocks.append(content_block)

        except Exception as e:
            logger.warning(f"Error extracting spatial content from page {page_num}: {e}")
            # Fallback to original method
            content_blocks = self._extract_page_content_as_blocks_legacy(page, page_num, structure_analysis)

        return content_blocks

    def _extract_page_content_as_blocks_legacy(self, page, page_num, structure_analysis):
        """Legacy method for extracting content blocks (fallback)"""
        content_blocks = []

        try:
            blocks = page.get_text("dict")["blocks"]

            for block_num, block in enumerate(blocks):
                if "lines" not in block:
                    continue

                # Extract text from block with improved line joining
                block_text = ""
                block_font_sizes = []
                block_formatting = {}

                for line_idx, line in enumerate(block["lines"]):
                    line_text = ""
                    for span in line["spans"]:
                        span_text = span.get("text", "")
                        line_text += span_text

                        # Collect formatting information
                        font_size = span.get("size", 12.0)
                        block_font_sizes.append(font_size)

                        if not block_formatting:
                            block_formatting = {
                                'font': span.get("font", ""),
                                'size': font_size,
                                'flags': span.get("flags", 0),
                                'color': span.get("color", 0)
                            }

                    # Use smart line joining instead of simple concatenation
                    if line_idx == 0:
                        block_text = line_text.strip()
                    else:
                        block_text = self.smart_join_lines(block_text, line_text.strip(), page_num)

                block_text = block_text.strip()
                if not block_text:
                    continue

                # Filter out unwanted content (page numbers, headers, footers)
                if self._should_filter_content(block_text, block.get("bbox", [0, 0, 0, 0]), page_num, structure_analysis):
                    logger.debug(f"Filtered out content: '{block_text[:50]}...'")
                    continue

                # Determine content type and create appropriate ContentBlock
                content_block = self._create_content_block_from_text(
                    block_text,
                    block_formatting,
                    structure_analysis,
                    page_num,
                    block_num,
                    block.get("bbox", [0, 0, 0, 0])
                )

                if content_block:
                    content_blocks.append(content_block)

        except Exception as e:
            logger.warning(f"Error extracting content from page {page_num}: {e}")

        return content_blocks

    def _create_content_block_from_text(self, text, formatting, structure_analysis, page_num, block_num, bbox):
        """Create appropriate ContentBlock based on text analysis"""
        import re

        # Ensure bbox is a tuple of 4 floats
        if isinstance(bbox, list):
            bbox = tuple(bbox)
        if len(bbox) != 4:
            bbox = (0, 0, 0, 0)

        # Classify content type using existing logic
        content_type = self._classify_content_type(text, formatting, structure_analysis)

        # Detect artifacts and metadata
        if self._is_processing_artifact(text):
            return Metadata(
                block_type=ContentType.METADATA,
                original_text=text,
                page_num=page_num,
                bbox=bbox,
                block_num=block_num,
                formatting=formatting,
                content=text,
                metadata_type="artifact"
            )

        # Create specific content block types
        if content_type in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            level = int(content_type[1])
            return Heading(
                block_type=ContentType.HEADING,
                original_text=text,
                page_num=page_num,
                bbox=bbox,
                block_num=block_num,
                formatting=formatting,
                level=level,
                content=text
            )

        elif content_type == 'list_item':
            return ListItem(
                block_type=ContentType.LIST_ITEM,
                original_text=text,
                page_num=page_num,
                bbox=bbox,
                block_num=block_num,
                formatting=formatting,
                content=text
            )

        elif content_type == 'table':
            return Table(
                block_type=ContentType.TABLE,
                original_text=text,
                page_num=page_num,
                bbox=bbox,
                block_num=block_num,
                formatting=formatting,
                markdown_content=text
            )

        elif self._is_code_block(text):
            return CodeBlock(
                block_type=ContentType.CODE_BLOCK,
                original_text=text,
                page_num=page_num,
                bbox=bbox,
                block_num=block_num,
                formatting=formatting,
                content=text
            )

        elif self._is_equation(text):
            return Equation(
                block_type=ContentType.EQUATION,
                original_text=text,
                page_num=page_num,
                bbox=bbox,
                block_num=block_num,
                formatting=formatting,
                content=text
            )

        else:  # Default to paragraph
            return Paragraph(
                block_type=ContentType.PARAGRAPH,
                original_text=text,
                page_num=page_num,
                bbox=bbox,
                block_num=block_num,
                formatting=formatting,
                content=text
            )

    def _is_processing_artifact(self, text):
        """Detect processing artifacts like [MISSING_PAGE], etc."""
        artifact_patterns = [
            r'\[MISSING_PAGE\]',
            r'\[PAGE_BREAK\]',
            r'\[ERROR\]',
            r'\\begin\{.*?\}',
            r'\\end\{.*?\}',
            r'```\s*$',  # Empty code blocks
        ]

        for pattern in artifact_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _is_code_block(self, text):
        """Detect code blocks"""
        code_indicators = [
            r'```',
            r'^\s*def\s+\w+\(',
            r'^\s*class\s+\w+',
            r'^\s*import\s+\w+',
            r'^\s*from\s+\w+\s+import',
            r'^\s*function\s+\w+\(',
            r'^\s*var\s+\w+\s*=',
            r'^\s*\w+\s*=\s*function\(',
        ]

        for pattern in code_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False

    def _is_equation(self, text):
        """Detect mathematical equations"""
        equation_indicators = [
            r'\$.*?\$',  # LaTeX inline math
            r'\\\[.*?\\\]',  # LaTeX display math
            r'\\begin\{equation\}',
            r'\\begin\{align\}',
            r'[∑∏∫∂∇±×÷≤≥≠≈∞]',  # Mathematical symbols
            r'[αβγδεζηθικλμνξοπρστυφχψω]',  # Greek letters
        ]

        for pattern in equation_indicators:
            if re.search(pattern, text):
                return True
        return False

    def _associate_images_with_text_blocks(self, content_blocks):
        """Associate images with nearby text content blocks for better placement"""
        logger.info("🔗 Associating images with nearby text content blocks...")

        # Separate images and text content by page
        pages_content = {}
        images_by_page = {}

        for block in content_blocks:
            page_num = block.page_num

            if isinstance(block, ImagePlaceholder):
                if page_num not in images_by_page:
                    images_by_page[page_num] = []
                images_by_page[page_num].append(block)
            else:
                if page_num not in pages_content:
                    pages_content[page_num] = []
                pages_content[page_num].append(block)

        images_associated = 0

        # Process each page
        for page_num in images_by_page:
            page_images = images_by_page[page_num]
            page_text = pages_content.get(page_num, [])

            if not page_text:
                continue

            # Sort text content by Y position (top to bottom)
            page_text.sort(key=lambda x: x.bbox[1], reverse=True)

            for image in page_images:
                best_text_block = self._find_best_text_block_for_image(image, page_text)

                if best_text_block:
                    # Add image reference to the text block
                    if not hasattr(best_text_block, '_associated_images'):
                        best_text_block._associated_images = []

                    # Calculate relative position for placement decision
                    image_y = image.bbox[1]
                    text_y = best_text_block.bbox[1]

                    placement_info = {
                        'image_block': image,
                        'placement': 'after' if image_y < text_y else 'before',
                        'distance': abs(image_y - text_y)
                    }

                    best_text_block._associated_images.append(placement_info)
                    images_associated += 1

                    logger.debug(f"Associated image {image.image_path} with text on page {page_num}")

        logger.info(f"🔗 Associated {images_associated} images with text content blocks")

    def _find_best_text_block_for_image(self, image_block, page_text_blocks):
        """Find the best text block to associate with an image"""
        if not page_text_blocks:
            return None

        image_center_y = (image_block.bbox[1] + image_block.bbox[3]) / 2

        best_block = None
        min_distance = float('inf')

        for text_block in page_text_blocks:
            # Skip very short text (likely headers/footers)
            if len(text_block.original_text.strip()) < 20:
                continue

            text_center_y = (text_block.bbox[1] + text_block.bbox[3]) / 2

            # Calculate distance between image and text centers
            distance = abs(image_center_y - text_center_y)

            # Prefer text that's close to the image
            if distance < min_distance and distance < 200:  # Within 200 points
                min_distance = distance
                best_block = text_block

        return best_block

    def _extract_content_with_structure(self, doc, images_by_page, structure_analysis):
        """Extract content while preserving document structure"""
        structured_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text blocks with formatting
            page_content = self._extract_page_content(page, page_num + 1, structure_analysis)
            structured_content.extend(page_content)
            
            # Add images for this page with enhanced logging
            if page_num + 1 in images_by_page:
                page_images = images_by_page[page_num + 1]
                logger.debug(f"Adding {len(page_images)} images for page {page_num + 1}")

                for img_ref in page_images:
                    img_item = {
                        'type': 'image',
                        'filename': img_ref['filename'],
                        'filepath': img_ref['filepath'],
                        'page_num': page_num + 1,
                        'position': {
                            'x0': img_ref['x0'],
                            'y0': img_ref['y0'],
                            'x1': img_ref['x1'],
                            'y1': img_ref['y1']
                        },
                        'width': img_ref['width'],
                        'height': img_ref['height']
                    }
                    structured_content.append(img_item)
                    logger.debug(f"Added image item: {img_ref['filename']} at position ({img_ref['x0']}, {img_ref['y0']})")
            else:
                logger.debug(f"No images found for page {page_num + 1}")
        
        # Improve image placement by associating images with nearby text
        self._associate_images_with_text(structured_content)

        return structured_content

    def _associate_images_with_text(self, structured_content):
        """Associate images with nearby text content for better placement"""
        logger.info("🔗 Associating images with nearby text content...")

        # Separate images and text content by page
        pages_content = {}
        images_by_page = {}

        for item in structured_content:
            page_num = item.get('page_num', 1)

            if item.get('type') == 'image':
                if page_num not in images_by_page:
                    images_by_page[page_num] = []
                images_by_page[page_num].append(item)
            else:
                if page_num not in pages_content:
                    pages_content[page_num] = []
                pages_content[page_num].append(item)

        images_associated = 0

        # Process each page
        for page_num in images_by_page:
            page_images = images_by_page[page_num]
            page_text = pages_content.get(page_num, [])

            if not page_text:
                continue

            # Sort text content by Y position (top to bottom)
            page_text.sort(key=lambda x: x.get('bbox', [0, 0, 0, 0])[1], reverse=True)

            for image in page_images:
                best_text_item = self._find_best_text_for_image(image, page_text)

                if best_text_item:
                    # Add image reference to the text item
                    if '_associated_images' not in best_text_item:
                        best_text_item['_associated_images'] = []

                    # Calculate relative position for placement decision
                    image_y = image.get('position', {}).get('y0', 0)
                    text_y = best_text_item.get('bbox', [0, 0, 0, 0])[1]

                    placement_info = {
                        'image_item': image,
                        'placement': 'after' if image_y < text_y else 'before',
                        'distance': abs(image_y - text_y)
                    }

                    best_text_item['_associated_images'].append(placement_info)
                    images_associated += 1

                    logger.debug(f"Associated image {image.get('filename')} with text on page {page_num}")

        logger.info(f"🔗 Associated {images_associated} images with text content")

    def _find_best_text_for_image(self, image, page_text_items):
        """Find the best text item to associate with an image"""
        if not page_text_items:
            return None

        image_position = image.get('position', {})
        image_y = image_position.get('y0', 0)
        image_center_y = (image_position.get('y0', 0) + image_position.get('y1', 0)) / 2

        best_item = None
        min_distance = float('inf')

        for text_item in page_text_items:
            # Skip very short text (likely headers/footers)
            text = text_item.get('text', '')
            if len(text.strip()) < 20:
                continue

            text_bbox = text_item.get('bbox', [0, 0, 0, 0])
            text_y = text_bbox[1]
            text_center_y = (text_bbox[1] + text_bbox[3]) / 2

            # Calculate distance between image and text centers
            distance = abs(image_center_y - text_center_y)

            # Prefer text that's close to the image
            if distance < min_distance and distance < 200:  # Within 200 points
                min_distance = distance
                best_item = text_item

        return best_item
    
    def smart_join_lines(self, previous_text, new_line, page_num=0):
        """Enhanced line joining that handles hyphenation and natural sentence flow with special handling for first pages"""
        if not previous_text or not new_line:
            return (previous_text or "") + " " + (new_line or "")

        # Special handling for first pages (0-2) where premature line endings are more common
        is_early_page = page_num <= 2

        # Check if previous line ends with a hyphen (word continuation)
        if previous_text.endswith('-'):
            # Remove hyphen and join directly (dehyphenation)
            return previous_text[:-1] + new_line

        # Check if previous line ends with sentence-ending punctuation
        if previous_text.rstrip().endswith(('.', '!', '?', ':', ';')):
            # Start new sentence with proper spacing
            return previous_text + " " + new_line

        # Enhanced logic for early pages
        if is_early_page:
            # More aggressive joining for early pages
            # Check if previous line seems incomplete (doesn't end with punctuation and is reasonably long)
            prev_stripped = previous_text.rstrip()
            if (len(prev_stripped) > 20 and
                not prev_stripped.endswith(('.', '!', '?', ':', ';', ',')) and
                not prev_stripped.endswith((')', ']', '"', "'")) and
                new_line and new_line[0].islower()):
                return previous_text + " " + new_line

            # Check for common continuation patterns in early pages
            if (prev_stripped.endswith(('και', 'ή', 'αλλά', 'όμως', 'επίσης', 'ωστόσο', 'παρόλα', 'μετά')) or
                new_line.startswith(('που', 'όπου', 'όταν', 'αν', 'επειδή', 'καθώς', 'ενώ'))):
                return previous_text + " " + new_line

        # Check if new line starts with a capital letter (likely new sentence)
        if new_line and new_line[0].isupper():
            # Check if previous line seems incomplete (no sentence-ending punctuation)
            # and is not too short (avoid joining headings)
            if len(previous_text.split()) > 3 and not previous_text.rstrip().endswith(('.', '!', '?', ':', ';')):
                # Likely continuation of sentence
                return previous_text + " " + new_line
            else:
                # Likely new sentence
                return previous_text + " " + new_line

        # Check if new line starts with lowercase (likely continuation)
        if new_line and new_line[0].islower():
            # Continuation of previous sentence
            return previous_text + " " + new_line

        # Check for incomplete sentences (no ending punctuation and reasonable length)
        prev_stripped = previous_text.rstrip()
        if (len(prev_stripped) > 15 and
            not prev_stripped.endswith(('.', '!', '?', ':', ';')) and
            len(new_line.split()) <= 10):  # Short continuation
            return previous_text + " " + new_line

        # Default: join with space
        return previous_text + " " + new_line

    def _extract_page_content(self, page, page_num, structure_analysis):
        """Extract content from a single page with structure detection"""
        page_content = []

        try:
            blocks = page.get_text("dict")["blocks"]

            for block_num, block in enumerate(blocks):
                if "lines" not in block:
                    continue

                # Extract text from block with improved line joining
                block_text = ""
                block_font_sizes = []
                block_formatting = {}

                for line_idx, line in enumerate(block["lines"]):
                    line_text = ""
                    for span in line["spans"]:
                        span_text = span.get("text", "")
                        line_text += span_text

                        # Collect formatting information
                        font_size = span.get("size", 12.0)
                        block_font_sizes.append(font_size)

                        if not block_formatting:
                            block_formatting = {
                                'font': span.get("font", ""),
                                'size': font_size,
                                'flags': span.get("flags", 0),
                                'color': span.get("color", 0)
                            }

                    # Use smart line joining instead of simple concatenation
                    if line_idx == 0:
                        block_text = line_text.strip()
                    else:
                        block_text = self.smart_join_lines(block_text, line_text.strip(), page_num)
                
                block_text = block_text.strip()
                if not block_text:
                    continue

                # Filter out unwanted content (page numbers, headers, footers)
                if self._should_filter_content(block_text, block.get("bbox", [0, 0, 0, 0]), page_num, structure_analysis):
                    logger.debug(f"Filtered out content: '{block_text[:50]}...'")
                    continue

                # Determine content type
                content_type = self._classify_content_type(
                    block_text,
                    block_formatting,
                    structure_analysis
                )

                # Create content item
                content_item = {
                    'type': content_type,
                    'text': block_text,
                    'page_num': page_num,
                    'block_num': block_num,
                    'formatting': block_formatting,
                    'bbox': block.get("bbox", [0, 0, 0, 0])
                }

                page_content.append(content_item)
        
        except Exception as e:
            logger.warning(f"Error extracting content from page {page_num}: {e}")
        
        return page_content

    def _should_filter_content(self, text, bbox, page_num, structure_analysis):
        """Determine if content should be filtered out (page numbers, headers, footers, etc.)"""
        import re

        # Get page dimensions for position analysis
        page_height = 792  # Standard PDF page height in points
        page_width = 612   # Standard PDF page width in points

        # Extract bbox coordinates
        x0, y0, x1, y1 = bbox

        # Calculate relative positions
        relative_y_top = y0 / page_height
        relative_y_bottom = y1 / page_height
        relative_x_center = (x0 + x1) / 2 / page_width

        # Filter criteria
        text_lower = text.lower().strip()
        text_length = len(text.strip())

        # 1. Filter very short text (likely page numbers or artifacts)
        if text_length < 3:
            return True

        # 2. Filter standalone numbers (likely page numbers)
        if re.match(r'^\d+$', text.strip()):
            return True

        # 3. Filter page number patterns (enhanced)
        page_number_patterns = [
            r'^\d+\s*$',  # Just a number
            r'^page\s+\d+$',  # "Page 1"
            r'^\d+\s*/\s*\d+$',  # "1/10"
            r'^\d+\s+of\s+\d+$',  # "1 of 10"
            r'^\d+\s*-\s*\d+$',  # "1-10"
            r'^\s*\d+\s*\n',  # Numbers at line start/end
            r'\n\s*\d+\s*$',  # Numbers at line end
            r'^\s*\d{1,3}\s*$',  # Standalone 1-3 digit numbers (page numbers)
        ]

        for pattern in page_number_patterns:
            if re.search(pattern, text_lower, re.MULTILINE):
                return True

        # Additional check for embedded page numbers in text flow
        # Look for isolated numbers that appear to be page numbers
        words = text.strip().split()
        if len(words) == 1 and words[0].isdigit() and 1 <= int(words[0]) <= 9999:
            return True

        # 4. Filter headers (top 10% of page)
        if relative_y_top > 0.9:
            # Common header patterns
            header_patterns = [
                r'^chapter\s+\d+',
                r'^section\s+\d+',
                r'^\d+\.\d+',  # Section numbers like "1.1"
                r'^[A-Z\s]+$',  # All caps (likely headers)
            ]

            for pattern in header_patterns:
                if re.match(pattern, text_lower) or (text_length < 50 and text.isupper()):
                    return True

        # 5. Filter footers (bottom 10% of page)
        if relative_y_bottom < 0.1:
            # Common footer patterns
            footer_patterns = [
                r'copyright',
                r'©',
                r'all rights reserved',
                r'confidential',
                r'draft',
                r'version\s+\d+',
                r'rev\s+\d+',
                r'www\.',
                r'http',
                r'@',
            ]

            for pattern in footer_patterns:
                if pattern in text_lower:
                    return True

            # Filter short text in footer area (likely page numbers or dates)
            if text_length < 30:
                return True

        # 6. Filter common document metadata
        metadata_patterns = [
            r'^document\s+title',
            r'^author:',
            r'^date:',
            r'^version:',
            r'^revision:',
            r'^draft',
            r'^confidential',
            r'^internal\s+use',
            r'^proprietary',
        ]

        for pattern in metadata_patterns:
            if re.match(pattern, text_lower):
                return True

        # 7. Filter repeated headers/footers across pages
        # This would require tracking content across pages - implement if needed

        return False

    def _classify_content_type(self, text, formatting, structure_analysis):
        """Classify content type with enhanced detection and length-based filtering"""
        import re

        font_size = formatting.get('size', 12.0)
        dominant_size = structure_analysis['dominant_font_size']
        text_lower = text.lower().strip()
        text_length = len(text.strip())
        word_count = len(text.strip().split())

        # Length-based filtering: Long text should not be headings.
        # This prevents paragraph fragments from being classified as headings.
        # Adjusted word_count from 13 to 12 (original was 15) for more conservative heading detection.
        if word_count > 12: # Max words for a heading
            return 'paragraph'

        # Adjusted text_length from 100 to 90 for more conservative heading detection.
        if text_length > 90: # Max characters for a heading
            return 'paragraph'

        # Check for list items first (before heading detection)
        list_patterns = [
            r'^\s*[•\-\*]\s+',  # Bullet points
            r'^\s*\d+[\.\)]\s+',  # Numbered lists
            r'^\s*[a-zA-Z][\.\)]\s+',  # Lettered lists
        ]

        for pattern in list_patterns:
            if re.match(pattern, text):
                return 'list_item'

        # Check for chapter titles (semantic detection)
        chapter_patterns = [
            r'^chapter\s+\d+',
            r'^section\s+\d+',
            r'^part\s+\d+',
            r'^\d+\.\s+[A-Z]',  # "1. Introduction"
            r'^[IVX]+\.\s+[A-Z]',  # Roman numerals
        ]

        for pattern in chapter_patterns:
            if re.match(pattern, text_lower):
                return 'h1'

        # Check for section numbering patterns (even with normal font size)
        section_patterns = [
            r'^\d+\.\d+\s+',  # "1.1 "
            r'^\d+\.\d+\.\d+\s+',  # "1.1.1 "
        ]

        for pattern in section_patterns:
            if re.match(pattern, text):
                return 'h3' if text.count('.') > 1 else 'h2'

        # Enhanced heading detection based on font size and content
        size_ratio = font_size / dominant_size

        # Check if text has heading characteristics first
        is_likely_heading = self._is_likely_heading(text, text_lower, word_count)

        classified_level = None
        log_reason = []

        # More conservative font size thresholds, but allow semantic detection
        if size_ratio > 1.4:  # Significantly larger than normal text
            if is_likely_heading:
                log_reason.append(f"SizeRatio > 1.4 ({size_ratio:.2f}) + LikelyH")
                if size_ratio > 1.8 or any(keyword in text_lower for keyword in ['chapter', 'part', 'introduction', 'conclusion']):
                    classified_level = 'h1'
                    log_reason.append("StrongerRatio/Keyword for H1")
                elif size_ratio > 1.5 or any(keyword in text_lower for keyword in ['section', 'method', 'result', 'discussion']):
                    classified_level = 'h2'
                    log_reason.append("MidRatio/Keyword for H2")
                else:
                    classified_level = 'h3'
                    log_reason.append("Default for SizeRatio > 1.4 + LikelyH")

        # Medium size increase with strong semantic indicators
        elif size_ratio > 1.2:  # Moderately larger
            if is_likely_heading:
                log_reason.append(f"SizeRatio > 1.2 ({size_ratio:.2f}) + LikelyH")
                if any(keyword in text_lower for keyword in ['introduction', 'conclusion', 'methodology', 'results', 'discussion', 'background', 'summary']):
                    classified_level = 'h2'
                    log_reason.append("Keyword for H2")
                elif word_count <= 3 and (text.istitle() or text.isupper()):  # Very short, formatted text
                    classified_level = 'h3'
                    log_reason.append("Short Title/Upper for H3")

        # Even normal size text can be heading with strong semantic indicators
        elif is_likely_heading and word_count <= 5: # Ensure it's short
            log_reason.append(f"LikelyH + WC<=5 ({word_count})")
            if any(keyword in text_lower for keyword in ['introduction', 'conclusion', 'methodology', 'results', 'discussion', 'background', 'summary', 'abstract', 'overview']):
                classified_level = 'h2'
                log_reason.append("Keyword for H2")

        if classified_level:
            logger.debug(f"[Legacy Classifier] Classified as {classified_level}: '{text[:100]}...'. Reason: {'; '.join(log_reason)}. Features: SizeRatio: {size_ratio:.2f}, LikelyH: {is_likely_heading}, WC: {word_count}, Len: {text_length}, FontSz: {font_size:.1f}")
            return classified_level

        # Default to paragraph
        # Log borderline cases for paragraphs if they had some heading characteristics
        if is_likely_heading or size_ratio > 1.1: # Potential borderline case
            logger.debug(f"[Legacy Classifier] Borderline paragraph: '{text[:100]}...'. SizeRatio: {size_ratio:.2f}, LikelyH: {is_likely_heading}, WC: {word_count}, Len: {text_length}, FontSz: {font_size:.1f}")
        return 'paragraph'

    def _is_likely_heading(self, text, text_lower, word_count):
        """Determine if text has characteristics of a heading (used in older _classify_content_type)"""
        # Short text is more likely to be a heading.
        # Adjusted word_count from 10 to 8 for more conservative heading detection.
        if word_count > 8: # Max words for a heading if not matching other strong criteria
            return False

        # Title case or all caps suggests heading.
        # Added word_count constraint (<=5) to avoid classifying short, capitalized sentences
        # (common in paragraphs) as headings.
        if (text.istitle() or text.isupper()) and word_count <= 5:
            logger.debug(f"Likely heading (title/upper, short): '{text[:100]}...'")
            return True

        # Common heading words
        heading_keywords = [
            'introduction', 'conclusion', 'methodology', 'results', 'discussion',
            'background', 'summary', 'abstract', 'overview', 'analysis',
            'findings', 'implications', 'recommendations', 'future', 'work'
        ]

        for keyword in heading_keywords:
            if keyword in text_lower:
                logger.debug(f"Likely heading (keyword: {keyword}): '{text[:100]}...'")
                return True

        # Ends with colon (often indicates section start)
        if text.strip().endswith(':'):
            logger.debug(f"Likely heading (ends with colon): '{text[:100]}...'")
            return True

        return False

    def extract_toc_from_content(self, structured_content):
        """Extract and enhance TOC by analyzing both automatic TOC and content text"""
        logger.info("🔍 Enhanced TOC extraction from content analysis...")

        toc_entries = []
        content_toc_entries = []

        # Look for TOC-like content in the text
        for i, item in enumerate(structured_content):
            text = item.get('text', '').strip()
            if not text:
                continue

            # Check if this looks like a TOC entry (contains dots and numbers)
            import re
            # Pattern for TOC entries like "Chapter 1 .................. 15"
            toc_pattern = r'^(.+?)\s*\.{3,}\s*(\d+)\s*$'
            match = re.match(toc_pattern, text)

            if match:
                title = match.group(1).strip()
                page_num = int(match.group(2))
                content_toc_entries.append({
                    'title': title,
                    'page': page_num,
                    'level': 1,  # Default level, can be refined
                    'source': 'content_analysis'
                })
                logger.debug(f"Found TOC entry from content: '{title}' -> page {page_num}")

            # Also check for numbered headings that might be TOC entries
            numbered_heading_pattern = r'^(\d+\.?\d*\.?\s+)(.+)$'
            numbered_match = re.match(numbered_heading_pattern, text)
            if numbered_match and len(text.split()) <= 10:  # Short enough to be a heading
                number_part = numbered_match.group(1).strip()
                title_part = numbered_match.group(2).strip()

                # Determine level based on numbering
                level = number_part.count('.') + 1 if '.' in number_part else 1

                content_toc_entries.append({
                    'title': f"{number_part} {title_part}",
                    'page': None,  # Page number not available from content
                    'level': level,
                    'source': 'numbered_heading'
                })

        # Combine with headings found in the document structure
        for item in structured_content:
            if item.get('type') in ['h1', 'h2', 'h3']:
                level = int(item['type'][1:])  # Extract level from h1, h2, h3
                toc_entries.append({
                    'title': item.get('text', '').strip(),
                    'page': item.get('page_num'),
                    'level': level,
                    'source': 'structure_analysis'
                })

        # Merge and deduplicate entries
        all_entries = content_toc_entries + toc_entries
        unique_entries = []
        seen_titles = set()

        for entry in all_entries:
            title_normalized = entry['title'].lower().strip()
            if title_normalized not in seen_titles and len(title_normalized) > 3:
                seen_titles.add(title_normalized)
                unique_entries.append(entry)

        logger.info(f"Enhanced TOC extraction found {len(unique_entries)} unique entries")
        return unique_entries

    def _is_likely_page_number(self, text):
        """Check if text is likely a page number"""
        text = text.strip()
        
        # Skip empty text
        if not text:
            return False
            
        # Check if it's just a number
        if text.isdigit():
            # Check if it's a reasonable page number (1-9999)
            try:
                num = int(text)
                if 1 <= num <= 9999:
                    return True
            except ValueError:
                pass
                
        # Check for common page number patterns
        page_patterns = [
            r'^\d+$',  # Just a number
            r'^page\s+\d+$',  # "page X"
            r'^p\.?\s*\d+$',  # "p. X" or "p X"
            r'^\d+\s*of\s*\d+$',  # "X of Y"
            r'^-\s*\d+\s*-$'  # "- X -"
        ]
        
        import re
        return any(re.match(pattern, text.lower()) for pattern in page_patterns)

    def extract_text(self, page_num):
        """Extract text from page while excluding visual content areas"""
        try:
            page = self.pdf_document[page_num]
            visual_areas = self._simple_visual_detection(page, page_num)
            
            # Get all text blocks
            text_blocks = page.get_text("dict")["blocks"]
            text_blocks = [b for b in text_blocks if "lines" in b]
            
            # Filter out text blocks that overlap with visual areas
            filtered_blocks = []
            for block in text_blocks:
                block_bbox = block["bbox"]
                should_exclude = False
                
                for visual_area in visual_areas:
                    if self._bboxes_overlap(block_bbox, visual_area['bbox']):
                        should_exclude = True
                        logger.debug(f"Excluding text block from page {page_num + 1} due to overlap with visual content")
                        break
                
                if not should_exclude:
                    filtered_blocks.append(block)
            
            # Extract text from filtered blocks
            text = ""
            for block in filtered_blocks:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text += span["text"] + " "
                    text += "\n"
                text += "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text with exclusions: {e}")
            return ""

    def _bboxes_overlap(self, bbox1, bbox2):
        """Check if two bounding boxes overlap"""
        return not (bbox1[2] < bbox2[0] or  # bbox1 is left of bbox2
                   bbox1[0] > bbox2[2] or  # bbox1 is right of bbox2
                   bbox1[3] < bbox2[1] or  # bbox1 is above bbox2
                   bbox1[1] > bbox2[3])    # bbox1 is below bbox2

    def _simple_visual_detection(self, page, page_num):
        """Simple and effective visual detection - identify areas to exclude from translation"""
        visual_areas = []

        try:
            page_rect = page.rect

            # Method 1: Check for vector drawings
            drawings = page.get_drawings()
            if len(drawings) >= 3:  # Lower threshold since we're excluding these areas
                # Create a large area covering most of the page
                padding = 30
                bbox = [
                    padding,
                    padding,
                    page_rect.width - padding,
                    page_rect.height - padding
                ]

                visual_areas.append({
                    'bbox': bbox,
                    'content_type': 'page_with_drawings',
                    'exclude_from_translation': True,  # Flag to exclude from translation
                    'confidence': 0.9
                })
                logger.debug(f"Page {page_num + 1}: Found {len(drawings)} drawings, excluding from translation")

            # Method 2: Check for raster images
            images = page.get_images(full=True)
            if images:
                for img in images:
                    try:
                        xref = img[0]
                        img_rects = page.get_image_rects(xref)

                        if img_rects:
                            for rect in img_rects:
                                # Create generous area around image
                                padding = 150  # Very generous padding
                                bbox = [
                                    max(0, rect.x0 - padding),
                                    max(0, rect.y0 - padding),
                                    min(page_rect.width, rect.x1 + padding),
                                    min(page_rect.height, rect.y1 + padding)
                                ]

                                visual_areas.append({
                                    'bbox': bbox,
                                    'content_type': 'image_area',
                                    'exclude_from_translation': True,  # Flag to exclude from translation
                                    'confidence': 0.95
                                })
                                logger.debug(f"Page {page_num + 1}: Found raster image, excluding from translation")
                    except:
                        continue

            # Method 3: Check for pages with low text density (might have visual content)
            if not visual_areas:  # Only if we haven't found anything else
                text_blocks = page.get_text("dict")["blocks"]
                text_blocks = [b for b in text_blocks if "lines" in b]

                if text_blocks:
                    total_text_area = 0
                    for block in text_blocks:
                        bbox = block["bbox"]
                        block_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        total_text_area += block_area

                    page_area = page_rect.width * page_rect.height
                    text_coverage = total_text_area / page_area

                    # If very low text coverage, might have visual content
                    if text_coverage < 0.2:  # Less than 20% text
                        padding = 50
                        bbox = [
                            padding,
                            padding,
                            page_rect.width - padding,
                            page_rect.height - padding
                        ]

                        visual_areas.append({
                            'bbox': bbox,
                            'content_type': 'sparse_text_page',
                            'exclude_from_translation': True,  # Flag to exclude from translation
                            'confidence': 0.8
                        })
                        logger.debug(f"Page {page_num + 1}: Low text coverage ({text_coverage:.2f}), excluding from translation")

        except Exception as e:
            logger.warning(f"Error in simple visual detection: {e}")

        return visual_areas

    def extract_toc_from_content_two_pass(self, doc, structure_info=None):
        """
        Robust two-pass ToC extraction:
        1. Extract from explicit ToC pages and heading structure.
        2. Reconcile, deduplicate, and organize hierarchically.
        Returns a list of dicts: {title, page, level, source}
        """
        import re
        if structure_info is None:
            structure_info = self.detect_document_structure(doc)

        toc_entries = []
        toc_page_numbers = structure_info.get('toc_pages', [])
        heading_hierarchy = structure_info.get('heading_hierarchy', {})
        section_boundaries = structure_info.get('section_boundaries', [])

        # --- Pass 1: Extract candidates ---
        # 1A. From explicit ToC pages
        for page_num in toc_page_numbers:
            page = doc[page_num]
            page_text = page.get_text()
            lines = page_text.split('\n')
            for line in lines:
                # Pattern: "Title .......... 5"
                m = re.match(r'^(.+?)\s*\.{3,}\s*(\d+)$', line.strip())
                if m:
                    title = m.group(1).strip()
                    page_ref = int(m.group(2))
                    toc_entries.append({
                        'title': title,
                        'page': page_ref,
                        'level': 1,  # Default, will refine in pass 2
                        'source': 'toc_page'
                    })
                # Pattern: "1.1 Section Title   7"
                m2 = re.match(r'^(\d+(?:\.\d+)*)(?:\s+)(.+?)\s+(\d+)$', line.strip())
                if m2:
                    number = m2.group(1)
                    title = m2.group(2).strip()
                    page_ref = int(m2.group(3))
                    level = number.count('.') + 1
                    toc_entries.append({
                        'title': f"{number} {title}",
                        'page': page_ref,
                        'level': level,
                        'source': 'toc_page_numbered'
                    })

        # 1B. From heading structure (section boundaries)
        for section in section_boundaries:
            heading = section.get('heading')
            page = section.get('start_page', None)
            level = section.get('level', 1)
            if heading and page is not None:
                toc_entries.append({
                    'title': heading,
                    'page': page + 1,  # Convert to 1-based page
                    'level': level,
                    'source': 'heading_structure'
                })

        # --- Pass 2: Reconcile, deduplicate, organize ---
        # Deduplicate by (title, page)
        seen = set()
        deduped = []
        for entry in toc_entries:
            key = (entry['title'].lower().strip(), entry['page'])
            if key not in seen:
                seen.add(key)
                deduped.append(entry)

        # Organize hierarchically (sort by page, then level)
        deduped.sort(key=lambda x: (x['page'], x['level']))

        # Optionally, build parent-child relationships (not required for flat list)
        # For now, just output the clean list
        return deduped

    def _clean_text_content(self, text: str) -> str:
        """
        Clean text content by removing footnotes and properly handling TOC bookmarks.
        """
        # Remove footnotes (assuming they are numbered and at the end of lines)
        text = re.sub(r'\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove footnote references in text (e.g., [1], [2], etc.)
        text = re.sub(r'\[\d+\]', '', text)
        
        # Handle TOC bookmarks - ensure there's a space between bookmark and text
        # First, handle bookmarks at the start of text
        text = re.sub(r'^(_Toc_Bookmark_\d+)([^\s])', r'\1 \2', text)
        # Then handle bookmarks in the middle of text
        text = re.sub(r'([^\s])(_Toc_Bookmark_\d+)([^\s])', r'\1 \2 \3', text)
        # Finally handle bookmarks at the end of text
        text = re.sub(r'([^\s])(_Toc_Bookmark_\d+)$', r'\1 \2', text)
        
        return text.strip()

    def _extract_page_content(self, page, page_num, structure_analysis):
        """Extract content from a page with enhanced cleaning"""
        try:
            # Get text blocks
            blocks = page.get_text("dict")["blocks"]
            content_blocks = []
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                # Clean the text content
                                text = self._clean_text_content(text)
                                if text:  # Only add if there's content after cleaning
                                    content_blocks.append({
                                        'text': text,
                                        'bbox': span["bbox"],
                                        'font': span["font"],
                                        'size': span["size"],
                                        'flags': span["flags"]
                                    })
        
            return content_blocks
        except Exception as e:
            logger.error(f"Error extracting content from page {page_num}: {e}")
            return []
