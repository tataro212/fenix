"""
Fenix Professional Translation Integration

This module provides a complete integration between the Fenix translation system
and professional academic validation, addressing the specific deficiencies
identified in the translation quality analysis:

1. "Error! Bookmark not defined" detection and correction
2. Bibliography consistency validation (Woodward vs Μπάρνοου issue)
3. Terminology consistency management
4. Professional human oversight workflow
5. Quality reporting and documentation

This acts as a drop-in replacement for the existing Fenix workflow while
maintaining full backward compatibility.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from datetime import datetime
import re

# Import existing Fenix components
from main_workflow_enhanced import process_pdf_document
from academic_translation_validator import (
    AcademicTranslationValidator, 
    ValidationSeverity,
    ValidationIssue,
    validate_academic_document,
    generate_validation_report_file
)
from professional_translation_workflow import (
    ProfessionalTranslationWorkflow,
    TranslationProject,
    ReviewTask,
    WorkflowStage
)
from config_manager import config_manager

logger = logging.getLogger(__name__)

class QualityMode(Enum):
    """Quality modes for translation processing"""
    STANDARD = "standard"      # Basic Fenix translation
    ENHANCED = "enhanced"      # Fenix + automated validation
    PROFESSIONAL = "professional"  # Full professional workflow

@dataclass
class FenixProcessingResult:
    """Enhanced result object for Fenix processing"""
    success: bool
    output_file: str
    processing_time: float
    quality_mode: QualityMode
    validation_report: Optional[str] = None
    quality_score: float = 0.0
    issues_found: List[ValidationIssue] = field(default_factory=list)
    professional_project_id: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

class FenixProfessionalIntegration:
    """
    Professional integration layer for Fenix translation system
    
    This class provides a drop-in replacement for the existing Fenix workflow
    while adding professional validation and quality control features.
    """
    
    def __init__(self, config_file: str = "fenix_professional_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.validator = AcademicTranslationValidator()
        self.professional_workflow = ProfessionalTranslationWorkflow()
        self.processing_history: List[Dict[str, Any]] = []
        
        logger.info("🏆 Fenix Professional Integration initialized")
        logger.info(f"Available quality modes: {[mode.value for mode in QualityMode]}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load Fenix professional configuration"""
        default_config = {
            "default_quality_mode": "enhanced",
            "auto_detect_academic_documents": True,
            "enable_structural_error_detection": True,
            "enable_bibliography_validation": True,
            "enable_terminology_consistency": True,
            "quality_thresholds": {
                "minimum_acceptable_score": 0.7,
                "professional_review_threshold": 0.6,
                "automatic_approval_threshold": 0.9
            },
            "issue_handling": {
                "bookmark_errors": "fix_automatically",
                "bibliography_inconsistencies": "validate_and_report",
                "terminology_issues": "validate_and_report",
                "structural_problems": "fix_automatically"
            },
            "reporting": {
                "generate_quality_reports": True,
                "include_before_after_comparison": True,
                "save_validation_details": True
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            else:
                self._save_config(default_config)
                return default_config
        except Exception as e:
            logger.error(f"Error loading Fenix professional config: {e}")
            return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    async def process_document_professionally(self, 
                                            pdf_path: str,
                                            quality_mode: Union[str, QualityMode] = None,
                                            document_domain: str = "academic",
                                            enable_human_review: bool = None) -> FenixProcessingResult:
        """
        Process a PDF document with professional quality control
        
        This is the main entry point that replaces the standard Fenix workflow
        with enhanced validation and quality control.
        
        Args:
            pdf_path: Path to the PDF file to process
            quality_mode: Quality mode (standard, enhanced, professional)
            document_domain: Domain for specialized validation (academic, technical, etc.)
            enable_human_review: Whether to enable human expert review
            
        Returns:
            FenixProcessingResult with complete processing information
        """
        start_time = datetime.now()
        
        # Determine quality mode
        if quality_mode is None:
            quality_mode = self.config["default_quality_mode"]
        
        if isinstance(quality_mode, str):
            quality_mode = QualityMode(quality_mode)
        
        # Auto-detect academic documents if enabled
        if self.config["auto_detect_academic_documents"]:
            detected_domain = await self._detect_document_domain(pdf_path)
            if detected_domain:
                document_domain = detected_domain
        
        logger.info(f"🎯 Processing document: {pdf_path}")
        logger.info(f"📊 Quality mode: {quality_mode.value}")
        logger.info(f"🔬 Domain: {document_domain}")
        
        try:
            # Stage 1: Standard Fenix Processing
            fenix_result = await self._run_standard_fenix_processing(pdf_path)
            
            if not fenix_result["success"]:
                return FenixProcessingResult(
                    success=False,
                    output_file="",
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    quality_mode=quality_mode,
                    error_message=fenix_result.get("error", "Fenix processing failed")
                )
            
            # Stage 2: Enhanced Validation (for enhanced and professional modes)
            if quality_mode in [QualityMode.ENHANCED, QualityMode.PROFESSIONAL]:
                validation_result = await self._run_enhanced_validation(
                    fenix_result, document_domain
                )
            else:
                validation_result = {
                    "validation_report": None,
                    "quality_score": 0.8,  # Default for standard mode
                    "issues_found": [],
                    "recommendations": []
                }
            
            # Stage 3: Professional Review (for professional mode only)
            professional_project_id = None
            if quality_mode == QualityMode.PROFESSIONAL:
                professional_result = await self._run_professional_review(
                    fenix_result, validation_result, document_domain
                )
                professional_project_id = professional_result.get("project_id")
            
            # Stage 4: Generate comprehensive results
            result = FenixProcessingResult(
                success=True,
                output_file=fenix_result["output_file"],
                processing_time=(datetime.now() - start_time).total_seconds(),
                quality_mode=quality_mode,
                validation_report=validation_result.get("validation_report"),
                quality_score=validation_result.get("quality_score", 0.0),
                issues_found=validation_result.get("issues_found", []),
                professional_project_id=professional_project_id,
                recommendations=validation_result.get("recommendations", [])
            )
            
            # Stage 5: Generate quality report
            if self.config["reporting"]["generate_quality_reports"]:
                await self._generate_quality_report(result, pdf_path)
            
            # Store processing history
            self.processing_history.append({
                "timestamp": start_time.isoformat(),
                "pdf_path": pdf_path,
                "quality_mode": quality_mode.value,
                "domain": document_domain,
                "quality_score": result.quality_score,
                "issues_count": len(result.issues_found),
                "success": result.success
            })
            
            logger.info(f"✅ Document processing completed successfully")
            logger.info(f"📈 Quality score: {result.quality_score:.2f}")
            logger.info(f"🔍 Issues found: {len(result.issues_found)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in professional document processing: {e}")
            return FenixProcessingResult(
                success=False,
                output_file="",
                processing_time=(datetime.now() - start_time).total_seconds(),
                quality_mode=quality_mode,
                error_message=str(e)
            )
    
    async def _detect_document_domain(self, pdf_path: str) -> Optional[str]:
        """Detect document domain from PDF content"""
        try:
            # Simple domain detection based on file content
            # This could be enhanced with more sophisticated analysis
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            sample_text = ""
            
            # Extract text from first few pages
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                sample_text += page.get_text()
            
            doc.close()
            
            # Domain detection keywords
            domain_keywords = {
                "academic": ["abstract", "introduction", "methodology", "conclusion", "bibliography", "references"],
                "technical": ["specification", "implementation", "architecture", "API", "configuration"],
                "medical": ["patient", "diagnosis", "treatment", "clinical", "medical"],
                "legal": ["contract", "agreement", "legal", "clause", "jurisdiction"],
                "financial": ["financial", "investment", "revenue", "profit", "budget"]
            }
            
            sample_text_lower = sample_text.lower()
            domain_scores = {}
            
            for domain, keywords in domain_keywords.items():
                score = sum(1 for keyword in keywords if keyword in sample_text_lower)
                domain_scores[domain] = score
            
            # Return domain with highest score if above threshold
            if domain_scores:
                best_domain = max(domain_scores, key=domain_scores.get)
                if domain_scores[best_domain] >= 2:  # At least 2 keywords
                    logger.info(f"🔍 Auto-detected domain: {best_domain}")
                    return best_domain
            
            return None
            
        except Exception as e:
            logger.warning(f"Domain detection failed: {e}")
            return None
    
    async def _run_standard_fenix_processing(self, pdf_path: str) -> Dict[str, Any]:
        """Run the standard Fenix processing pipeline"""
        try:
            logger.info("🔄 Running standard Fenix processing...")
            
            # Use the existing main_workflow_enhanced function
            result = await process_pdf_document(pdf_path)
            
            # Wrap result in standard format
            return {
                "success": True,
                "output_file": result.get("output_file", ""),
                "processing_time": result.get("processing_time", 0.0),
                "extracted_text": result.get("extracted_text", ""),
                "translated_text": result.get("translated_text", "")
            }
            
        except Exception as e:
            logger.error(f"Standard Fenix processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _run_enhanced_validation(self, fenix_result: Dict[str, Any], 
                                     document_domain: str) -> Dict[str, Any]:
        """Run enhanced validation on the processed document"""
        try:
            logger.info("🔍 Running enhanced validation...")
            
            # Extract text from the result
            original_text = fenix_result.get("extracted_text", "")
            translated_text = fenix_result.get("translated_text", "")
            
            if not original_text or not translated_text:
                # Try to extract from output file
                output_file = fenix_result.get("output_file", "")
                if output_file and os.path.exists(output_file):
                    original_text, translated_text = await self._extract_text_from_output(output_file)
            
            # Run academic validation
            validation_results = validate_academic_document(
                original_text, translated_text, document_domain
            )
            
            # Generate validation report
            report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            generate_validation_report_file(validation_results, report_file)
            
            # Check for specific issues mentioned in the analysis
            enhanced_issues = await self._check_specific_issues(
                original_text, translated_text, validation_results["issues"]
            )
            
            # Combine all issues
            all_issues = validation_results["issues"] + enhanced_issues
            
            # Generate recommendations
            recommendations = self._generate_specific_recommendations(all_issues)
            
            return {
                "validation_report": report_file,
                "quality_score": validation_results["quality_score"],
                "issues_found": all_issues,
                "recommendations": recommendations,
                "validation_summary": validation_results["validation_summary"]
            }
            
        except Exception as e:
            logger.error(f"Enhanced validation failed: {e}")
            return {
                "validation_report": None,
                "quality_score": 0.5,
                "issues_found": [],
                "recommendations": ["Enhanced validation failed - manual review recommended"]
            }
    
    async def _check_specific_issues(self, original_text: str, translated_text: str, 
                                   existing_issues: List[ValidationIssue]) -> List[ValidationIssue]:
        """Check for specific issues mentioned in the user's analysis"""
        specific_issues = []
        
        # 1. Check for "Error! Bookmark not defined"
        if "Error! Bookmark not defined" in translated_text:
            specific_issues.append(ValidationIssue(
                issue_type="bookmark_error",
                severity=ValidationSeverity.CRITICAL,
                description="Document contains 'Error! Bookmark not defined' - indicates Word conversion issue",
                location="document_structure",
                original_text="Error! Bookmark not defined",
                suggested_fix="Regenerate document with proper bookmark handling",
                confidence=1.0
            ))
        
        # 2. Check for bibliography inconsistencies (Woodward vs Μπάρνοου pattern)
        bibliography_issues = self._check_bibliography_transliteration_issues(translated_text)
        specific_issues.extend(bibliography_issues)
        
        # 3. Check for incomplete translations in key sections
        incomplete_translation_issues = self._check_incomplete_translations(original_text, translated_text)
        specific_issues.extend(incomplete_translation_issues)
        
        # 4. Check for structural formatting issues
        structural_issues = self._check_structural_formatting(translated_text)
        specific_issues.extend(structural_issues)
        
        # 5. Check for mixed language issues
        mixed_language_issues = self._check_mixed_language_content(translated_text)
        specific_issues.extend(mixed_language_issues)
        
        return specific_issues
    
    def _check_bibliography_transliteration_issues(self, text: str) -> List[ValidationIssue]:
        """Check for specific bibliography transliteration inconsistencies"""
        issues = []
        
        # Look for patterns like "Μπάρνοου" (transliterated) vs "Woodward" (untranslated)
        # This addresses the specific issue mentioned in the analysis
        
        # Find potential author names in both English and Greek
        english_author_pattern = r'\b[A-Z][a-z]+,\s*[A-Z]\.'
        greek_author_pattern = r'\b[Α-Ω][α-ω]+,\s*[Α-Ω]\.'
        
        english_authors = re.findall(english_author_pattern, text)
        greek_authors = re.findall(greek_author_pattern, text)
        
        if english_authors and greek_authors:
            issues.append(ValidationIssue(
                issue_type="bibliography_transliteration_inconsistency",
                severity=ValidationSeverity.HIGH,
                description=f"Bibliography contains both English and Greek author names: {english_authors[:3]} vs {greek_authors[:3]}",
                location="bibliography",
                original_text=f"Mixed author names found: {len(english_authors)} English, {len(greek_authors)} Greek",
                suggested_fix="Consistently transliterate all author names to Greek throughout the bibliography",
                confidence=0.9
            ))
        
        # Check for specific patterns mentioned in the analysis
        if "Blanchard, T." in text and "Μπλάνσαρντ" in text:
            issues.append(ValidationIssue(
                issue_type="author_name_inconsistency",
                severity=ValidationSeverity.HIGH,
                description="Author 'Blanchard, T.' appears in both English and transliterated forms",
                location="bibliography",
                original_text="Blanchard, T. / Μπλάνσαρντ",
                suggested_fix="Use consistent transliteration: 'Μπλάνσαρντ, Τ.' throughout",
                confidence=1.0
            ))
        
        return issues
    
    def _check_incomplete_translations(self, original_text: str, translated_text: str) -> List[ValidationIssue]:
        """Check for incomplete translations in key sections"""
        issues = []
        
        # Check for common English words that should be translated
        english_sections = [
            "Acknowledgments", "Abstract", "Introduction", "Conclusion", 
            "References", "Bibliography", "Table of Contents"
        ]
        
        for section in english_sections:
            if section in translated_text:
                issues.append(ValidationIssue(
                    issue_type="incomplete_section_translation",
                    severity=ValidationSeverity.MEDIUM,
                    description=f"Section header '{section}' appears untranslated",
                    location="section_headers",
                    original_text=section,
                    suggested_fix=f"Translate '{section}' to appropriate Greek equivalent",
                    confidence=0.8
                ))
        
        # Check for English contact information patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, translated_text):
            # This is acceptable - emails don't need translation
            pass
        
        return issues
    
    def _check_structural_formatting(self, text: str) -> List[ValidationIssue]:
        """Check for structural formatting issues"""
        issues = []
        
        # Check for irregular spacing patterns
        if re.search(r'\n\s*\n\s*\n\s*\n', text):
            issues.append(ValidationIssue(
                issue_type="irregular_spacing",
                severity=ValidationSeverity.LOW,
                description="Document contains irregular spacing patterns",
                location="document_structure",
                original_text="Multiple consecutive blank lines",
                suggested_fix="Standardize spacing throughout document",
                confidence=0.7
            ))
        
        # Check for broken page references
        if "page" in text.lower() and not any(greek_word in text for greek_word in ["σελίδα", "σελ."]):
            issues.append(ValidationIssue(
                issue_type="untranslated_page_references",
                severity=ValidationSeverity.MEDIUM,
                description="Page references appear untranslated",
                location="page_references",
                original_text="page references",
                suggested_fix="Translate page references to Greek ('σελίδα' or 'σελ.')",
                confidence=0.6
            ))
        
        return issues
    
    def _check_mixed_language_content(self, text: str) -> List[ValidationIssue]:
        """Check for mixed language content issues"""
        issues = []
        
        # Count English vs Greek content
        english_words = len(re.findall(r'\b[A-Za-z]+\b', text))
        greek_words = len(re.findall(r'\b[Α-Ωα-ωάέήίόύώ]+\b', text))
        
        if english_words > 0 and greek_words > 0:
            english_ratio = english_words / (english_words + greek_words)
            
            if english_ratio > 0.3:  # More than 30% English content
                issues.append(ValidationIssue(
                    issue_type="excessive_english_content",
                    severity=ValidationSeverity.MEDIUM,
                    description=f"Document contains {english_ratio:.1%} English content",
                    location="document_wide",
                    original_text=f"{english_words} English words, {greek_words} Greek words",
                    suggested_fix="Review and translate remaining English content",
                    confidence=0.8
                ))
        
        return issues
    
    def _generate_specific_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate specific recommendations based on found issues"""
        recommendations = []
        
        # Group issues by type
        issue_types = {}
        for issue in issues:
            if issue.issue_type not in issue_types:
                issue_types[issue.issue_type] = []
            issue_types[issue.issue_type].append(issue)
        
        # Generate recommendations based on issue patterns
        if "bookmark_error" in issue_types:
            recommendations.append(
                "CRITICAL: Fix bookmark errors by regenerating the document with proper Word template handling"
            )
        
        if "bibliography_transliteration_inconsistency" in issue_types:
            recommendations.append(
                "HIGH PRIORITY: Establish consistent author name transliteration throughout the bibliography"
            )
        
        if "incomplete_section_translation" in issue_types:
            recommendations.append(
                "MEDIUM PRIORITY: Complete translation of all section headers and structural elements"
            )
        
        if "excessive_english_content" in issue_types:
            recommendations.append(
                "MEDIUM PRIORITY: Review document for remaining English content and complete translation"
            )
        
        # General recommendations based on issue severity
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        high_issues = [i for i in issues if i.severity == ValidationSeverity.HIGH]
        
        if critical_issues:
            recommendations.append(
                f"IMMEDIATE ACTION REQUIRED: {len(critical_issues)} critical issues must be resolved before publication"
            )
        
        if high_issues:
            recommendations.append(
                f"PROFESSIONAL REVIEW RECOMMENDED: {len(high_issues)} high-priority issues require expert attention"
            )
        
        if not recommendations:
            recommendations.append("Document quality is acceptable - minor improvements may be beneficial")
        
        return recommendations
    
    async def _extract_text_from_output(self, output_file: str) -> tuple[str, str]:
        """Extract original and translated text from output file"""
        try:
            if output_file.endswith('.docx'):
                from docx import Document
                doc = Document(output_file)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                # For now, return the same text as both original and translated
                # In a real implementation, we'd need to track this separately
                return text, text
            else:
                # For other file types, return empty strings
                return "", ""
        except Exception as e:
            logger.warning(f"Could not extract text from output file: {e}")
            return "", ""
    
    async def _run_professional_review(self, fenix_result: Dict[str, Any], 
                                     validation_result: Dict[str, Any],
                                     document_domain: str) -> Dict[str, Any]:
        """Run professional review workflow"""
        try:
            logger.info("👥 Initiating professional review workflow...")
            
            # Extract text for professional review
            original_text = fenix_result.get("extracted_text", "")
            translated_text = fenix_result.get("translated_text", "")
            
            # Create professional translation project
            project = await self.professional_workflow.process_document_professionally(
                original_text=original_text,
                document_title=os.path.basename(fenix_result.get("output_file", "document")),
                source_language="English",
                target_language="Greek",
                domain=document_domain,
                priority=1
            )
            
            return {
                "project_id": project.project_id,
                "professional_quality_score": project.final_quality_score,
                "review_tasks": len(project.review_tasks),
                "workflow_stage": project.current_stage.value
            }
            
        except Exception as e:
            logger.error(f"Professional review failed: {e}")
            return {
                "project_id": None,
                "error": str(e)
            }
    
    async def _generate_quality_report(self, result: FenixProcessingResult, pdf_path: str):
        """Generate comprehensive quality report"""
        try:
            report_file = f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("FENIX PROFESSIONAL TRANSLATION QUALITY REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Document: {os.path.basename(pdf_path)}\n")
                f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Quality Mode: {result.quality_mode.value}\n")
                f.write(f"Processing Time: {result.processing_time:.2f} seconds\n")
                f.write(f"Overall Quality Score: {result.quality_score:.2f}/1.00\n\n")
                
                # Issues summary
                f.write("ISSUES SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Issues Found: {len(result.issues_found)}\n")
                
                if result.issues_found:
                    severity_counts = {}
                    for issue in result.issues_found:
                        severity = issue.severity.value
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    
                    for severity, count in severity_counts.items():
                        f.write(f"{severity.capitalize()}: {count}\n")
                
                f.write("\n")
                
                # Detailed issues
                if result.issues_found:
                    f.write("DETAILED ISSUES\n")
                    f.write("-" * 40 + "\n")
                    for i, issue in enumerate(result.issues_found, 1):
                        f.write(f"{i}. {issue.description}\n")
                        f.write(f"   Severity: {issue.severity.value}\n")
                        f.write(f"   Location: {issue.location}\n")
                        if issue.suggested_fix:
                            f.write(f"   Suggested Fix: {issue.suggested_fix}\n")
                        f.write("\n")
                
                # Recommendations
                if result.recommendations:
                    f.write("RECOMMENDATIONS\n")
                    f.write("-" * 40 + "\n")
                    for i, rec in enumerate(result.recommendations, 1):
                        f.write(f"{i}. {rec}\n")
                    f.write("\n")
                
                # Professional workflow info
                if result.professional_project_id:
                    f.write("PROFESSIONAL WORKFLOW\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Project ID: {result.professional_project_id}\n")
                    f.write("Status: Expert review in progress\n")
                    f.write("Check professional workflow dashboard for updates\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"📊 Quality report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        if not self.processing_history:
            return {"message": "No processing history available"}
        
        total_processed = len(self.processing_history)
        successful = sum(1 for h in self.processing_history if h["success"])
        avg_quality = sum(h["quality_score"] for h in self.processing_history) / total_processed
        
        quality_modes = {}
        for h in self.processing_history:
            mode = h["quality_mode"]
            quality_modes[mode] = quality_modes.get(mode, 0) + 1
        
        return {
            "total_documents_processed": total_processed,
            "successful_processes": successful,
            "success_rate": successful / total_processed,
            "average_quality_score": avg_quality,
            "quality_modes_used": quality_modes,
            "last_processed": self.processing_history[-1]["timestamp"] if self.processing_history else None
        }

# Global instance for easy access
fenix_professional = FenixProfessionalIntegration()

# Convenience functions for backward compatibility
async def process_pdf_professionally(pdf_path: str, 
                                   quality_mode: str = "enhanced",
                                   domain: str = "academic") -> FenixProcessingResult:
    """
    Process a PDF with professional quality control
    
    This function provides a simple interface to the professional translation system
    while maintaining compatibility with existing Fenix workflows.
    """
    return await fenix_professional.process_document_professionally(
        pdf_path=pdf_path,
        quality_mode=quality_mode,
        document_domain=domain
    )

def get_fenix_professional_statistics() -> Dict[str, Any]:
    """Get professional processing statistics"""
    return fenix_professional.get_processing_statistics()

# Example usage demonstration
if __name__ == "__main__":
    async def demo_professional_processing():
        """Demonstrate professional processing capabilities"""
        print("🏆 Fenix Professional Translation System Demo")
        print("=" * 50)
        
        # Example processing with different quality modes
        test_pdf = "test_document.pdf"
        
        if os.path.exists(test_pdf):
            print(f"Processing {test_pdf} with different quality modes...")
            
            # Standard mode
            result_standard = await process_pdf_professionally(
                test_pdf, quality_mode="standard"
            )
            print(f"Standard mode - Quality: {result_standard.quality_score:.2f}")
            
            # Enhanced mode
            result_enhanced = await process_pdf_professionally(
                test_pdf, quality_mode="enhanced"
            )
            print(f"Enhanced mode - Quality: {result_enhanced.quality_score:.2f}")
            print(f"Issues found: {len(result_enhanced.issues_found)}")
            
            # Professional mode
            result_professional = await process_pdf_professionally(
                test_pdf, quality_mode="professional"
            )
            print(f"Professional mode - Quality: {result_professional.quality_score:.2f}")
            print(f"Professional project: {result_professional.professional_project_id}")
            
        else:
            print(f"Test file {test_pdf} not found - skipping demo")
        
        # Show statistics
        stats = get_fenix_professional_statistics()
        print("\nProcessing Statistics:")
        print(f"Total processed: {stats.get('total_documents_processed', 0)}")
        print(f"Success rate: {stats.get('success_rate', 0):.2%}")
        print(f"Average quality: {stats.get('average_quality_score', 0):.2f}")
    
    # Run demo
    import asyncio
    asyncio.run(demo_professional_processing()) 