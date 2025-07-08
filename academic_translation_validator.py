"""
Academic Translation Validation System

This module implements comprehensive validation for academic document translations,
addressing the specific deficiencies identified in translation quality analysis:

1. Bibliography consistency validation
2. Proper noun and author name handling
3. Academic terminology management
4. Document structure validation
5. Translation quality scoring

Based on the analysis of translation deficiencies in academic documents.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import unicodedata
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class ValidationIssue:
    """Represents a validation issue found in the translation"""
    issue_type: str
    severity: ValidationSeverity
    description: str
    location: str
    original_text: str
    suggested_fix: Optional[str] = None
    confidence: float = 1.0

@dataclass
class AuthorNameEntry:
    """Represents an author name with various forms"""
    original_name: str
    transliterated_name: str
    alternative_forms: List[str] = field(default_factory=list)
    language_of_origin: Optional[str] = None
    consistency_rule: Optional[str] = None

@dataclass
class AcademicTerm:
    """Represents an academic term with domain-specific translations"""
    source_term: str
    target_translation: str
    domain: str
    context: str
    confidence: float
    alternatives: List[str] = field(default_factory=list)

class EnhancedAcademicGlossary:
    """
    Enhanced glossary system for academic translations with proper noun handling,
    domain-specific terminology, and consistency management.
    """
    
    def __init__(self, glossary_file: str = "academic_glossary.json"):
        self.glossary_file = glossary_file
        self.academic_terms: Dict[str, AcademicTerm] = {}
        self.author_names: Dict[str, AuthorNameEntry] = {}
        self.domain_vocabularies: Dict[str, Dict[str, str]] = {}
        self.consistency_rules: Dict[str, str] = {}
        self.proper_noun_patterns: List[str] = []
        self.load_glossary()
        
    def load_glossary(self):
        """Load enhanced academic glossary from file"""
        try:
            if os.path.exists(self.glossary_file):
                with open(self.glossary_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._parse_glossary_data(data)
                logger.info(f"Loaded enhanced academic glossary with {len(self.academic_terms)} terms")
            else:
                self._create_default_academic_glossary()
        except Exception as e:
            logger.error(f"Error loading academic glossary: {e}")
            self._create_default_academic_glossary()
    
    def _parse_glossary_data(self, data: Dict[str, Any]):
        """Parse glossary data from JSON"""
        # Load academic terms
        for term_data in data.get('academic_terms', []):
            term = AcademicTerm(
                source_term=term_data['source_term'],
                target_translation=term_data['target_translation'],
                domain=term_data['domain'],
                context=term_data.get('context', ''),
                confidence=term_data.get('confidence', 1.0),
                alternatives=term_data.get('alternatives', [])
            )
            self.academic_terms[term.source_term.lower()] = term
        
        # Load author names
        for name_data in data.get('author_names', []):
            author = AuthorNameEntry(
                original_name=name_data['original_name'],
                transliterated_name=name_data['transliterated_name'],
                alternative_forms=name_data.get('alternative_forms', []),
                language_of_origin=name_data.get('language_of_origin'),
                consistency_rule=name_data.get('consistency_rule')
            )
            self.author_names[author.original_name.lower()] = author
        
        # Load domain vocabularies
        self.domain_vocabularies = data.get('domain_vocabularies', {})
        
        # Load consistency rules
        self.consistency_rules = data.get('consistency_rules', {})
        
        # Load proper noun patterns
        self.proper_noun_patterns = data.get('proper_noun_patterns', [])
    
    def _create_default_academic_glossary(self):
        """Create default academic glossary with common terms"""
        default_data = {
            'academic_terms': [
                {
                    'source_term': 'counterfactual',
                    'target_translation': 'αντιπραγματικός',
                    'domain': 'philosophy',
                    'context': 'causation theory',
                    'confidence': 0.95,
                    'alternatives': ['υποθετικός', 'εναλλακτικός']
                },
                {
                    'source_term': 'interventionist',
                    'target_translation': 'παρεμβατικός',
                    'domain': 'philosophy',
                    'context': 'causation theory',
                    'confidence': 0.9,
                    'alternatives': ['επεμβατικός']
                },
                {
                    'source_term': 'specificity',
                    'target_translation': 'ειδικότητα',
                    'domain': 'philosophy',
                    'context': 'causation theory',
                    'confidence': 0.85,
                    'alternatives': ['συγκεκριμένη φύση']
                },
                {
                    'source_term': 'stability',
                    'target_translation': 'σταθερότητα',
                    'domain': 'philosophy',
                    'context': 'causation theory',
                    'confidence': 0.9,
                    'alternatives': ['στερεότητα']
                }
            ],
            'author_names': [
                {
                    'original_name': 'Woodward, J.',
                    'transliterated_name': 'Γούντγουαρντ, Τζ.',
                    'alternative_forms': ['Woodward, James', 'J. Woodward'],
                    'language_of_origin': 'English',
                    'consistency_rule': 'transliterate_surname_first'
                },
                {
                    'original_name': 'Blanchard, T.',
                    'transliterated_name': 'Μπλάνσαρντ, Τ.',
                    'alternative_forms': ['Blanchard, Thomas', 'T. Blanchard'],
                    'language_of_origin': 'English',
                    'consistency_rule': 'transliterate_surname_first'
                }
            ],
            'domain_vocabularies': {
                'philosophy': {
                    'causation': 'αιτιότητα',
                    'metaphysics': 'μεταφυσική',
                    'epistemology': 'επιστημολογία',
                    'ontology': 'οντολογία'
                },
                'science': {
                    'hypothesis': 'υπόθεση',
                    'methodology': 'μεθοδολογία',
                    'empirical': 'εμπειρικός',
                    'theoretical': 'θεωρητικός'
                }
            },
            'consistency_rules': {
                'author_names': 'Always transliterate author names consistently throughout the document',
                'technical_terms': 'Use established Greek translations for technical terms',
                'citations': 'Maintain consistent citation format throughout'
            },
            'proper_noun_patterns': [
                r'\b[A-Z][a-z]+ [A-Z]\.',  # Author names like "Smith, J."
                r'\b[A-Z][a-z]+, [A-Z][a-z]+',  # Full names like "Smith, John"
                r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*',  # Acronyms like "DOI", "URL"
                r'\b\d{4}[a-z]?',  # Years like "2023", "2023a"
            ]
        }
        
        self._parse_glossary_data(default_data)
        self.save_glossary()
    
    def save_glossary(self):
        """Save the enhanced glossary to file"""
        try:
            data = {
                'academic_terms': [
                    {
                        'source_term': term.source_term,
                        'target_translation': term.target_translation,
                        'domain': term.domain,
                        'context': term.context,
                        'confidence': term.confidence,
                        'alternatives': term.alternatives
                    }
                    for term in self.academic_terms.values()
                ],
                'author_names': [
                    {
                        'original_name': author.original_name,
                        'transliterated_name': author.transliterated_name,
                        'alternative_forms': author.alternative_forms,
                        'language_of_origin': author.language_of_origin,
                        'consistency_rule': author.consistency_rule
                    }
                    for author in self.author_names.values()
                ],
                'domain_vocabularies': self.domain_vocabularies,
                'consistency_rules': self.consistency_rules,
                'proper_noun_patterns': self.proper_noun_patterns
            }
            
            with open(self.glossary_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved enhanced academic glossary to {self.glossary_file}")
        except Exception as e:
            logger.error(f"Error saving academic glossary: {e}")
    
    def get_academic_term_translation(self, term: str, domain: str = None) -> Optional[AcademicTerm]:
        """Get translation for an academic term"""
        term_lower = term.lower()
        
        # Direct match
        if term_lower in self.academic_terms:
            academic_term = self.academic_terms[term_lower]
            if domain is None or academic_term.domain == domain:
                return academic_term
        
        # Domain-specific vocabulary
        if domain and domain in self.domain_vocabularies:
            domain_vocab = self.domain_vocabularies[domain]
            if term_lower in domain_vocab:
                return AcademicTerm(
                    source_term=term,
                    target_translation=domain_vocab[term_lower],
                    domain=domain,
                    context=f"Domain vocabulary: {domain}",
                    confidence=0.8
                )
        
        return None
    
    def get_author_name_translation(self, author_name: str) -> Optional[AuthorNameEntry]:
        """Get consistent translation for author name"""
        author_lower = author_name.lower()
        
        # Direct match
        if author_lower in self.author_names:
            return self.author_names[author_lower]
        
        # Try alternative forms
        for author in self.author_names.values():
            if author_lower in [alt.lower() for alt in author.alternative_forms]:
                return author
        
        return None
    
    def detect_proper_nouns(self, text: str) -> List[str]:
        """Detect proper nouns in text using patterns"""
        proper_nouns = []
        
        for pattern in self.proper_noun_patterns:
            matches = re.findall(pattern, text)
            proper_nouns.extend(matches)
        
        return list(set(proper_nouns))
    
    def add_academic_term(self, source_term: str, target_translation: str, 
                         domain: str, context: str = "", confidence: float = 1.0):
        """Add new academic term to glossary"""
        term = AcademicTerm(
            source_term=source_term,
            target_translation=target_translation,
            domain=domain,
            context=context,
            confidence=confidence
        )
        self.academic_terms[source_term.lower()] = term
        logger.info(f"Added academic term: {source_term} -> {target_translation}")
    
    def add_author_name(self, original_name: str, transliterated_name: str,
                       alternative_forms: List[str] = None, language_of_origin: str = None):
        """Add new author name to glossary"""
        author = AuthorNameEntry(
            original_name=original_name,
            transliterated_name=transliterated_name,
            alternative_forms=alternative_forms or [],
            language_of_origin=language_of_origin,
            consistency_rule="transliterate_surname_first"
        )
        self.author_names[original_name.lower()] = author
        logger.info(f"Added author name: {original_name} -> {transliterated_name}")

class BibliographyConsistencyValidator:
    """
    Validates bibliography consistency in academic translations,
    ensuring uniform author name handling and citation formatting.
    """
    
    def __init__(self, glossary: EnhancedAcademicGlossary):
        self.glossary = glossary
        self.author_name_variations: Dict[str, List[str]] = defaultdict(list)
        self.citation_patterns: List[str] = []
        
    def validate_bibliography(self, bibliography_text: str) -> List[ValidationIssue]:
        """Validate bibliography for consistency issues"""
        issues = []
        
        # Extract bibliography entries
        entries = self._extract_bibliography_entries(bibliography_text)
        
        if not entries:
            issues.append(ValidationIssue(
                issue_type="bibliography_structure",
                severity=ValidationSeverity.HIGH,
                description="No bibliography entries found or bibliography structure not recognized",
                location="bibliography",
                original_text=bibliography_text[:100] + "..." if len(bibliography_text) > 100 else bibliography_text
            ))
            return issues
        
        # Validate each entry
        for i, entry in enumerate(entries):
            entry_issues = self._validate_bibliography_entry(entry, i)
            issues.extend(entry_issues)
        
        # Check for consistency across entries
        consistency_issues = self._check_bibliography_consistency(entries)
        issues.extend(consistency_issues)
        
        return issues
    
    def _extract_bibliography_entries(self, text: str) -> List[str]:
        """Extract individual bibliography entries from text"""
        # Common bibliography patterns
        patterns = [
            r'^\s*\d+\.\s+(.+?)(?=^\s*\d+\.|$)',  # Numbered entries
            r'^\s*([A-Z][^.]+\..*?)(?=^\s*[A-Z][^.]+\.|$)',  # Author-year entries
            r'^\s*([A-Z][^.]+\..*?)(?=^\n\s*[A-Z][^.]+\.|$)',  # Line-separated entries
        ]
        
        entries = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            if matches:
                entries.extend([match.strip() for match in matches])
                break
        
        return entries
    
    def _validate_bibliography_entry(self, entry: str, entry_index: int) -> List[ValidationIssue]:
        """Validate a single bibliography entry"""
        issues = []
        
        # Check for author name consistency
        author_issues = self._check_author_name_consistency(entry, entry_index)
        issues.extend(author_issues)
        
        # Check for mixed languages
        language_issues = self._check_language_consistency(entry, entry_index)
        issues.extend(language_issues)
        
        # Check for incomplete translations
        translation_issues = self._check_translation_completeness(entry, entry_index)
        issues.extend(translation_issues)
        
        return issues
    
    def _check_author_name_consistency(self, entry: str, entry_index: int) -> List[ValidationIssue]:
        """Check for author name consistency in bibliography entry"""
        issues = []
        
        # Extract potential author names
        author_patterns = [
            r'([A-Z][a-z]+,\s*[A-Z]\.)',  # Surname, Initial
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+)',  # Surname, Firstname
            r'([A-Z][a-z]+\s+[A-Z]\.)',  # Firstname Initial Surname
        ]
        
        found_authors = []
        for pattern in author_patterns:
            matches = re.findall(pattern, entry)
            found_authors.extend(matches)
        
        for author in found_authors:
            # Check if author should be transliterated
            author_entry = self.glossary.get_author_name_translation(author)
            if author_entry:
                # Check if the entry uses the correct transliteration
                if author_entry.transliterated_name not in entry:
                    issues.append(ValidationIssue(
                        issue_type="author_name_inconsistency",
                        severity=ValidationSeverity.HIGH,
                        description=f"Author name '{author}' should be transliterated as '{author_entry.transliterated_name}'",
                        location=f"bibliography_entry_{entry_index}",
                        original_text=entry,
                        suggested_fix=entry.replace(author, author_entry.transliterated_name)
                    ))
            else:
                # Check if this is a proper noun that should be transliterated
                if self._is_likely_proper_noun(author):
                    issues.append(ValidationIssue(
                        issue_type="untransliterated_author_name",
                        severity=ValidationSeverity.MEDIUM,
                        description=f"Author name '{author}' appears to be untransliterated",
                        location=f"bibliography_entry_{entry_index}",
                        original_text=entry,
                        suggested_fix=f"Consider transliterating '{author}' consistently"
                    ))
        
        return issues
    
    def _check_language_consistency(self, entry: str, entry_index: int) -> List[ValidationIssue]:
        """Check for language consistency in bibliography entry"""
        issues = []
        
        # Detect mixed languages (English words in Greek text)
        english_words = re.findall(r'\b[A-Za-z]{3,}\b', entry)
        greek_chars = re.findall(r'[Α-Ωα-ωάέήίόύώ]', entry)
        
        if english_words and greek_chars:
            # Check if English words are proper nouns or technical terms
            untranslated_words = []
            for word in english_words:
                if not self._is_acceptable_english_word(word):
                    untranslated_words.append(word)
            
            if untranslated_words:
                issues.append(ValidationIssue(
                    issue_type="mixed_language_bibliography",
                    severity=ValidationSeverity.MEDIUM,
                    description=f"Bibliography entry contains untranslated English words: {', '.join(untranslated_words)}",
                    location=f"bibliography_entry_{entry_index}",
                    original_text=entry,
                    suggested_fix="Translate or consistently transliterate all terms"
                ))
        
        return issues
    
    def _check_translation_completeness(self, entry: str, entry_index: int) -> List[ValidationIssue]:
        """Check if bibliography entry is completely translated"""
        issues = []
        
        # Check for common English bibliography terms that should be translated
        english_terms = {
            'In': 'Στο',
            'Ed.': 'Επιμ.',
            'Eds.': 'Επιμ.',
            'Vol.': 'Τόμ.',
            'No.': 'Αρ.',
            'pp.': 'σσ.',
            'University Press': 'Πανεπιστημιακές Εκδόσεις',
            'Journal of': 'Περιοδικό',
            'Proceedings of': 'Πρακτικά'
        }
        
        for english_term, greek_term in english_terms.items():
            if english_term in entry and greek_term not in entry:
                issues.append(ValidationIssue(
                    issue_type="incomplete_bibliography_translation",
                    severity=ValidationSeverity.MEDIUM,
                    description=f"Bibliography term '{english_term}' should be translated to '{greek_term}'",
                    location=f"bibliography_entry_{entry_index}",
                    original_text=entry,
                    suggested_fix=entry.replace(english_term, greek_term)
                ))
        
        return issues
    
    def _check_bibliography_consistency(self, entries: List[str]) -> List[ValidationIssue]:
        """Check consistency across all bibliography entries"""
        issues = []
        
        # Check for consistent author name handling
        author_name_issues = self._check_cross_entry_author_consistency(entries)
        issues.extend(author_name_issues)
        
        # Check for consistent formatting
        formatting_issues = self._check_bibliography_formatting_consistency(entries)
        issues.extend(formatting_issues)
        
        return issues
    
    def _check_cross_entry_author_consistency(self, entries: List[str]) -> List[ValidationIssue]:
        """Check if the same author is handled consistently across entries"""
        issues = []
        
        # Extract all author names from all entries
        all_authors = defaultdict(list)
        
        for i, entry in enumerate(entries):
            authors = self._extract_authors_from_entry(entry)
            for author in authors:
                all_authors[self._normalize_author_name(author)].append((author, i))
        
        # Check for inconsistent representations of the same author
        for normalized_name, author_instances in all_authors.items():
            if len(author_instances) > 1:
                # Check if all instances are identical
                representations = [author for author, _ in author_instances]
                if len(set(representations)) > 1:
                    issues.append(ValidationIssue(
                        issue_type="inconsistent_author_representation",
                        severity=ValidationSeverity.HIGH,
                        description=f"Author represented inconsistently: {', '.join(set(representations))}",
                        location="bibliography_cross_reference",
                        original_text=f"Entries: {[i for _, i in author_instances]}",
                        suggested_fix="Use consistent representation for the same author throughout"
                    ))
        
        return issues
    
    def _extract_authors_from_entry(self, entry: str) -> List[str]:
        """Extract author names from a bibliography entry"""
        # This is a simplified extraction - could be enhanced
        patterns = [
            r'([A-Z][a-z]+,\s*[A-Z]\.)',
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+)',
            r'([Α-Ω][α-ω]+,\s*[Α-Ω]\.)',  # Greek names
            r'([Α-Ω][α-ω]+,\s*[Α-Ω][α-ω]+)',  # Greek names
        ]
        
        authors = []
        for pattern in patterns:
            matches = re.findall(pattern, entry)
            authors.extend(matches)
        
        return authors
    
    def _normalize_author_name(self, author: str) -> str:
        """Normalize author name for comparison"""
        # Remove diacritics, convert to lowercase, standardize spacing
        normalized = unicodedata.normalize('NFD', author.lower())
        normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        return normalized
    
    def _check_bibliography_formatting_consistency(self, entries: List[str]) -> List[ValidationIssue]:
        """Check for consistent formatting across bibliography entries"""
        issues = []
        
        # Check for consistent punctuation patterns
        punctuation_patterns = []
        for entry in entries:
            # Extract punctuation pattern (simplified)
            pattern = re.sub(r'[^.,;:()]', '', entry)
            punctuation_patterns.append(pattern)
        
        # If there's significant variation in punctuation patterns, flag it
        unique_patterns = set(punctuation_patterns)
        if len(unique_patterns) > len(entries) * 0.5:  # More than 50% variation
            issues.append(ValidationIssue(
                issue_type="inconsistent_bibliography_formatting",
                severity=ValidationSeverity.MEDIUM,
                description="Bibliography entries show inconsistent formatting patterns",
                location="bibliography_formatting",
                original_text="Multiple formatting patterns detected",
                suggested_fix="Standardize bibliography formatting throughout"
            ))
        
        return issues
    
    def _is_likely_proper_noun(self, text: str) -> bool:
        """Check if text is likely a proper noun"""
        # Simple heuristic: starts with capital letter, contains mostly Latin characters
        if not text or not text[0].isupper():
            return False
        
        latin_chars = sum(1 for c in text if ord(c) < 128)
        return latin_chars / len(text) > 0.8
    
    def _is_acceptable_english_word(self, word: str) -> bool:
        """Check if an English word is acceptable in Greek text"""
        # Common acceptable English words in academic Greek
        acceptable_words = {
            'DOI', 'URL', 'ISBN', 'ISSN', 'PDF', 'HTML', 'XML',
            'et', 'al', 'ibid', 'op', 'cit', 'cf', 'vs', 'e.g.', 'i.e.',
            'PhD', 'MA', 'BA', 'MSc', 'BSc'
        }
        
        return word.upper() in acceptable_words or len(word) <= 2

class AcademicTranslationValidator:
    """
    Main validator class that coordinates all validation aspects
    """
    
    def __init__(self, glossary_file: str = "academic_glossary.json"):
        self.glossary = EnhancedAcademicGlossary(glossary_file)
        self.bibliography_validator = BibliographyConsistencyValidator(self.glossary)
        self.validation_history: List[Dict[str, Any]] = []
    
    def validate_academic_translation(self, original_text: str, translated_text: str,
                                    document_type: str = "academic") -> Dict[str, Any]:
        """
        Perform comprehensive validation of academic translation
        
        Returns a detailed validation report with issues and recommendations
        """
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'document_type': document_type,
            'validation_summary': {
                'total_issues': 0,
                'critical_issues': 0,
                'high_priority_issues': 0,
                'medium_priority_issues': 0,
                'low_priority_issues': 0
            },
            'issues': [],
            'recommendations': [],
            'quality_score': 0.0
        }
        
        try:
            # Validate bibliography if present
            bibliography_issues = self._validate_bibliography_section(translated_text)
            validation_report['issues'].extend(bibliography_issues)
            
            # Validate terminology consistency
            terminology_issues = self._validate_terminology_consistency(translated_text)
            validation_report['issues'].extend(terminology_issues)
            
            # Validate document structure
            structure_issues = self._validate_document_structure(translated_text)
            validation_report['issues'].extend(structure_issues)
            
            # Validate proper noun handling
            proper_noun_issues = self._validate_proper_noun_handling(original_text, translated_text)
            validation_report['issues'].extend(proper_noun_issues)
            
            # Calculate summary statistics
            self._calculate_validation_summary(validation_report)
            
            # Generate recommendations
            self._generate_recommendations(validation_report)
            
            # Calculate quality score
            validation_report['quality_score'] = self._calculate_quality_score(validation_report)
            
            # Store validation history
            self.validation_history.append(validation_report)
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_report['error'] = str(e)
            return validation_report
    
    def _validate_bibliography_section(self, text: str) -> List[ValidationIssue]:
        """Validate bibliography section if present"""
        issues = []
        
        # Try to find bibliography section
        bibliography_patterns = [
            r'(?i)(bibliography|βιβλιογραφία|αναφορές|references)(.*?)(?=\n\n|\Z)',
            r'(?i)(works cited|έργα που αναφέρονται)(.*?)(?=\n\n|\Z)',
            r'(?i)(sources|πηγές)(.*?)(?=\n\n|\Z)'
        ]
        
        for pattern in bibliography_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                bibliography_text = match.group(2)
                bibliography_issues = self.bibliography_validator.validate_bibliography(bibliography_text)
                issues.extend(bibliography_issues)
                break
        
        return issues
    
    def _validate_terminology_consistency(self, text: str) -> List[ValidationIssue]:
        """Validate terminology consistency throughout the document"""
        issues = []
        
        # Check for consistent use of academic terms
        term_usage = defaultdict(list)
        
        for term_key, academic_term in self.glossary.academic_terms.items():
            source_term = academic_term.source_term
            target_term = academic_term.target_translation
            
            # Find all occurrences of the source term
            source_occurrences = [(m.start(), m.end()) for m in re.finditer(re.escape(source_term), text, re.IGNORECASE)]
            target_occurrences = [(m.start(), m.end()) for m in re.finditer(re.escape(target_term), text, re.IGNORECASE)]
            
            if source_occurrences and target_occurrences:
                issues.append(ValidationIssue(
                    issue_type="terminology_inconsistency",
                    severity=ValidationSeverity.MEDIUM,
                    description=f"Term '{source_term}' appears in both original and translated forms",
                    location="terminology_consistency",
                    original_text=f"Found {len(source_occurrences)} occurrences of '{source_term}' and {len(target_occurrences)} of '{target_term}'",
                    suggested_fix=f"Use consistently: '{target_term}'"
                ))
            elif source_occurrences:
                issues.append(ValidationIssue(
                    issue_type="untranslated_terminology",
                    severity=ValidationSeverity.HIGH,
                    description=f"Academic term '{source_term}' appears untranslated",
                    location="terminology_consistency",
                    original_text=f"Found {len(source_occurrences)} untranslated occurrences",
                    suggested_fix=f"Translate to: '{target_term}'"
                ))
        
        return issues
    
    def _validate_document_structure(self, text: str) -> List[ValidationIssue]:
        """Validate document structure and formatting"""
        issues = []
        
        # Check for structural errors mentioned in the analysis
        structural_problems = [
            (r'Error!\s*Bookmark\s*not\s*defined', "Bookmark reference error", ValidationSeverity.CRITICAL),
            (r'\[TRANSLATION_ERROR[^\]]*\]', "Translation error marker", ValidationSeverity.CRITICAL),
            (r'\[UNTRANSLATED[^\]]*\]', "Untranslated content marker", ValidationSeverity.HIGH),
            (r'\n\s*\n\s*\n\s*\n', "Excessive blank lines", ValidationSeverity.LOW),
            (r'^\s*$\n^\s*$\n^\s*$', "Multiple consecutive blank lines", ValidationSeverity.LOW)
        ]
        
        for pattern, description, severity in structural_problems:
            matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
            if matches:
                for match in matches:
                    issues.append(ValidationIssue(
                        issue_type="document_structure",
                        severity=severity,
                        description=description,
                        location=f"position_{match.start()}",
                        original_text=match.group(),
                        suggested_fix="Fix structural formatting"
                    ))
        
        return issues
    
    def _validate_proper_noun_handling(self, original_text: str, translated_text: str) -> List[ValidationIssue]:
        """Validate proper noun handling between original and translated text"""
        issues = []
        
        # Detect proper nouns in original text
        original_proper_nouns = self.glossary.detect_proper_nouns(original_text)
        translated_proper_nouns = self.glossary.detect_proper_nouns(translated_text)
        
        # Check if proper nouns are handled consistently
        for proper_noun in original_proper_nouns:
            author_entry = self.glossary.get_author_name_translation(proper_noun)
            if author_entry:
                # Check if the correct transliteration is used
                if author_entry.transliterated_name not in translated_text:
                    issues.append(ValidationIssue(
                        issue_type="proper_noun_handling",
                        severity=ValidationSeverity.MEDIUM,
                        description=f"Proper noun '{proper_noun}' not correctly transliterated",
                        location="proper_noun_consistency",
                        original_text=proper_noun,
                        suggested_fix=f"Use transliteration: '{author_entry.transliterated_name}'"
                    ))
        
        return issues
    
    def _calculate_validation_summary(self, validation_report: Dict[str, Any]):
        """Calculate summary statistics for validation report"""
        issues = validation_report['issues']
        summary = validation_report['validation_summary']
        
        summary['total_issues'] = len(issues)
        
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                summary['critical_issues'] += 1
            elif issue.severity == ValidationSeverity.HIGH:
                summary['high_priority_issues'] += 1
            elif issue.severity == ValidationSeverity.MEDIUM:
                summary['medium_priority_issues'] += 1
            elif issue.severity == ValidationSeverity.LOW:
                summary['low_priority_issues'] += 1
    
    def _generate_recommendations(self, validation_report: Dict[str, Any]):
        """Generate recommendations based on validation issues"""
        issues = validation_report['issues']
        recommendations = []
        
        # Critical issues
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            recommendations.append({
                'priority': 'critical',
                'recommendation': 'Address all critical structural errors immediately',
                'details': f"Found {len(critical_issues)} critical issues that prevent proper document display"
            })
        
        # Bibliography issues
        bibliography_issues = [i for i in issues if 'bibliography' in i.issue_type]
        if bibliography_issues:
            recommendations.append({
                'priority': 'high',
                'recommendation': 'Standardize bibliography formatting and author name handling',
                'details': f"Found {len(bibliography_issues)} bibliography consistency issues"
            })
        
        # Terminology issues
        terminology_issues = [i for i in issues if 'terminology' in i.issue_type]
        if terminology_issues:
            recommendations.append({
                'priority': 'medium',
                'recommendation': 'Implement consistent terminology throughout the document',
                'details': f"Found {len(terminology_issues)} terminology consistency issues"
            })
        
        # General quality improvement
        if len(issues) > 10:
            recommendations.append({
                'priority': 'medium',
                'recommendation': 'Consider comprehensive editorial review',
                'details': f"Document has {len(issues)} total issues requiring systematic review"
            })
        
        validation_report['recommendations'] = recommendations
    
    def _calculate_quality_score(self, validation_report: Dict[str, Any]) -> float:
        """Calculate overall quality score based on validation results"""
        issues = validation_report['issues']
        
        if not issues:
            return 1.0
        
        # Weight different severity levels
        severity_weights = {
            ValidationSeverity.CRITICAL: 1.0,
            ValidationSeverity.HIGH: 0.7,
            ValidationSeverity.MEDIUM: 0.4,
            ValidationSeverity.LOW: 0.1
        }
        
        total_penalty = sum(severity_weights.get(issue.severity, 0.1) for issue in issues)
        
        # Calculate score (higher penalty = lower score)
        # Normalize by document length estimate (assume 1000 words baseline)
        base_score = 1.0
        penalty_factor = total_penalty / 10.0  # Scale penalty
        
        quality_score = max(0.0, base_score - penalty_factor)
        return round(quality_score, 2)
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a human-readable validation report"""
        report_lines = []
        
        report_lines.append("=" * 60)
        report_lines.append("ACADEMIC TRANSLATION VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {validation_results['timestamp']}")
        report_lines.append(f"Document Type: {validation_results['document_type']}")
        report_lines.append(f"Overall Quality Score: {validation_results['quality_score']:.2f}/1.00")
        report_lines.append("")
        
        # Summary
        summary = validation_results['validation_summary']
        report_lines.append("VALIDATION SUMMARY")
        report_lines.append("-" * 20)
        report_lines.append(f"Total Issues: {summary['total_issues']}")
        report_lines.append(f"Critical Issues: {summary['critical_issues']}")
        report_lines.append(f"High Priority: {summary['high_priority_issues']}")
        report_lines.append(f"Medium Priority: {summary['medium_priority_issues']}")
        report_lines.append(f"Low Priority: {summary['low_priority_issues']}")
        report_lines.append("")
        
        # Issues by category
        issues_by_type = defaultdict(list)
        for issue in validation_results['issues']:
            issues_by_type[issue.issue_type].append(issue)
        
        if issues_by_type:
            report_lines.append("ISSUES BY CATEGORY")
            report_lines.append("-" * 20)
            
            for issue_type, issues in issues_by_type.items():
                report_lines.append(f"\n{issue_type.upper().replace('_', ' ')} ({len(issues)} issues):")
                for issue in issues[:3]:  # Show first 3 issues of each type
                    report_lines.append(f"  • {issue.description}")
                    if issue.suggested_fix:
                        report_lines.append(f"    → Suggested fix: {issue.suggested_fix}")
                
                if len(issues) > 3:
                    report_lines.append(f"  ... and {len(issues) - 3} more issues of this type")
        
        # Recommendations
        if validation_results['recommendations']:
            report_lines.append("\nRECOMMENDATIONS")
            report_lines.append("-" * 15)
            
            for rec in validation_results['recommendations']:
                report_lines.append(f"\n{rec['priority'].upper()} PRIORITY:")
                report_lines.append(f"  {rec['recommendation']}")
                report_lines.append(f"  Details: {rec['details']}")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)

# Import required modules
import os
from datetime import datetime

# Create default instance
academic_validator = AcademicTranslationValidator()

def validate_academic_document(original_text: str, translated_text: str, 
                              document_type: str = "academic") -> Dict[str, Any]:
    """
    Convenience function for validating academic translations
    
    Args:
        original_text: Original document text
        translated_text: Translated document text
        document_type: Type of document (academic, scientific, etc.)
    
    Returns:
        Validation report dictionary
    """
    return academic_validator.validate_academic_translation(
        original_text, translated_text, document_type
    )

def generate_validation_report_file(validation_results: Dict[str, Any], 
                                   output_file: str = "validation_report.txt") -> str:
    """
    Generate and save validation report to file
    
    Args:
        validation_results: Results from validate_academic_document
        output_file: Output file path
    
    Returns:
        Path to generated report file
    """
    report_content = academic_validator.generate_validation_report(validation_results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Validation report saved to {output_file}")
    return output_file 