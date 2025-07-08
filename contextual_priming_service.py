"""
Contextual Priming Service for Enhanced Translation Quality

This service implements the user's idea of analyzing document context at startup
and using that information to prime all translation prompts for higher specificity
and better domain-aware translations.

Key Features:
- Document domain analysis
- Technical terminology extraction
- Style and tone detection
- Subject matter identification
- Context caching for efficiency
- Integration with existing translation services
"""

import asyncio
import json
import logging
import os
import re
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import Counter
import time

# Import existing services
from config_manager import config_manager
from utils import get_cache_key

logger = logging.getLogger(__name__)

@dataclass
class DocumentContext:
    """Comprehensive document context for translation priming"""
    
    # Core document information
    document_type: str  # academic, technical, legal, medical, business, etc.
    domain: str  # specific field like "machine learning", "contract law", etc.
    subject_matter: str  # main topic/theme
    
    # Style and tone
    writing_style: str  # formal, informal, technical, conversational
    tone: str  # authoritative, explanatory, persuasive, neutral
    audience: str  # expert, general public, students, professionals
    
    # Technical aspects
    key_terminology: Dict[str, int]  # term -> frequency
    technical_level: str  # basic, intermediate, advanced, expert
    jargon_density: float  # ratio of technical terms to total words
    
    # Linguistic features
    sentence_complexity: str  # simple, compound, complex
    vocabulary_level: str  # basic, intermediate, advanced
    
    # Translation guidance
    translation_priorities: List[str]  # ordered list of what to prioritize
    domain_specific_instructions: str  # specific instructions for this domain
    terminology_consistency_rules: Dict[str, str]  # term -> preferred translation
    
    # Metadata
    analysis_confidence: float  # 0.0 to 1.0
    sample_size: int  # number of characters analyzed
    creation_timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentContext':
        """Create from dictionary"""
        return cls(**data)

class DocumentContextAnalyzer:
    """Analyzes documents to extract contextual information for translation priming"""
    
    def __init__(self):
        self.gemini_settings = config_manager.gemini_settings
        self.model = None
        
        # Initialize Gemini model for analysis
        if self.gemini_settings['api_key']:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_settings['api_key'])
            self.model = genai.GenerativeModel(self.gemini_settings['model_name'])
        
        # Domain-specific patterns and indicators
        self.domain_patterns = {
            'academic': [
                r'\b(abstract|introduction|methodology|results|discussion|conclusion|bibliography|references)\b',
                r'\b(hypothesis|research|study|analysis|findings|significant|correlation)\b',
                r'\b(et al\.|ibid\.|op\. cit\.|cf\.|viz\.|i\.e\.|e\.g\.)\b'
            ],
            'technical': [
                r'\b(algorithm|implementation|system|architecture|framework|protocol)\b',
                r'\b(API|SDK|CPU|GPU|RAM|database|server|client|interface)\b',
                r'\b(function|method|class|variable|parameter|return|exception)\b'
            ],
            'medical': [
                r'\b(patient|diagnosis|treatment|symptoms|therapy|clinical|medical)\b',
                r'\b(disease|syndrome|disorder|condition|pathology|etiology)\b',
                r'\b(mg|ml|dose|administration|prescription|contraindication)\b'
            ],
            'legal': [
                r'\b(contract|agreement|clause|provision|liability|jurisdiction)\b',
                r'\b(plaintiff|defendant|court|judge|jury|verdict|appeal)\b',
                r'\b(statute|regulation|compliance|violation|penalty|damages)\b'
            ],
            'business': [
                r'\b(revenue|profit|loss|investment|market|customer|client)\b',
                r'\b(strategy|management|operations|finance|accounting|budget)\b',
                r'\b(KPI|ROI|B2B|B2C|CEO|CFO|CTO|stakeholder)\b'
            ],
            'scientific': [
                r'\b(experiment|hypothesis|variable|control|sample|data|statistics)\b',
                r'\b(theory|model|phenomenon|observation|measurement|analysis)\b',
                r'\b(significant|correlation|variance|standard deviation|p-value)\b'
            ]
        }
        
        # Technical terminology patterns
        self.technical_patterns = [
            r'\b[A-Z]{2,}(?:[A-Z][a-z]*)*\b',  # Acronyms
            r'\b\w+(?:-\w+)+\b',  # Hyphenated technical terms
            r'\b\w*[Tt]ech\w*\b',  # Tech-related words
            r'\b\w*[Ss]ystem\w*\b',  # System-related words
            r'\b\w*[Aa]lgorithm\w*\b',  # Algorithm-related words
        ]
    
    async def analyze_document_context(self, text_sample: str, 
                                     document_title: str = "",
                                     max_sample_size: int = 8000) -> DocumentContext:
        """
        Analyze document context using both pattern matching and AI analysis
        """
        logger.info(f"🔍 Analyzing document context (sample size: {len(text_sample)} chars)")
        
        # Truncate sample if too large
        if len(text_sample) > max_sample_size:
            text_sample = text_sample[:max_sample_size]
        
        # Pattern-based analysis
        pattern_results = self._analyze_patterns(text_sample)
        
        # AI-powered analysis
        ai_results = await self._analyze_with_ai(text_sample, document_title)
        
        # Combine results
        context = self._combine_analysis_results(pattern_results, ai_results, text_sample)
        
        logger.info(f"✅ Document context analysis complete:")
        logger.info(f"   Domain: {context.domain}")
        logger.info(f"   Type: {context.document_type}")
        logger.info(f"   Style: {context.writing_style}")
        logger.info(f"   Technical Level: {context.technical_level}")
        logger.info(f"   Confidence: {context.analysis_confidence:.2f}")
        
        return context
    
    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text using pattern matching"""
        results = {
            'domain_scores': {},
            'technical_terms': [],
            'sentence_complexity': 'simple',
            'jargon_density': 0.0
        }
        
        # Domain detection
        for domain, patterns in self.domain_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            results['domain_scores'][domain] = score
        
        # Technical terminology extraction
        for pattern in self.technical_patterns:
            matches = re.findall(pattern, text)
            results['technical_terms'].extend(matches)
        
        # Calculate jargon density
        total_words = len(text.split())
        technical_words = len(results['technical_terms'])
        results['jargon_density'] = technical_words / max(total_words, 1)
        
        # Sentence complexity analysis
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if avg_sentence_length > 25:
            results['sentence_complexity'] = 'complex'
        elif avg_sentence_length > 15:
            results['sentence_complexity'] = 'compound'
        else:
            results['sentence_complexity'] = 'simple'
        
        return results
    
    async def _analyze_with_ai(self, text: str, title: str = "") -> Dict[str, Any]:
        """Use AI to analyze document context"""
        if not self.model:
            logger.warning("AI analysis unavailable - no Gemini model")
            return {}
        
        try:
            # Create comprehensive analysis prompt
            analysis_prompt = f"""
Analyze the following document excerpt and provide a comprehensive context analysis for translation purposes.

Document Title: {title}

Document Text:
{text}

Please analyze and provide the following information in JSON format:

{{
    "document_type": "academic|technical|legal|medical|business|scientific|literary|other",
    "domain": "specific field or subject area",
    "subject_matter": "main topic or theme",
    "writing_style": "formal|informal|technical|conversational|academic",
    "tone": "authoritative|explanatory|persuasive|neutral|critical|supportive",
    "audience": "expert|general|students|professionals|laypeople",
    "technical_level": "basic|intermediate|advanced|expert",
    "vocabulary_level": "basic|intermediate|advanced",
    "key_concepts": ["list", "of", "main", "concepts"],
    "translation_priorities": ["accuracy|fluency|terminology|style", "in", "order"],
    "domain_instructions": "specific instructions for translating this type of content",
    "confidence": 0.85
}}

Provide ONLY the JSON response, no additional text.
"""
            
            response = await self.model.generate_content_async(
                analysis_prompt,
                generation_config={
                    'temperature': 0.3,  # Lower temperature for more consistent analysis
                    'max_output_tokens': 1000
                }
            )
            
            if response and response.text:
                # Parse JSON response
                try:
                    ai_analysis = json.loads(response.text.strip())
                    return ai_analysis
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse AI analysis JSON: {e}")
                    # Try to extract JSON from response
                    json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    if json_match:
                        try:
                            ai_analysis = json.loads(json_match.group())
                            return ai_analysis
                        except json.JSONDecodeError:
                            pass
            
            logger.warning("AI analysis failed - using pattern-based analysis only")
            return {}
            
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return {}
    
    def _combine_analysis_results(self, pattern_results: Dict[str, Any], 
                                ai_results: Dict[str, Any], 
                                text_sample: str) -> DocumentContext:
        """Combine pattern-based and AI analysis results"""
        
        # Determine document type and domain
        if ai_results.get('document_type'):
            document_type = ai_results['document_type']
        else:
            # Use pattern-based domain detection
            domain_scores = pattern_results.get('domain_scores', {})
            document_type = max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
        
        domain = ai_results.get('domain', document_type)
        
        # Extract key terminology
        key_terminology = {}
        technical_terms = pattern_results.get('technical_terms', [])
        term_counts = Counter(technical_terms)
        key_terminology.update(term_counts)
        
        # Add AI-identified concepts
        if ai_results.get('key_concepts'):
            for concept in ai_results['key_concepts']:
                key_terminology[concept] = key_terminology.get(concept, 0) + 1
        
        # Determine technical level
        jargon_density = pattern_results.get('jargon_density', 0.0)
        if jargon_density > 0.3:
            technical_level = 'expert'
        elif jargon_density > 0.15:
            technical_level = 'advanced'
        elif jargon_density > 0.05:
            technical_level = 'intermediate'
        else:
            technical_level = 'basic'
        
        # Override with AI analysis if available
        technical_level = ai_results.get('technical_level', technical_level)
        
        # Create translation priorities
        translation_priorities = ai_results.get('translation_priorities', [
            'accuracy', 'terminology', 'style', 'fluency'
        ])
        
        # Generate domain-specific instructions
        domain_instructions = ai_results.get('domain_instructions', 
                                           self._generate_domain_instructions(document_type, domain))
        
        # Calculate confidence
        ai_confidence = ai_results.get('confidence', 0.0)
        pattern_confidence = 0.7 if pattern_results.get('domain_scores') else 0.3
        overall_confidence = (ai_confidence + pattern_confidence) / 2 if ai_confidence > 0 else pattern_confidence
        
        return DocumentContext(
            document_type=document_type,
            domain=domain,
            subject_matter=ai_results.get('subject_matter', domain),
            writing_style=ai_results.get('writing_style', 'formal'),
            tone=ai_results.get('tone', 'neutral'),
            audience=ai_results.get('audience', 'general'),
            key_terminology=key_terminology,
            technical_level=technical_level,
            jargon_density=jargon_density,
            sentence_complexity=pattern_results.get('sentence_complexity', 'simple'),
            vocabulary_level=ai_results.get('vocabulary_level', 'intermediate'),
            translation_priorities=translation_priorities,
            domain_specific_instructions=domain_instructions,
            terminology_consistency_rules={},  # Will be populated later
            analysis_confidence=overall_confidence,
            sample_size=len(text_sample),
            creation_timestamp=time.time()
        )
    
    def _generate_domain_instructions(self, document_type: str, domain: str) -> str:
        """Generate domain-specific translation instructions"""
        instructions = {
            'academic': "Maintain academic tone and precision. Preserve technical terminology. Use formal language appropriate for scholarly discourse.",
            'technical': "Prioritize technical accuracy. Maintain consistency in technical terms. Preserve code snippets and technical specifications.",
            'medical': "Ensure medical accuracy. Use standard medical terminology. Maintain precision in dosages, procedures, and diagnoses.",
            'legal': "Maintain legal precision. Use appropriate legal terminology. Preserve the exact meaning of legal concepts and obligations.",
            'business': "Use professional business language. Maintain clarity for business stakeholders. Preserve financial and operational terminology.",
            'scientific': "Maintain scientific accuracy. Use precise scientific terminology. Preserve mathematical expressions and scientific notation."
        }
        
        base_instruction = instructions.get(document_type, "Maintain the document's professional tone and terminology.")
        return f"{base_instruction} Domain focus: {domain}."

class ContextualPrimingCache:
    """Manages caching of document context analysis"""
    
    def __init__(self, cache_file: str = "contextual_priming_cache.json"):
        self.cache_file = cache_file
        self.cache = {}
        self.load_cache()
    
    def load_cache(self):
        """Load context cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    # Convert dict entries back to DocumentContext objects
                    for key, value in cache_data.items():
                        self.cache[key] = DocumentContext.from_dict(value)
                logger.info(f"Loaded {len(self.cache)} cached context analyses")
            except Exception as e:
                logger.error(f"Error loading context cache: {e}")
                self.cache = {}
    
    def save_cache(self):
        """Save context cache to file"""
        try:
            # Convert DocumentContext objects to dicts for JSON serialization
            cache_data = {key: context.to_dict() for key, context in self.cache.items()}
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.cache)} context analyses to cache")
        except Exception as e:
            logger.error(f"Error saving context cache: {e}")
    
    def get_context(self, document_id: str) -> Optional[DocumentContext]:
        """Get cached context for a document"""
        return self.cache.get(document_id)
    
    def cache_context(self, document_id: str, context: DocumentContext):
        """Cache a document context"""
        self.cache[document_id] = context
    
    def generate_document_id(self, text_sample: str, title: str = "") -> str:
        """Generate a unique ID for a document based on its content"""
        content = f"{title}|{text_sample[:2000]}"  # Use first 2000 chars for ID
        return hashlib.md5(content.encode('utf-8')).hexdigest()

class ContextualPrimingService:
    """
    Main service that provides contextual priming for translation
    
    This implements the user's idea of analyzing document context at startup
    and using that information to enhance all translation prompts.
    """
    
    def __init__(self):
        self.analyzer = DocumentContextAnalyzer()
        self.cache = ContextualPrimingCache()
        self.current_context: Optional[DocumentContext] = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize_document_context(self, text_sample: str, 
                                        document_title: str = "",
                                        force_reanalysis: bool = False) -> DocumentContext:
        """
        Initialize document context at the start of translation process
        
        This is the main entry point for the contextual priming system.
        Call this before starting any translation work.
        """
        self.logger.info("🚀 Initializing contextual priming system...")
        
        # Generate document ID for caching
        document_id = self.cache.generate_document_id(text_sample, document_title)
        
        # Check cache first
        if not force_reanalysis:
            cached_context = self.cache.get_context(document_id)
            if cached_context:
                self.logger.info("📋 Using cached document context analysis")
                self.current_context = cached_context
                return cached_context
        
        # Perform new analysis
        self.logger.info("🔍 Performing new document context analysis...")
        context = await self.analyzer.analyze_document_context(text_sample, document_title)
        
        # Cache the result
        self.cache.cache_context(document_id, context)
        self.cache.save_cache()
        
        # Set as current context
        self.current_context = context
        
        self.logger.info("✅ Contextual priming system initialized successfully")
        return context
    
    def get_contextual_prompt_enhancement(self, target_language: str) -> str:
        """
        Generate contextual enhancement text to be added to translation prompts
        
        This is the core of the contextual priming system - it provides
        domain-specific context that will be included in every translation prompt.
        """
        if not self.current_context:
            return ""
        
        context = self.current_context
        
        # Build comprehensive contextual guidance
        enhancement_parts = [
            f"🎯 CONTEXTUAL TRANSLATION GUIDANCE:",
            f"Document Type: {context.document_type.title()}",
            f"Domain: {context.domain}",
            f"Subject Matter: {context.subject_matter}",
            f"Style: {context.writing_style} | Tone: {context.tone}",
            f"Audience: {context.audience} | Technical Level: {context.technical_level}",
            "",
            f"🔑 DOMAIN-SPECIFIC INSTRUCTIONS:",
            context.domain_specific_instructions,
            ""
        ]
        
        # Add key terminology guidance
        if context.key_terminology:
            top_terms = sorted(context.key_terminology.items(), 
                             key=lambda x: x[1], reverse=True)[:10]
            enhancement_parts.extend([
                f"🏷️ KEY TERMINOLOGY (maintain consistency):",
                ", ".join([term for term, _ in top_terms]),
                ""
            ])
        
        # Add translation priorities
        if context.translation_priorities:
            enhancement_parts.extend([
                f"📋 TRANSLATION PRIORITIES:",
                " > ".join(context.translation_priorities),
                ""
            ])
        
        # Add specific guidance based on technical level
        if context.technical_level in ['advanced', 'expert']:
            enhancement_parts.append(
                "⚡ HIGH TECHNICAL PRECISION REQUIRED: Maintain exact technical terminology and concepts."
            )
        
        return "\n".join(enhancement_parts)
    
    def get_context_summary(self) -> str:
        """Get a brief summary of the current context"""
        if not self.current_context:
            return "No context available"
        
        context = self.current_context
        return (f"{context.document_type.title()} document in {context.domain} "
                f"({context.technical_level} level, {context.writing_style} style)")
    
    def get_terminology_guidance(self) -> Dict[str, str]:
        """Get terminology consistency guidance"""
        if not self.current_context:
            return {}
        
        return self.current_context.terminology_consistency_rules
    
    def update_terminology_rules(self, term_translations: Dict[str, str]):
        """Update terminology consistency rules based on actual translations"""
        if not self.current_context:
            return
        
        self.current_context.terminology_consistency_rules.update(term_translations)
        
        # Update cache
        document_id = self.cache.generate_document_id("", "")  # Simplified for update
        self.cache.cache_context(document_id, self.current_context)
        self.cache.save_cache()

# Global instance
contextual_priming_service = ContextualPrimingService() 