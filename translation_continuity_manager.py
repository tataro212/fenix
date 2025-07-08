"""
Translation Continuity Manager for Fenix Document Translation

Provides contextual coherence across translation batches/pages to solve the user's 
identified issue of lacking continuity between translation boundaries.

This system maintains sliding context windows, terminology consistency, and narrative 
flow to ensure seamless translation across document segments.
"""

import logging
import re
import hashlib
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import time
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ContextWindow:
    """Represents a sliding context window for translation continuity"""
    
    # Context content
    previous_sentences: List[str] = field(default_factory=list)
    following_sentences: List[str] = field(default_factory=list)
    
    # Terminology tracking
    key_terms: Dict[str, str] = field(default_factory=dict)  # original -> translated
    domain_phrases: List[str] = field(default_factory=list)
    
    # Narrative context
    document_section: str = ""
    writing_style: str = ""
    current_topic: str = ""
    
    # Metadata
    batch_id: str = ""
    page_number: int = 0
    position_in_document: float = 0.0  # 0.0 to 1.0
    
    def get_context_summary(self) -> str:
        """Generate a concise context summary for translation prompts"""
        context_parts = []
        
        # Add previous context
        if self.previous_sentences:
            prev_context = " ".join(self.previous_sentences[-2:])  # Last 2 sentences
            context_parts.append(f"Previous context: {prev_context}")
        
        # Add key terminology
        if self.key_terms:
            terms_list = [f"{orig}→{trans}" for orig, trans in list(self.key_terms.items())[:5]]
            context_parts.append(f"Key terms: {', '.join(terms_list)}")
        
        # Add section/topic context
        if self.current_topic:
            context_parts.append(f"Topic: {self.current_topic}")
        
        return " | ".join(context_parts)

@dataclass
class TranslationBatch:
    """Represents a batch of content to be translated with contextual awareness"""
    
    batch_id: str
    content: List[str]  # List of text segments
    page_numbers: List[int]
    context_window: ContextWindow
    
    # Processing metadata
    estimated_tokens: int = 0
    priority: int = 2  # 1=high, 2=medium, 3=low
    processing_time: float = 0.0
    
    def get_enhanced_prompt_context(self) -> str:
        """Generate enhanced context for translation prompts"""
        prompt_parts = []
        
        # Add contextual information
        context_summary = self.context_window.get_context_summary()
        if context_summary:
            prompt_parts.append(f"Translation Context: {context_summary}")
        
        # Add document flow guidance
        if self.context_window.position_in_document < 0.2:
            prompt_parts.append("Document Flow: Introduction/Beginning section")
        elif self.context_window.position_in_document > 0.8:
            prompt_parts.append("Document Flow: Conclusion/Final section")
        else:
            prompt_parts.append("Document Flow: Main content section")
        
        return "\n".join(prompt_parts)

class TranslationContinuityManager:
    """
    Advanced Translation Continuity Manager
    
    Provides sophisticated contextual coherence across translation boundaries by:
    1. Maintaining sliding context windows across batches/pages
    2. Tracking terminology consistency throughout the document  
    3. Preserving narrative flow and discourse coherence
    4. Integrating seamlessly with existing translation strategies
    """
    
    def __init__(self, context_window_size: int = 3, terminology_cache_size: int = 200):
        """
        Initialize Translation Continuity Manager
        
        Args:
            context_window_size: Number of sentences to maintain in sliding window
            terminology_cache_size: Maximum number of terminology mappings to track
        """
        self.context_window_size = context_window_size
        self.terminology_cache_size = terminology_cache_size
        
        # Context tracking
        self.context_history = deque(maxlen=50)  # Last 50 context windows
        self.global_terminology = {}  # Global terminology mappings
        self.section_terminology = defaultdict(dict)  # Section-specific terminology
        
        # Document flow tracking
        self.document_sections = []  # List of identified sections
        self.current_section = ""
        self.section_transitions = []  # Track section boundaries
        
        # Coherence patterns
        self.discourse_markers = self._load_discourse_markers()
        self.topic_keywords = defaultdict(set)  # Topic -> keywords mapping
        
        # Performance tracking
        self.stats = {
            'batches_processed': 0,
            'terminology_hits': 0,
            'context_applications': 0,
            'coherence_improvements': 0
        }
        
        logger.info("🔗 Translation Continuity Manager initialized")
        logger.info(f"   📏 Context window: {context_window_size} sentences")
        logger.info(f"   🏷️ Terminology cache: {terminology_cache_size} terms")
    
    def _load_discourse_markers(self) -> Dict[str, List[str]]:
        """Load discourse markers for different languages"""
        return {
            'continuation': [
                'furthermore', 'moreover', 'additionally', 'in addition', 'also',
                'επιπλέον', 'επίσης', 'ακόμη', 'επιπροσθέτως'
            ],
            'contrast': [
                'however', 'nevertheless', 'on the other hand', 'whereas', 'but',
                'ωστόσο', 'εντούτοις', 'από την άλλη πλευρά', 'ενώ', 'αλλά'
            ],
            'consequence': [
                'therefore', 'thus', 'consequently', 'as a result', 'hence',
                'επομένως', 'άρα', 'κατά συνέπεια', 'ως αποτέλεσμα', 'εξ ου και'
            ],
            'elaboration': [
                'specifically', 'in particular', 'for example', 'namely', 'that is',
                'συγκεκριμένα', 'ειδικότερα', 'για παράδειγμα', 'δηλαδή', 'ήτοι'
            ]
        }
    
    def analyze_document_structure(self, document_content: List[Dict[str, Any]]) -> None:
        """
        Analyze document structure to identify sections and topics
        
        Args:
            document_content: List of content blocks with text and metadata
        """
        try:
            logger.info("🔍 Analyzing document structure for contextual awareness...")
            
            current_section = "Introduction"
            section_content = []
            
            for i, content_block in enumerate(document_content):
                text = content_block.get('text', '')
                
                # Detect section boundaries
                if self._is_section_boundary(text, i, len(document_content)):
                    # Save previous section
                    if section_content:
                        self.document_sections.append({
                            'name': current_section,
                            'content': section_content,
                            'start_index': max(0, i - len(section_content)),
                            'end_index': i
                        })
                        section_content = []
                    
                    # Start new section
                    current_section = self._extract_section_name(text)
                    self.section_transitions.append(i)
                
                section_content.append(content_block)
                
                # Extract key terminology and topics
                self._extract_terminology_from_text(text, current_section)
                self._extract_topic_keywords(text, current_section)
            
            # Add final section
            if section_content:
                self.document_sections.append({
                    'name': current_section,
                    'content': section_content,
                    'start_index': len(document_content) - len(section_content),
                    'end_index': len(document_content)
                })
            
            logger.info(f"✅ Document structure analysis complete:")
            logger.info(f"   📑 Sections identified: {len(self.document_sections)}")
            logger.info(f"   🏷️ Terminology entries: {len(self.global_terminology)}")
            logger.info(f"   🎯 Topic keywords: {sum(len(kw) for kw in self.topic_keywords.values())}")
            
        except Exception as e:
            logger.error(f"❌ Document structure analysis failed: {e}")
    
    def _is_section_boundary(self, text: str, index: int, total_blocks: int) -> bool:
        """Detect if text represents a section boundary"""
        # Check for heading patterns
        heading_patterns = [
            r'^\d+\.\s+',  # "1. Introduction"
            r'^[A-Z][A-Z\s]+$',  # "METHODOLOGY"
            r'^\w+\s*\n',  # "Introduction\n"
            r'^[IVX]+\.\s+',  # "I. Introduction"
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text.strip()):
                return True
        
        # Check for common section keywords
        section_keywords = [
            'introduction', 'background', 'methodology', 'methods', 'results', 
            'discussion', 'conclusion', 'references', 'acknowledgments',
            'εισαγωγή', 'υπόβαθρο', 'μεθοδολογία', 'αποτελέσματα', 'συζήτηση', 'συμπεράσματα'
        ]
        
        text_lower = text.lower().strip()
        for keyword in section_keywords:
            if text_lower.startswith(keyword) and len(text.strip()) < 100:
                return True
        
        return False
    
    def _extract_section_name(self, text: str) -> str:
        """Extract clean section name from heading text"""
        # Remove numbering and clean up
        clean_text = re.sub(r'^\d+\.?\s*', '', text.strip())
        clean_text = re.sub(r'^[IVX]+\.?\s*', '', clean_text)
        clean_text = clean_text.strip().title()
        
        # Limit length
        if len(clean_text) > 50:
            clean_text = clean_text[:47] + "..."
        
        return clean_text or "Section"
    
    def _extract_terminology_from_text(self, text: str, section: str) -> None:
        """Extract potentially important terminology from text"""
        # Look for technical terms, proper nouns, and domain-specific phrases
        term_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b\w+ology\b|\b\w+graphy\b|\b\w+metry\b',  # Technical suffixes
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+\s+(?:method|algorithm|approach|technique|model)\b',  # Technical phrases
        ]
        
        for pattern in term_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 3 and match.lower() not in ['the', 'and', 'for', 'with']:
                    # Store as potential terminology (will be confirmed when translated)
                    self.section_terminology[section][match] = None
    
    def _extract_topic_keywords(self, text: str, section: str) -> None:
        """Extract topic-relevant keywords from text"""
        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b\w{4,}\b', text.lower())
        
        # Filter common words
        common_words = {
            'this', 'that', 'with', 'from', 'they', 'were', 'been', 'have', 
            'their', 'said', 'each', 'which', 'what', 'where', 'when', 'than',
            'αυτό', 'είναι', 'στην', 'από', 'για', 'του', 'της', 'και', 'που'
        }
        
        meaningful_words = [w for w in words if w not in common_words and len(w) > 4]
        
        # Add to topic keywords
        for word in meaningful_words[:10]:  # Limit to top 10 per text block
            self.topic_keywords[section].add(word)
    
    def create_context_window(self, current_batch: List[str], batch_index: int, 
                            all_batches: List[List[str]], page_number: int = 1) -> ContextWindow:
        """
        Create contextual window for current batch
        
        Args:
            current_batch: Current batch of text segments to translate
            batch_index: Index of current batch in all_batches
            all_batches: All translation batches in document
            page_number: Current page number
        
        Returns:
            ContextWindow with relevant contextual information
        """
        try:
            # Calculate document position
            total_batches = len(all_batches)
            position_in_document = batch_index / max(1, total_batches - 1) if total_batches > 1 else 0.0
            
            # Get previous context (1-2 sentences from previous batch)
            previous_sentences = []
            if batch_index > 0:
                prev_batch = all_batches[batch_index - 1]
                # Extract last 1-2 sentences from previous batch
                prev_text = " ".join(prev_batch)
                prev_sentences = self._extract_sentences(prev_text)[-2:]  # Last 2 sentences
                previous_sentences = [s.strip() for s in prev_sentences if s.strip()]
            
            # Get following context (1 sentence from next batch, if available)
            following_sentences = []
            if batch_index < len(all_batches) - 1:
                next_batch = all_batches[batch_index + 1]
                next_text = " ".join(next_batch)
                next_sentences = self._extract_sentences(next_text)[:1]  # First sentence
                following_sentences = [s.strip() for s in next_sentences if s.strip()]
            
            # Determine current section and topic
            current_section = self._determine_current_section(batch_index, total_batches)
            current_topic = self._extract_current_topic(current_batch)
            
            # Get relevant terminology
            relevant_terms = self._get_relevant_terminology(current_section, current_batch)
            
            # Create context window
            context_window = ContextWindow(
                previous_sentences=previous_sentences,
                following_sentences=following_sentences,
                key_terms=relevant_terms,
                document_section=current_section,
                current_topic=current_topic,
                batch_id=f"batch_{batch_index}",
                page_number=page_number,
                position_in_document=position_in_document
            )
            
            # Add to context history
            self.context_history.append(context_window)
            
            logger.debug(f"🔗 Created context window for batch {batch_index}:")
            logger.debug(f"   📝 Previous context: {len(previous_sentences)} sentences")
            logger.debug(f"   📝 Following context: {len(following_sentences)} sentences")
            logger.debug(f"   🏷️ Key terms: {len(relevant_terms)}")
            logger.debug(f"   🎯 Topic: {current_topic}")
            
            return context_window
            
        except Exception as e:
            logger.error(f"❌ Failed to create context window for batch {batch_index}: {e}")
            # Return minimal context window
            return ContextWindow(
                batch_id=f"batch_{batch_index}",
                page_number=page_number,
                position_in_document=position_in_document
            )
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text using simple rules"""
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean and filter
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum sentence length
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def _determine_current_section(self, batch_index: int, total_batches: int) -> str:
        """Determine which document section the current batch belongs to"""
        if not self.document_sections:
            # Fallback based on position
            position = batch_index / max(1, total_batches - 1)
            if position < 0.2:
                return "Introduction"
            elif position > 0.8:
                return "Conclusion"
            else:
                return "Main Content"
        
        # Find section based on document structure analysis
        for section in self.document_sections:
            # Map batch index to content index (approximate)
            content_index = int(batch_index * len(self.document_sections) / total_batches)
            if section['start_index'] <= content_index <= section['end_index']:
                return section['name']
        
        return "Main Content"
    
    def _extract_current_topic(self, current_batch: List[str]) -> str:
        """Extract current topic from batch content"""
        combined_text = " ".join(current_batch).lower()
        
        # Look for topic indicators
        topic_patterns = [
            r'(?:discuss|examine|analyze|study|investigate)\s+(\w+(?:\s+\w+){0,2})',
            r'(?:the|this)\s+(\w+(?:\s+\w+){0,1})\s+(?:method|approach|technique|algorithm)',
            r'(\w+(?:\s+\w+){0,2})\s+(?:plays?|shows?|indicates?|suggests?)',
        ]
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, combined_text)
            if matches:
                topic = matches[0].strip().title()
                if len(topic) > 3:
                    return topic
        
        # Fallback: use most frequent meaningful words
        words = re.findall(r'\b\w{5,}\b', combined_text)
        if words:
            word_counts = defaultdict(int)
            for word in words:
                if word not in ['which', 'where', 'these', 'those', 'their']:
                    word_counts[word] += 1
            
            if word_counts:
                most_common = max(word_counts.items(), key=lambda x: x[1])
                return most_common[0].title()
        
        return "General"
    
    def _get_relevant_terminology(self, section: str, current_batch: List[str]) -> Dict[str, str]:
        """Get terminology relevant to current batch"""
        relevant_terms = {}
        
        # Get section-specific terminology
        section_terms = self.section_terminology.get(section, {})
        
        # Get global terminology that appears in current batch
        combined_text = " ".join(current_batch).lower()
        
        for term, translation in self.global_terminology.items():
            if term.lower() in combined_text and translation:
                relevant_terms[term] = translation
        
        # Add section-specific terms that have translations
        for term, translation in section_terms.items():
            if translation and term.lower() in combined_text:
                relevant_terms[term] = translation
        
        # Limit to most relevant terms
        return dict(list(relevant_terms.items())[:10])
    
    def enhance_translation_prompt(self, original_prompt: str, context_window: ContextWindow) -> str:
        """
        Enhance translation prompt with contextual information
        
        Args:
            original_prompt: Original translation prompt
            context_window: Context window for current batch
        
        Returns:
            Enhanced prompt with contextual coherence guidance
        """
        try:
            enhanced_prompt_parts = [original_prompt]
            
            # Add contextual coherence section
            context_parts = []
            
            # Previous context for continuity
            if context_window.previous_sentences:
                prev_context = " ".join(context_window.previous_sentences[-1:])  # Last sentence
                context_parts.append(f"Previous context: \"{prev_context}\"")
            
            # Key terminology for consistency
            if context_window.key_terms:
                terms_list = [f"{orig} → {trans}" for orig, trans in list(context_window.key_terms.items())[:3]]
                context_parts.append(f"Use consistent terminology: {'; '.join(terms_list)}")
            
            # Document flow guidance
            flow_guidance = self._generate_flow_guidance(context_window)
            if flow_guidance:
                context_parts.append(flow_guidance)
            
            # Add coherence instructions
            if context_parts:
                coherence_section = (
                    "\n\nCONTEXTUAL COHERENCE GUIDANCE:\n"
                    + "\n".join(f"• {part}" for part in context_parts)
                    + "\n• Maintain narrative flow and terminology consistency with previous content"
                    + "\n• Ensure smooth transitions between sentences and paragraphs"
                )
                enhanced_prompt_parts.append(coherence_section)
            
            self.stats['context_applications'] += 1
            
            enhanced_prompt = "\n".join(enhanced_prompt_parts)
            
            logger.debug(f"🔗 Enhanced translation prompt with contextual guidance")
            logger.debug(f"   📏 Context elements: {len(context_parts)}")
            logger.debug(f"   🎯 Flow guidance: {'Yes' if flow_guidance else 'No'}")
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"❌ Failed to enhance translation prompt: {e}")
            return original_prompt
    
    def _generate_flow_guidance(self, context_window: ContextWindow) -> str:
        """Generate document flow guidance based on context"""
        guidance_parts = []
        
        # Position-based guidance
        if context_window.position_in_document < 0.1:
            guidance_parts.append("Beginning of document - use introductory tone")
        elif context_window.position_in_document > 0.9:
            guidance_parts.append("End of document - use concluding tone")
        
        # Section-based guidance
        if "introduction" in context_window.document_section.lower():
            guidance_parts.append("Introduction section - establish context and objectives")
        elif "conclusion" in context_window.document_section.lower():
            guidance_parts.append("Conclusion section - summarize and finalize")
        elif "method" in context_window.document_section.lower():
            guidance_parts.append("Methodology section - use precise technical language")
        
        # Topic-based guidance
        if context_window.current_topic and context_window.current_topic != "General":
            guidance_parts.append(f"Focus topic: {context_window.current_topic}")
        
        return "; ".join(guidance_parts) if guidance_parts else ""
    
    def update_terminology_mapping(self, original_terms: List[str], translated_terms: List[str], 
                                 section: str = "") -> None:
        """
        Update terminology mappings from successful translations
        
        Args:
            original_terms: List of original language terms
            translated_terms: List of corresponding translated terms
            section: Document section for context-specific tracking
        """
        try:
            for orig, trans in zip(original_terms, translated_terms):
                if orig and trans and orig != trans:
                    # Update global terminology
                    self.global_terminology[orig] = trans
                    
                    # Update section-specific terminology
                    if section:
                        self.section_terminology[section][orig] = trans
                    
                    self.stats['terminology_hits'] += 1
            
            # Limit cache size
            if len(self.global_terminology) > self.terminology_cache_size:
                # Remove oldest entries (simple FIFO)
                items_to_remove = len(self.global_terminology) - self.terminology_cache_size
                for _ in range(items_to_remove):
                    oldest_key = next(iter(self.global_terminology))
                    del self.global_terminology[oldest_key]
            
            logger.debug(f"🏷️ Updated terminology mappings: {len(original_terms)} new entries")
            
        except Exception as e:
            logger.error(f"❌ Failed to update terminology mapping: {e}")
    
    def get_continuity_statistics(self) -> Dict[str, Any]:
        """Get comprehensive continuity management statistics"""
        return {
            'context_windows_created': len(self.context_history),
            'terminology_mappings': len(self.global_terminology),
            'document_sections': len(self.document_sections),
            'context_applications': self.stats['context_applications'],
            'terminology_hits': self.stats['terminology_hits'],
            'coherence_improvements': self.stats['coherence_improvements'],
            'average_context_size': sum(len(cw.previous_sentences) + len(cw.following_sentences) 
                                      for cw in self.context_history) / max(1, len(self.context_history)),
            'section_coverage': len(self.section_terminology),
            'topic_keywords_total': sum(len(keywords) for keywords in self.topic_keywords.values())
        }
    
    def export_continuity_data(self, output_path: str) -> None:
        """Export continuity data for analysis or reuse"""
        try:
            continuity_data = {
                'global_terminology': self.global_terminology,
                'section_terminology': dict(self.section_terminology),
                'document_sections': self.document_sections,
                'topic_keywords': {k: list(v) for k, v in self.topic_keywords.items()},
                'statistics': self.get_continuity_statistics(),
                'export_timestamp': time.time()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(continuity_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📁 Exported continuity data to: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export continuity data: {e}")
    
    def import_continuity_data(self, input_path: str) -> None:
        """Import previously saved continuity data"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                continuity_data = json.load(f)
            
            self.global_terminology = continuity_data.get('global_terminology', {})
            self.section_terminology = defaultdict(dict, continuity_data.get('section_terminology', {}))
            self.document_sections = continuity_data.get('document_sections', [])
            
            topic_keywords = continuity_data.get('topic_keywords', {})
            self.topic_keywords = defaultdict(set)
            for k, v in topic_keywords.items():
                self.topic_keywords[k] = set(v)
            
            logger.info(f"📁 Imported continuity data from: {input_path}")
            logger.info(f"   🏷️ Terminology entries: {len(self.global_terminology)}")
            logger.info(f"   📑 Document sections: {len(self.document_sections)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to import continuity data: {e}")


# Global instance for easy access
translation_continuity_manager = TranslationContinuityManager() 