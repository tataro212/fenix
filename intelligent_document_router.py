"""
Intelligent Document Router for Fenix PDF Translation Pipeline

This module analyzes document complexity and routes documents to the most appropriate
processing pipeline for optimal performance and quality.
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import our analysis and processing modules
from document_complexity_analyzer import (
    DocumentComplexityAnalyzer, 
    ComplexityAnalysis, 
    DocumentComplexity,
    document_complexity_analyzer
)
from simple_document_processor import (
    SimpleDocumentProcessor,
    SimpleProcessingResult,
    simple_document_processor
)

logger = logging.getLogger(__name__)

class ProcessingStrategy(Enum):
    """Available processing strategies"""
    FAST_TRACK_SIMPLE = "fast_track_simple"
    HYBRID_PROCESSING = "hybrid_processing"
    FULL_DIGITAL_TWIN = "full_digital_twin"

@dataclass
class RoutingDecision:
    """Document routing decision with rationale"""
    strategy: ProcessingStrategy
    confidence: float
    analysis: ComplexityAnalysis
    reasoning: str
    estimated_time_minutes: float
    estimated_cost_factor: float  # Relative to simple processing

@dataclass
class ProcessingResult:
    """Unified processing result"""
    success: bool
    output_path: str
    strategy_used: ProcessingStrategy
    processing_time: float
    complexity_analysis: ComplexityAnalysis
    routing_decision: RoutingDecision
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None

class IntelligentDocumentRouter:
    """
    Intelligent routing system that analyzes documents and selects optimal processing strategy.
    
    Processing Strategies:
    1. FAST_TRACK_SIMPLE: Direct text extraction + translation for simple documents
    2. HYBRID_PROCESSING: Selective Digital Twin features for moderate complexity
    3. FULL_DIGITAL_TWIN: Complete structure-preserving processing for complex documents
    """
    
    def __init__(self, gemini_service=None):
        self.logger = logging.getLogger(__name__)
        self.gemini_service = gemini_service
        
        # Initialize processors
        self.complexity_analyzer = document_complexity_analyzer
        self.simple_processor = simple_document_processor
        self.simple_processor.gemini_service = gemini_service
        
        # Performance tracking
        self.performance_stats = {
            'total_documents_processed': 0,
            'fast_track_count': 0,
            'hybrid_count': 0,
            'digital_twin_count': 0,
            'average_processing_time': 0.0,
            'success_rate': 0.0
        }
    
    async def route_and_process_document(
        self, 
        pdf_path: str, 
        output_path: str, 
        target_language: str = "el",
        force_strategy: Optional[ProcessingStrategy] = None
    ) -> ProcessingResult:
        """
        Analyze document complexity and route to appropriate processing pipeline.
        
        Args:
            pdf_path: Path to input PDF
            output_path: Path for output Word document
            target_language: Target language code
            force_strategy: Optional strategy override
            
        Returns:
            ProcessingResult with comprehensive details
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.info(f"🎯 Starting intelligent document routing")
            self.logger.info(f"   Input: {os.path.basename(pdf_path)}")
            self.logger.info(f"   Output: {os.path.basename(output_path)}")
            
            # Step 1: Analyze document complexity
            self.logger.info("📊 Analyzing document complexity...")
            complexity_analysis = self.complexity_analyzer.analyze_document(pdf_path)
            
            # Step 2: Make routing decision
            if force_strategy:
                routing_decision = RoutingDecision(
                    strategy=force_strategy,
                    confidence=1.0,
                    analysis=complexity_analysis,
                    reasoning="User-specified strategy override",
                    estimated_time_minutes=self._estimate_processing_time(force_strategy, complexity_analysis),
                    estimated_cost_factor=self._estimate_cost_factor(force_strategy)
                )
            else:
                routing_decision = self._make_routing_decision(complexity_analysis)
            
            self.logger.info(f"🚦 Routing Decision: {routing_decision.strategy.value.upper()}")
            self.logger.info(f"   Confidence: {routing_decision.confidence:.2f}")
            self.logger.info(f"   Reasoning: {routing_decision.reasoning}")
            self.logger.info(f"   Estimated time: {routing_decision.estimated_time_minutes:.1f} minutes")
            
            # Step 3: Execute processing strategy
            processing_result = await self._execute_processing_strategy(
                routing_decision, 
                pdf_path, 
                output_path, 
                target_language
            )
            
            # Step 4: Update performance statistics
            self._update_performance_stats(routing_decision, processing_result)
            
            total_time = asyncio.get_event_loop().time() - start_time
            
            return ProcessingResult(
                success=processing_result.get('success', False),
                output_path=output_path,
                strategy_used=routing_decision.strategy,
                processing_time=total_time,
                complexity_analysis=complexity_analysis,
                routing_decision=routing_decision,
                error_message=processing_result.get('error_message'),
                performance_metrics=processing_result.get('performance_metrics')
            )
            
        except Exception as e:
            self.logger.error(f"❌ Document routing failed: {e}")
            return ProcessingResult(
                success=False,
                output_path=output_path,
                strategy_used=ProcessingStrategy.FULL_DIGITAL_TWIN,  # Default fallback
                processing_time=asyncio.get_event_loop().time() - start_time,
                complexity_analysis=complexity_analysis if 'complexity_analysis' in locals() else None,
                routing_decision=None,
                error_message=str(e)
            )
    
    def _make_routing_decision(self, analysis: ComplexityAnalysis) -> RoutingDecision:
        """
        Make intelligent routing decision based on complexity analysis.
        
        Args:
            analysis: Document complexity analysis
            
        Returns:
            RoutingDecision with strategy and reasoning
        """
        # Decision matrix based on complexity factors
        decision_factors = {
            'complexity_level': analysis.complexity_level,
            'confidence': analysis.confidence_score,
            'page_count': analysis.page_count,
            'has_images': analysis.has_images,
            'has_tables': analysis.has_tables,
            'has_toc': analysis.has_toc,
            'has_complex_formatting': analysis.has_complex_formatting,
            'font_diversity': analysis.font_diversity
        }
        
        # Fast Track Decision Logic
        if (analysis.complexity_level == DocumentComplexity.SIMPLE and
            analysis.confidence_score > 0.8 and
            not analysis.has_images and
            not analysis.has_tables and
            analysis.font_diversity <= 3):
            
            return RoutingDecision(
                strategy=ProcessingStrategy.FAST_TRACK_SIMPLE,
                confidence=analysis.confidence_score,
                analysis=analysis,
                reasoning="Simple document with high confidence - optimal for fast-track processing",
                estimated_time_minutes=self._estimate_processing_time(ProcessingStrategy.FAST_TRACK_SIMPLE, analysis),
                estimated_cost_factor=1.0
            )
        
        # Hybrid Processing Decision Logic
        elif (analysis.complexity_level == DocumentComplexity.MODERATE or
              (analysis.complexity_level == DocumentComplexity.SIMPLE and 
               (analysis.has_images or analysis.has_tables or analysis.font_diversity > 3))):
            
            return RoutingDecision(
                strategy=ProcessingStrategy.HYBRID_PROCESSING,
                confidence=max(0.6, analysis.confidence_score - 0.1),
                analysis=analysis,
                reasoning="Moderate complexity - hybrid processing balances speed and quality",
                estimated_time_minutes=self._estimate_processing_time(ProcessingStrategy.HYBRID_PROCESSING, analysis),
                estimated_cost_factor=2.5
            )
        
        # Full Digital Twin Decision Logic (default for complex documents)
        else:
            return RoutingDecision(
                strategy=ProcessingStrategy.FULL_DIGITAL_TWIN,
                confidence=analysis.confidence_score,
                analysis=analysis,
                reasoning="Complex document requires full Digital Twin processing for structure preservation",
                estimated_time_minutes=self._estimate_processing_time(ProcessingStrategy.FULL_DIGITAL_TWIN, analysis),
                estimated_cost_factor=5.0
            )
    
    def _estimate_processing_time(self, strategy: ProcessingStrategy, analysis: ComplexityAnalysis) -> float:
        """Estimate processing time in minutes based on strategy and complexity"""
        base_time_per_page = {
            ProcessingStrategy.FAST_TRACK_SIMPLE: 0.1,  # 6 seconds per page
            ProcessingStrategy.HYBRID_PROCESSING: 0.3,  # 18 seconds per page
            ProcessingStrategy.FULL_DIGITAL_TWIN: 0.8   # 48 seconds per page
        }
        
        base_time = base_time_per_page[strategy] * analysis.page_count
        
        # Adjust for complexity factors
        complexity_multiplier = 1.0
        if analysis.has_images:
            complexity_multiplier += 0.2
        if analysis.has_tables:
            complexity_multiplier += 0.3
        if analysis.has_complex_formatting:
            complexity_multiplier += 0.2
        if analysis.font_diversity > 5:
            complexity_multiplier += 0.1
        
        return base_time * complexity_multiplier
    
    def _estimate_cost_factor(self, strategy: ProcessingStrategy) -> float:
        """Estimate relative cost factor compared to simple processing"""
        cost_factors = {
            ProcessingStrategy.FAST_TRACK_SIMPLE: 1.0,
            ProcessingStrategy.HYBRID_PROCESSING: 2.5,
            ProcessingStrategy.FULL_DIGITAL_TWIN: 5.0
        }
        return cost_factors[strategy]
    
    async def _execute_processing_strategy(
        self, 
        routing_decision: RoutingDecision, 
        pdf_path: str, 
        output_path: str, 
        target_language: str
    ) -> Dict[str, Any]:
        """
        Execute the selected processing strategy.
        
        Args:
            routing_decision: Routing decision with strategy
            pdf_path: Input PDF path
            output_path: Output Word document path
            target_language: Target language code
            
        Returns:
            Processing result dictionary
        """
        strategy = routing_decision.strategy
        analysis = routing_decision.analysis
        
        try:
            if strategy == ProcessingStrategy.FAST_TRACK_SIMPLE:
                return await self._execute_fast_track_processing(
                    pdf_path, output_path, target_language, analysis
                )
            
            elif strategy == ProcessingStrategy.HYBRID_PROCESSING:
                return await self._execute_hybrid_processing(
                    pdf_path, output_path, target_language, analysis
                )
            
            elif strategy == ProcessingStrategy.FULL_DIGITAL_TWIN:
                return await self._execute_digital_twin_processing(
                    pdf_path, output_path, target_language, analysis
                )
            
            else:
                raise ValueError(f"Unknown processing strategy: {strategy}")
                
        except Exception as e:
            self.logger.error(f"Processing strategy execution failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'performance_metrics': {'execution_error': True}
            }
    
    async def _execute_fast_track_processing(
        self, 
        pdf_path: str, 
        output_path: str, 
        target_language: str, 
        analysis: ComplexityAnalysis
    ) -> Dict[str, Any]:
        """Execute fast-track simple processing"""
        self.logger.info("🚀 Executing fast-track simple processing")
        
        result = await self.simple_processor.process_simple_document(
            pdf_path, output_path, target_language
        )
        
        return {
            'success': result.success,
            'error_message': result.error_message,
            'performance_metrics': {
                'text_blocks_processed': result.text_blocks_processed,
                'translation_time': result.translation_time,
                'total_time': result.total_time,
                'processing_strategy': 'fast_track_simple'
            }
        }
    
    async def _execute_hybrid_processing(
        self, 
        pdf_path: str, 
        output_path: str, 
        target_language: str, 
        analysis: ComplexityAnalysis
    ) -> Dict[str, Any]:
        """Execute hybrid processing (selective Digital Twin features)"""
        self.logger.info("🔄 Executing hybrid processing")
        
        # For now, fall back to Digital Twin processing
        # TODO: Implement true hybrid processing that selectively uses Digital Twin features
        return await self._execute_digital_twin_processing(
            pdf_path, output_path, target_language, analysis
        )
    
    async def _execute_digital_twin_processing(
        self, 
        pdf_path: str, 
        output_path: str, 
        target_language: str, 
        analysis: ComplexityAnalysis
    ) -> Dict[str, Any]:
        """Execute full Digital Twin processing"""
        self.logger.info("🏗️ Executing full Digital Twin processing")
        
        try:
            # Import Digital Twin components
            from processing_strategies import ProcessingStrategyExecutor
            from gemini_service import GeminiService
            
            # Initialize Digital Twin processor
            if not self.gemini_service:
                self.gemini_service = GeminiService()
            
            strategy_executor = ProcessingStrategyExecutor(self.gemini_service)
            
            # Configure TOC handling based on analysis
            if hasattr(strategy_executor, 'document_generator'):
                strategy_executor.document_generator._skip_toc_generation = not analysis.has_toc
            
            # Execute Digital Twin strategy
            result = await strategy_executor.execute_strategy_digital_twin(
                pdf_path=pdf_path,
                output_dir=os.path.dirname(output_path),
                target_language=target_language
            )
            
            return {
                'success': result.success if hasattr(result, 'success') else True,
                'error_message': None,
                'performance_metrics': {
                    'processing_strategy': 'full_digital_twin',
                    'toc_generation_skipped': not analysis.has_toc,
                    'digital_twin_result': str(result)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Digital Twin processing failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'performance_metrics': {'digital_twin_error': True}
            }
    
    def _update_performance_stats(self, routing_decision: RoutingDecision, processing_result: Dict[str, Any]):
        """Update performance statistics"""
        self.performance_stats['total_documents_processed'] += 1
        
        if routing_decision.strategy == ProcessingStrategy.FAST_TRACK_SIMPLE:
            self.performance_stats['fast_track_count'] += 1
        elif routing_decision.strategy == ProcessingStrategy.HYBRID_PROCESSING:
            self.performance_stats['hybrid_count'] += 1
        elif routing_decision.strategy == ProcessingStrategy.FULL_DIGITAL_TWIN:
            self.performance_stats['digital_twin_count'] += 1
        
        # Update success rate
        if processing_result.get('success', False):
            current_success_rate = self.performance_stats['success_rate']
            total_docs = self.performance_stats['total_documents_processed']
            self.performance_stats['success_rate'] = (
                (current_success_rate * (total_docs - 1) + 1.0) / total_docs
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        total_docs = self.performance_stats['total_documents_processed']
        
        if total_docs == 0:
            return {'message': 'No documents processed yet'}
        
        return {
            'total_documents_processed': total_docs,
            'strategy_distribution': {
                'fast_track_simple': {
                    'count': self.performance_stats['fast_track_count'],
                    'percentage': (self.performance_stats['fast_track_count'] / total_docs) * 100
                },
                'hybrid_processing': {
                    'count': self.performance_stats['hybrid_count'],
                    'percentage': (self.performance_stats['hybrid_count'] / total_docs) * 100
                },
                'full_digital_twin': {
                    'count': self.performance_stats['digital_twin_count'],
                    'percentage': (self.performance_stats['digital_twin_count'] / total_docs) * 100
                }
            },
            'success_rate': self.performance_stats['success_rate'] * 100,
            'average_processing_time': self.performance_stats['average_processing_time']
        }


# Global instance for easy access
intelligent_document_router = IntelligentDocumentRouter() 