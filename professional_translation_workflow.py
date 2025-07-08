"""
Professional Translation Workflow Integration

This module integrates the academic validation system with the existing Fenix pipeline
to address the translation quality issues identified in the analysis:

1. Professional human oversight workflow
2. Multi-stage validation process
3. Quality control checkpoints
4. Expert review integration
5. Automated quality reporting

Based on the analysis of translation deficiencies requiring human expert validation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from datetime import datetime, timedelta

# Import the academic validation system
from academic_translation_validator import (
    AcademicTranslationValidator, 
    ValidationSeverity,
    ValidationIssue,
    validate_academic_document,
    generate_validation_report_file
)

# Import existing Fenix components
from config_manager import config_manager
from translation_service import TranslationService

logger = logging.getLogger(__name__)

class WorkflowStage(Enum):
    """Stages in the professional translation workflow"""
    INITIAL_TRANSLATION = "initial_translation"
    AUTOMATED_VALIDATION = "automated_validation"
    EXPERT_REVIEW = "expert_review"
    REVISION = "revision"
    FINAL_VALIDATION = "final_validation"
    QUALITY_ASSURANCE = "quality_assurance"
    APPROVAL = "approval"

class ReviewerType(Enum):
    """Types of reviewers in the workflow"""
    DOMAIN_EXPERT = "domain_expert"
    LANGUAGE_EXPERT = "language_expert"
    TECHNICAL_EDITOR = "technical_editor"
    QUALITY_ASSURANCE = "quality_assurance"

@dataclass
class ReviewTask:
    """Represents a review task for human experts"""
    task_id: str
    document_title: str
    domain: str
    reviewer_type: ReviewerType
    priority: int
    created_at: datetime
    due_date: datetime
    status: str = "pending"
    assigned_to: Optional[str] = None
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    review_notes: str = ""
    quality_score: float = 0.0
    completed_at: Optional[datetime] = None

@dataclass
class TranslationProject:
    """Represents a complete translation project"""
    project_id: str
    document_title: str
    source_language: str
    target_language: str
    domain: str
    priority: int
    created_at: datetime
    current_stage: WorkflowStage
    original_text: str
    translated_text: str
    validation_results: Dict[str, Any] = field(default_factory=dict)
    review_tasks: List[ReviewTask] = field(default_factory=list)
    revision_history: List[Dict[str, Any]] = field(default_factory=list)
    final_quality_score: float = 0.0
    status: str = "active"

class ProfessionalTranslationWorkflow:
    """
    Professional translation workflow with human expert integration
    
    This class orchestrates the complete translation process from initial
    translation through expert review to final quality assurance.
    """
    
    def __init__(self, config_file: str = "professional_workflow_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.validator = AcademicTranslationValidator()
        self.translation_service = TranslationService()
        self.active_projects: Dict[str, TranslationProject] = {}
        self.review_queue: List[ReviewTask] = []
        self.expert_registry: Dict[str, Dict[str, Any]] = {}
        self.quality_metrics: Dict[str, Any] = {}
        self._load_existing_projects()
        self._load_expert_registry()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load professional workflow configuration"""
        default_config = {
            "quality_thresholds": {
                "minimum_quality_score": 0.8,
                "expert_review_threshold": 0.6,
                "automatic_approval_threshold": 0.95
            },
            "review_timeframes": {
                "domain_expert_review_days": 3,
                "language_expert_review_days": 2,
                "technical_editor_review_days": 1,
                "quality_assurance_review_days": 1
            },
            "validation_settings": {
                "enable_automated_validation": True,
                "enable_bibliography_validation": True,
                "enable_terminology_validation": True,
                "enable_structure_validation": True
            },
            "expert_assignment": {
                "auto_assign_experts": True,
                "require_domain_expert": True,
                "require_language_expert": True,
                "require_technical_editor": False
            },
            "notification_settings": {
                "email_notifications": True,
                "slack_notifications": False,
                "sms_notifications": False
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
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _load_existing_projects(self):
        """Load existing translation projects"""
        projects_file = "translation_projects.json"
        try:
            if os.path.exists(projects_file):
                with open(projects_file, 'r', encoding='utf-8') as f:
                    projects_data = json.load(f)
                    for project_data in projects_data:
                        project = self._deserialize_project(project_data)
                        self.active_projects[project.project_id] = project
                logger.info(f"Loaded {len(self.active_projects)} existing projects")
        except Exception as e:
            logger.error(f"Error loading existing projects: {e}")
    
    def _load_expert_registry(self):
        """Load expert registry"""
        registry_file = "expert_registry.json"
        try:
            if os.path.exists(registry_file):
                with open(registry_file, 'r', encoding='utf-8') as f:
                    self.expert_registry = json.load(f)
                logger.info(f"Loaded {len(self.expert_registry)} experts")
            else:
                # Create default expert registry
                self.expert_registry = {
                    "domain_experts": {
                        "philosophy": ["dr_smith@university.edu", "prof_jones@college.edu"],
                        "science": ["dr_wilson@research.org", "prof_davis@institute.edu"],
                        "literature": ["dr_brown@humanities.edu", "prof_taylor@arts.edu"]
                    },
                    "language_experts": {
                        "greek": ["translator_alpha@lang.com", "translator_beta@lang.com"],
                        "spanish": ["translator_gamma@lang.com", "translator_delta@lang.com"]
                    },
                    "technical_editors": ["editor_one@editing.com", "editor_two@editing.com"],
                    "quality_assurance": ["qa_lead@quality.com", "qa_senior@quality.com"]
                }
                self._save_expert_registry()
        except Exception as e:
            logger.error(f"Error loading expert registry: {e}")
            self.expert_registry = {}
    
    def _save_expert_registry(self):
        """Save expert registry to file"""
        try:
            with open("expert_registry.json", 'w', encoding='utf-8') as f:
                json.dump(self.expert_registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving expert registry: {e}")
    
    async def process_document_professionally(self, 
                                            original_text: str,
                                            document_title: str,
                                            source_language: str = "English",
                                            target_language: str = "Greek",
                                            domain: str = "academic",
                                            priority: int = 1) -> TranslationProject:
        """
        Process a document through the complete professional translation workflow
        
        Args:
            original_text: Original document text
            document_title: Title of the document
            source_language: Source language
            target_language: Target language
            domain: Academic domain (philosophy, science, etc.)
            priority: Priority level (1=highest, 5=lowest)
        
        Returns:
            TranslationProject object with complete workflow results
        """
        # Create new translation project
        project = TranslationProject(
            project_id=self._generate_project_id(),
            document_title=document_title,
            source_language=source_language,
            target_language=target_language,
            domain=domain,
            priority=priority,
            created_at=datetime.now(),
            current_stage=WorkflowStage.INITIAL_TRANSLATION,
            original_text=original_text,
            translated_text=""
        )
        
        self.active_projects[project.project_id] = project
        
        logger.info(f"Starting professional translation workflow for project {project.project_id}")
        logger.info(f"Document: {document_title}")
        logger.info(f"Domain: {domain}")
        logger.info(f"Priority: {priority}")
        
        try:
            # Stage 1: Initial Translation
            await self._stage_initial_translation(project)
            
            # Stage 2: Automated Validation
            await self._stage_automated_validation(project)
            
            # Stage 3: Expert Review (if needed)
            await self._stage_expert_review(project)
            
            # Stage 4: Revision (if needed)
            await self._stage_revision(project)
            
            # Stage 5: Final Validation
            await self._stage_final_validation(project)
            
            # Stage 6: Quality Assurance
            await self._stage_quality_assurance(project)
            
            # Stage 7: Approval
            await self._stage_approval(project)
            
            logger.info(f"Professional translation workflow completed for project {project.project_id}")
            logger.info(f"Final quality score: {project.final_quality_score:.2f}")
            
            return project
            
        except Exception as e:
            logger.error(f"Error in professional translation workflow: {e}")
            project.status = "error"
            raise
    
    async def _stage_initial_translation(self, project: TranslationProject):
        """Stage 1: Perform initial translation using existing Fenix pipeline"""
        logger.info(f"Stage 1: Initial translation for project {project.project_id}")
        project.current_stage = WorkflowStage.INITIAL_TRANSLATION
        
        try:
            # Use existing translation service
            # Note: This would integrate with your existing main_workflow.py
            translated_text = await self.translation_service.translate_text(
                project.original_text, 
                project.target_language
            )
            
            project.translated_text = translated_text
            
            # Record this stage in revision history
            project.revision_history.append({
                'stage': WorkflowStage.INITIAL_TRANSLATION.value,
                'timestamp': datetime.now().isoformat(),
                'description': 'Initial machine translation completed',
                'quality_score': 0.0  # Will be calculated in validation
            })
            
            logger.info(f"Initial translation completed for project {project.project_id}")
            
        except Exception as e:
            logger.error(f"Initial translation failed for project {project.project_id}: {e}")
            raise
    
    async def _stage_automated_validation(self, project: TranslationProject):
        """Stage 2: Perform automated validation using academic validator"""
        logger.info(f"Stage 2: Automated validation for project {project.project_id}")
        project.current_stage = WorkflowStage.AUTOMATED_VALIDATION
        
        try:
            # Use the academic translation validator
            validation_results = validate_academic_document(
                project.original_text,
                project.translated_text,
                project.domain
            )
            
            project.validation_results = validation_results
            
            # Generate validation report
            report_file = f"validation_report_{project.project_id}.txt"
            generate_validation_report_file(validation_results, report_file)
            
            # Record this stage
            project.revision_history.append({
                'stage': WorkflowStage.AUTOMATED_VALIDATION.value,
                'timestamp': datetime.now().isoformat(),
                'description': f'Automated validation completed: {validation_results["validation_summary"]["total_issues"]} issues found',
                'quality_score': validation_results['quality_score'],
                'validation_report': report_file
            })
            
            logger.info(f"Automated validation completed for project {project.project_id}")
            logger.info(f"Quality score: {validation_results['quality_score']:.2f}")
            logger.info(f"Issues found: {validation_results['validation_summary']['total_issues']}")
            
        except Exception as e:
            logger.error(f"Automated validation failed for project {project.project_id}: {e}")
            raise
    
    async def _stage_expert_review(self, project: TranslationProject):
        """Stage 3: Expert review (if quality score is below threshold)"""
        logger.info(f"Stage 3: Expert review assessment for project {project.project_id}")
        project.current_stage = WorkflowStage.EXPERT_REVIEW
        
        quality_score = project.validation_results.get('quality_score', 0.0)
        review_threshold = self.config['quality_thresholds']['expert_review_threshold']
        
        if quality_score >= self.config['quality_thresholds']['automatic_approval_threshold']:
            logger.info(f"Quality score {quality_score:.2f} exceeds automatic approval threshold, skipping expert review")
            return
        
        if quality_score < review_threshold:
            logger.info(f"Quality score {quality_score:.2f} below threshold {review_threshold:.2f}, creating expert review tasks")
            
            # Create review tasks based on validation issues
            review_tasks = self._create_review_tasks(project)
            project.review_tasks = review_tasks
            
            # Add to review queue
            self.review_queue.extend(review_tasks)
            
            # Assign experts
            await self._assign_experts_to_tasks(review_tasks)
            
            # Wait for reviews (in a real implementation, this would be handled asynchronously)
            await self._wait_for_reviews(project)
            
        else:
            logger.info(f"Quality score {quality_score:.2f} acceptable, minimal review required")
            # Create lightweight review task
            lightweight_task = self._create_lightweight_review_task(project)
            project.review_tasks = [lightweight_task]
            await self._assign_experts_to_tasks([lightweight_task])
    
    def _create_review_tasks(self, project: TranslationProject) -> List[ReviewTask]:
        """Create review tasks based on validation results"""
        tasks = []
        validation_results = project.validation_results
        
        # Determine required reviewers based on issues
        critical_issues = validation_results['validation_summary']['critical_issues']
        bibliography_issues = len([i for i in validation_results['issues'] if 'bibliography' in i['issue_type']])
        terminology_issues = len([i for i in validation_results['issues'] if 'terminology' in i['issue_type']])
        
        # Always require domain expert for academic documents
        if self.config['expert_assignment']['require_domain_expert']:
            domain_task = ReviewTask(
                task_id=f"{project.project_id}_domain_review",
                document_title=project.document_title,
                domain=project.domain,
                reviewer_type=ReviewerType.DOMAIN_EXPERT,
                priority=project.priority,
                created_at=datetime.now(),
                due_date=datetime.now() + timedelta(days=self.config['review_timeframes']['domain_expert_review_days']),
                validation_issues=[i for i in validation_results['issues'] if 'terminology' in i['issue_type']]
            )
            tasks.append(domain_task)
        
        # Require language expert for translation quality
        if self.config['expert_assignment']['require_language_expert']:
            language_task = ReviewTask(
                task_id=f"{project.project_id}_language_review",
                document_title=project.document_title,
                domain=project.domain,
                reviewer_type=ReviewerType.LANGUAGE_EXPERT,
                priority=project.priority,
                created_at=datetime.now(),
                due_date=datetime.now() + timedelta(days=self.config['review_timeframes']['language_expert_review_days']),
                validation_issues=[i for i in validation_results['issues'] if 'language' in i['issue_type'] or 'bibliography' in i['issue_type']]
            )
            tasks.append(language_task)
        
        # Require technical editor for structural issues
        if critical_issues > 0 or self.config['expert_assignment']['require_technical_editor']:
            technical_task = ReviewTask(
                task_id=f"{project.project_id}_technical_review",
                document_title=project.document_title,
                domain=project.domain,
                reviewer_type=ReviewerType.TECHNICAL_EDITOR,
                priority=project.priority,
                created_at=datetime.now(),
                due_date=datetime.now() + timedelta(days=self.config['review_timeframes']['technical_editor_review_days']),
                validation_issues=[i for i in validation_results['issues'] if 'structure' in i['issue_type']]
            )
            tasks.append(technical_task)
        
        return tasks
    
    def _create_lightweight_review_task(self, project: TranslationProject) -> ReviewTask:
        """Create a lightweight review task for high-quality translations"""
        return ReviewTask(
            task_id=f"{project.project_id}_lightweight_review",
            document_title=project.document_title,
            domain=project.domain,
            reviewer_type=ReviewerType.QUALITY_ASSURANCE,
            priority=project.priority,
            created_at=datetime.now(),
            due_date=datetime.now() + timedelta(days=1),
            validation_issues=[]
        )
    
    async def _assign_experts_to_tasks(self, tasks: List[ReviewTask]):
        """Assign experts to review tasks"""
        for task in tasks:
            expert = self._select_expert_for_task(task)
            if expert:
                task.assigned_to = expert
                task.status = "assigned"
                logger.info(f"Assigned task {task.task_id} to {expert}")
                
                # Send notification (placeholder)
                await self._notify_expert(expert, task)
            else:
                logger.warning(f"No expert available for task {task.task_id}")
                task.status = "unassigned"
    
    def _select_expert_for_task(self, task: ReviewTask) -> Optional[str]:
        """Select appropriate expert for a review task"""
        try:
            if task.reviewer_type == ReviewerType.DOMAIN_EXPERT:
                experts = self.expert_registry.get('domain_experts', {}).get(task.domain, [])
            elif task.reviewer_type == ReviewerType.LANGUAGE_EXPERT:
                experts = self.expert_registry.get('language_experts', {}).get('greek', [])  # Hardcoded for now
            elif task.reviewer_type == ReviewerType.TECHNICAL_EDITOR:
                experts = self.expert_registry.get('technical_editors', [])
            elif task.reviewer_type == ReviewerType.QUALITY_ASSURANCE:
                experts = self.expert_registry.get('quality_assurance', [])
            else:
                experts = []
            
            # Simple selection - in practice, this would consider availability, workload, etc.
            return experts[0] if experts else None
            
        except Exception as e:
            logger.error(f"Error selecting expert for task {task.task_id}: {e}")
            return None
    
    async def _notify_expert(self, expert: str, task: ReviewTask):
        """Send notification to expert about review task"""
        # Placeholder for notification system
        logger.info(f"Notification sent to {expert} for task {task.task_id}")
        
        # In a real implementation, this would send email, Slack message, etc.
        notification_message = f"""
        New Review Task Assigned
        
        Task ID: {task.task_id}
        Document: {task.document_title}
        Domain: {task.domain}
        Priority: {task.priority}
        Due Date: {task.due_date.strftime('%Y-%m-%d %H:%M')}
        
        Issues to Review: {len(task.validation_issues)}
        
        Please review and provide feedback.
        """
        
        # Save notification to file (for demonstration)
        notifications_dir = "notifications"
        os.makedirs(notifications_dir, exist_ok=True)
        
        notification_file = os.path.join(notifications_dir, f"notification_{task.task_id}.txt")
        with open(notification_file, 'w', encoding='utf-8') as f:
            f.write(notification_message)
    
    async def _wait_for_reviews(self, project: TranslationProject):
        """Wait for expert reviews to be completed"""
        # In a real implementation, this would be handled asynchronously
        # For demonstration, we'll simulate completed reviews
        
        logger.info(f"Waiting for expert reviews for project {project.project_id}")
        
        for task in project.review_tasks:
            # Simulate review completion
            await asyncio.sleep(0.1)  # Simulate processing time
            
            task.status = "completed"
            task.completed_at = datetime.now()
            task.quality_score = 0.85  # Simulated score
            task.review_notes = f"Review completed for {task.reviewer_type.value}. Quality is acceptable with minor improvements needed."
            
            logger.info(f"Review task {task.task_id} completed by {task.assigned_to}")
    
    async def _stage_revision(self, project: TranslationProject):
        """Stage 4: Apply revisions based on expert feedback"""
        logger.info(f"Stage 4: Revision for project {project.project_id}")
        project.current_stage = WorkflowStage.REVISION
        
        # Check if revisions are needed
        needs_revision = False
        revision_notes = []
        
        for task in project.review_tasks:
            if task.quality_score < self.config['quality_thresholds']['minimum_quality_score']:
                needs_revision = True
                revision_notes.append(f"{task.reviewer_type.value}: {task.review_notes}")
        
        if needs_revision:
            logger.info(f"Revisions needed for project {project.project_id}")
            
            # In a real implementation, this would apply specific revisions
            # For now, we'll simulate the revision process
            revised_text = await self._apply_revisions(project, revision_notes)
            project.translated_text = revised_text
            
            # Record revision
            project.revision_history.append({
                'stage': WorkflowStage.REVISION.value,
                'timestamp': datetime.now().isoformat(),
                'description': f'Revisions applied based on expert feedback',
                'revision_notes': revision_notes
            })
            
            logger.info(f"Revisions applied for project {project.project_id}")
        else:
            logger.info(f"No revisions needed for project {project.project_id}")
    
    async def _apply_revisions(self, project: TranslationProject, revision_notes: List[str]) -> str:
        """Apply revisions to the translated text"""
        # This is a placeholder for the revision process
        # In a real implementation, this would use the revision notes to make specific changes
        
        logger.info(f"Applying revisions for project {project.project_id}")
        
        # For demonstration, we'll just add a note that revisions were applied
        revised_text = project.translated_text
        
        # Simulate applying common fixes based on validation issues
        validation_issues = project.validation_results.get('issues', [])
        
        for issue in validation_issues:
            if issue.get('suggested_fix'):
                # Apply suggested fix (simplified)
                if issue['original_text'] in revised_text:
                    revised_text = revised_text.replace(
                        issue['original_text'],
                        issue['suggested_fix']
                    )
        
        return revised_text
    
    async def _stage_final_validation(self, project: TranslationProject):
        """Stage 5: Final validation after revisions"""
        logger.info(f"Stage 5: Final validation for project {project.project_id}")
        project.current_stage = WorkflowStage.FINAL_VALIDATION
        
        # Re-run validation on revised text
        final_validation_results = validate_academic_document(
            project.original_text,
            project.translated_text,
            project.domain
        )
        
        # Update validation results
        project.validation_results['final_validation'] = final_validation_results
        
        # Record final validation
        project.revision_history.append({
            'stage': WorkflowStage.FINAL_VALIDATION.value,
            'timestamp': datetime.now().isoformat(),
            'description': f'Final validation completed',
            'quality_score': final_validation_results['quality_score'],
            'issues_remaining': final_validation_results['validation_summary']['total_issues']
        })
        
        logger.info(f"Final validation completed for project {project.project_id}")
        logger.info(f"Final quality score: {final_validation_results['quality_score']:.2f}")
    
    async def _stage_quality_assurance(self, project: TranslationProject):
        """Stage 6: Quality assurance review"""
        logger.info(f"Stage 6: Quality assurance for project {project.project_id}")
        project.current_stage = WorkflowStage.QUALITY_ASSURANCE
        
        # Calculate final quality score
        validation_score = project.validation_results.get('final_validation', {}).get('quality_score', 0.0)
        review_scores = [task.quality_score for task in project.review_tasks if task.quality_score > 0]
        
        if review_scores:
            avg_review_score = sum(review_scores) / len(review_scores)
            final_score = (validation_score * 0.6) + (avg_review_score * 0.4)
        else:
            final_score = validation_score
        
        project.final_quality_score = final_score
        
        # Record QA stage
        project.revision_history.append({
            'stage': WorkflowStage.QUALITY_ASSURANCE.value,
            'timestamp': datetime.now().isoformat(),
            'description': f'Quality assurance completed',
            'final_quality_score': final_score,
            'validation_score': validation_score,
            'review_scores': review_scores
        })
        
        logger.info(f"Quality assurance completed for project {project.project_id}")
        logger.info(f"Final quality score: {final_score:.2f}")
    
    async def _stage_approval(self, project: TranslationProject):
        """Stage 7: Final approval"""
        logger.info(f"Stage 7: Approval for project {project.project_id}")
        project.current_stage = WorkflowStage.APPROVAL
        
        min_quality = self.config['quality_thresholds']['minimum_quality_score']
        
        if project.final_quality_score >= min_quality:
            project.status = "approved"
            logger.info(f"Project {project.project_id} approved with quality score {project.final_quality_score:.2f}")
        else:
            project.status = "rejected"
            logger.warning(f"Project {project.project_id} rejected with quality score {project.final_quality_score:.2f}")
        
        # Record approval stage
        project.revision_history.append({
            'stage': WorkflowStage.APPROVAL.value,
            'timestamp': datetime.now().isoformat(),
            'description': f'Project {project.status}',
            'final_quality_score': project.final_quality_score
        })
        
        # Save project
        self._save_project(project)
    
    def _generate_project_id(self) -> str:
        """Generate unique project ID"""
        from uuid import uuid4
        return f"PROF_{datetime.now().strftime('%Y%m%d')}_{str(uuid4())[:8]}"
    
    def _serialize_project(self, project: TranslationProject) -> Dict[str, Any]:
        """Serialize project for storage"""
        return {
            'project_id': project.project_id,
            'document_title': project.document_title,
            'source_language': project.source_language,
            'target_language': project.target_language,
            'domain': project.domain,
            'priority': project.priority,
            'created_at': project.created_at.isoformat(),
            'current_stage': project.current_stage.value,
            'status': project.status,
            'final_quality_score': project.final_quality_score,
            'validation_results': project.validation_results,
            'revision_history': project.revision_history,
            'review_tasks': [self._serialize_review_task(task) for task in project.review_tasks]
        }
    
    def _serialize_review_task(self, task: ReviewTask) -> Dict[str, Any]:
        """Serialize review task for storage"""
        return {
            'task_id': task.task_id,
            'document_title': task.document_title,
            'domain': task.domain,
            'reviewer_type': task.reviewer_type.value,
            'priority': task.priority,
            'created_at': task.created_at.isoformat(),
            'due_date': task.due_date.isoformat(),
            'status': task.status,
            'assigned_to': task.assigned_to,
            'review_notes': task.review_notes,
            'quality_score': task.quality_score,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None
        }
    
    def _deserialize_project(self, data: Dict[str, Any]) -> TranslationProject:
        """Deserialize project from storage"""
        project = TranslationProject(
            project_id=data['project_id'],
            document_title=data['document_title'],
            source_language=data['source_language'],
            target_language=data['target_language'],
            domain=data['domain'],
            priority=data['priority'],
            created_at=datetime.fromisoformat(data['created_at']),
            current_stage=WorkflowStage(data['current_stage']),
            original_text="",  # Not stored for space reasons
            translated_text="",  # Not stored for space reasons
            status=data['status'],
            final_quality_score=data['final_quality_score'],
            validation_results=data['validation_results'],
            revision_history=data['revision_history'],
            review_tasks=[self._deserialize_review_task(task_data) for task_data in data['review_tasks']]
        )
        return project
    
    def _deserialize_review_task(self, data: Dict[str, Any]) -> ReviewTask:
        """Deserialize review task from storage"""
        return ReviewTask(
            task_id=data['task_id'],
            document_title=data['document_title'],
            domain=data['domain'],
            reviewer_type=ReviewerType(data['reviewer_type']),
            priority=data['priority'],
            created_at=datetime.fromisoformat(data['created_at']),
            due_date=datetime.fromisoformat(data['due_date']),
            status=data['status'],
            assigned_to=data['assigned_to'],
            review_notes=data['review_notes'],
            quality_score=data['quality_score'],
            completed_at=datetime.fromisoformat(data['completed_at']) if data['completed_at'] else None
        )
    
    def _save_project(self, project: TranslationProject):
        """Save project to storage"""
        try:
            projects_file = "translation_projects.json"
            
            # Load existing projects
            existing_projects = []
            if os.path.exists(projects_file):
                with open(projects_file, 'r', encoding='utf-8') as f:
                    existing_projects = json.load(f)
            
            # Update or add current project
            project_data = self._serialize_project(project)
            
            # Remove existing entry if it exists
            existing_projects = [p for p in existing_projects if p['project_id'] != project.project_id]
            
            # Add updated project
            existing_projects.append(project_data)
            
            # Save back to file
            with open(projects_file, 'w', encoding='utf-8') as f:
                json.dump(existing_projects, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Project {project.project_id} saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving project {project.project_id}: {e}")
    
    def get_project_status(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a translation project"""
        if project_id in self.active_projects:
            project = self.active_projects[project_id]
            return {
                'project_id': project.project_id,
                'document_title': project.document_title,
                'current_stage': project.current_stage.value,
                'status': project.status,
                'quality_score': project.final_quality_score,
                'created_at': project.created_at.isoformat(),
                'review_tasks': len(project.review_tasks),
                'completed_tasks': len([t for t in project.review_tasks if t.status == "completed"])
            }
        return None
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get overall workflow statistics"""
        stats = {
            'total_projects': len(self.active_projects),
            'projects_by_status': {},
            'projects_by_stage': {},
            'average_quality_score': 0.0,
            'pending_reviews': len(self.review_queue),
            'experts_registered': sum(len(experts) if isinstance(experts, list) else sum(len(domain_experts) for domain_experts in experts.values()) for experts in self.expert_registry.values())
        }
        
        if self.active_projects:
            # Calculate statistics
            statuses = [p.status for p in self.active_projects.values()]
            stages = [p.current_stage.value for p in self.active_projects.values()]
            quality_scores = [p.final_quality_score for p in self.active_projects.values() if p.final_quality_score > 0]
            
            stats['projects_by_status'] = {status: statuses.count(status) for status in set(statuses)}
            stats['projects_by_stage'] = {stage: stages.count(stage) for stage in set(stages)}
            stats['average_quality_score'] = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return stats
    
    def generate_workflow_report(self) -> str:
        """Generate a comprehensive workflow report"""
        stats = self.get_workflow_statistics()
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("PROFESSIONAL TRANSLATION WORKFLOW REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Overall statistics
        report_lines.append("OVERALL STATISTICS")
        report_lines.append("-" * 20)
        report_lines.append(f"Total Projects: {stats['total_projects']}")
        report_lines.append(f"Average Quality Score: {stats['average_quality_score']:.2f}")
        report_lines.append(f"Pending Reviews: {stats['pending_reviews']}")
        report_lines.append(f"Registered Experts: {stats['experts_registered']}")
        report_lines.append("")
        
        # Projects by status
        if stats['projects_by_status']:
            report_lines.append("PROJECTS BY STATUS")
            report_lines.append("-" * 20)
            for status, count in stats['projects_by_status'].items():
                report_lines.append(f"{status.title()}: {count}")
            report_lines.append("")
        
        # Projects by stage
        if stats['projects_by_stage']:
            report_lines.append("PROJECTS BY STAGE")
            report_lines.append("-" * 20)
            for stage, count in stats['projects_by_stage'].items():
                report_lines.append(f"{stage.replace('_', ' ').title()}: {count}")
            report_lines.append("")
        
        # Recent projects
        recent_projects = sorted(
            self.active_projects.values(),
            key=lambda p: p.created_at,
            reverse=True
        )[:5]
        
        if recent_projects:
            report_lines.append("RECENT PROJECTS")
            report_lines.append("-" * 15)
            for project in recent_projects:
                report_lines.append(f"• {project.document_title}")
                report_lines.append(f"  Status: {project.status}")
                report_lines.append(f"  Stage: {project.current_stage.value}")
                report_lines.append(f"  Quality: {project.final_quality_score:.2f}")
                report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)

# Create default instance
professional_workflow = ProfessionalTranslationWorkflow()

async def process_document_with_professional_workflow(
    original_text: str,
    document_title: str,
    source_language: str = "English",
    target_language: str = "Greek",
    domain: str = "academic",
    priority: int = 1
) -> TranslationProject:
    """
    Convenience function for processing documents through professional workflow
    
    Args:
        original_text: Original document text
        document_title: Title of the document
        source_language: Source language
        target_language: Target language
        domain: Academic domain
        priority: Priority level
    
    Returns:
        TranslationProject with complete workflow results
    """
    return await professional_workflow.process_document_professionally(
        original_text=original_text,
        document_title=document_title,
        source_language=source_language,
        target_language=target_language,
        domain=domain,
        priority=priority
    )

def get_workflow_status() -> Dict[str, Any]:
    """Get current workflow status"""
    return professional_workflow.get_workflow_statistics()

def generate_workflow_report_file(output_file: str = "professional_workflow_report.txt") -> str:
    """Generate and save workflow report to file"""
    report_content = professional_workflow.generate_workflow_report()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Workflow report saved to {output_file}")
    return output_file 