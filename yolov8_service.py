"""
YOLOv8 Document Layout Analysis Microservice

This FastAPI microservice provides state-of-the-art visual content detection
using YOLOv8 fine-tuned on the PubLayNet dataset for document layout analysis.

Key Features:
- YOLOv8 model fine-tuned on PubLayNet dataset
- FastAPI web service for scalable deployment
- High-precision detection of figures, tables, text, titles, lists
- GPU acceleration support
- Comprehensive error handling and logging
"""

import os
import logging
import io
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import configparser

# FastAPI imports
try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# YOLOv8 and image processing imports
try:
    from ultralytics import YOLO
    import torch
    from PIL import Image
    import numpy as np
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YOLOv8DocumentAnalyzer:
    """
    YOLOv8-powered document layout analyzer fine-tuned on PubLayNet.
    
    This class provides supreme accuracy in detecting and classifying
    document layout elements including figures, tables, text blocks, etc.
    """
    
    def __init__(self, model_path: str = "yolov8m.pt"):
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.config = {
            'model_path': model_path,
            'confidence_threshold': 0.5,
            'iou_threshold': 0.4,
            'max_detections': 100,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'image_size': 640,  # YOLOv8 input size
            'publaynet_classes': {
                0: 'text',
                1: 'title', 
                2: 'list',
                3: 'table',
                4: 'figure',
                5: 'caption',
                6: 'quote',
                7: 'footnote',
                8: 'equation',
                9: 'marginalia',
                10: 'bibliography',
                11: 'header',
                12: 'footer'
            }
        }
        
        # Performance statistics
        self.stats = {
            'images_processed': 0,
            'detections_made': 0,
            'errors': 0,
            'average_inference_time': 0.0
        }
        
        # Load model
        self.model = self._load_model()
        
        self.logger.info(f"🚀 YOLOv8 Document Analyzer initialized:")
        self.logger.info(f"   📱 Device: {self.config['device']}")
        self.logger.info(f"   🎯 Model: {model_path}")
        self.logger.info(f"   🏷️ Classes: {list(self.config['publaynet_classes'].values())}")
    
    def _load_model(self) -> YOLO:
        """Load YOLOv8 model with error handling"""
        try:
            if not YOLO_AVAILABLE:
                raise Exception("YOLOv8 (ultralytics) not available")
            
            # Check if custom fine-tuned model exists
            if os.path.exists(self.config['model_path']):
                self.logger.info(f"📥 Loading custom model: {self.config['model_path']}")
                model = YOLO(self.config['model_path'])
            else:
                # Fallback to pre-trained model
                self.logger.warning(f"Custom model not found: {self.config['model_path']}")
                self.logger.info("📥 Loading pre-trained YOLOv8 model...")
                model = YOLO('yolov8m.pt')  # Medium model for balance of speed/accuracy
            
            # Move to appropriate device
            model.to(self.config['device'])
            
            self.logger.info(f"✅ Model loaded successfully on {self.config['device']}")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load YOLOv8 model: {e}")
            raise
    
    def detect_layout_elements(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect document layout elements in image using YOLOv8.
        
        Args:
            image: PIL Image of document page
            
        Returns:
            List of detected elements with labels, confidence, and bounding boxes
        """
        try:
            import time
            start_time = time.time()
            
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Run YOLOv8 inference
            results = self.model(
                image,
                conf=self.config['confidence_threshold'],
                iou=self.config['iou_threshold'],
                max_det=self.config['max_detections'],
                imgsz=self.config['image_size'],
                verbose=False
            )
            
            # Parse results
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract detection data
                        class_id = int(box.cls.cpu().numpy()[0])
                        confidence = float(box.conf.cpu().numpy()[0])
                        bbox = box.xyxy.cpu().numpy()[0].tolist()  # [x1, y1, x2, y2]
                        
                        # Map class ID to label
                        label = self.config['publaynet_classes'].get(class_id, 'unknown')
                        
                        detection = {
                            'label': label,
                            'confidence': confidence,
                            'bounding_box': bbox,
                            'class_id': class_id
                        }
                        
                        detections.append(detection)
            
            # Update statistics
            inference_time = time.time() - start_time
            self.stats['images_processed'] += 1
            self.stats['detections_made'] += len(detections)
            
            # Update average inference time
            total_time = self.stats['average_inference_time'] * (self.stats['images_processed'] - 1) + inference_time
            self.stats['average_inference_time'] = total_time / self.stats['images_processed']
            
            self.logger.debug(f"🎯 Detected {len(detections)} elements in {inference_time:.3f}s")
            
            return detections
            
        except Exception as e:
            self.logger.error(f"❌ Detection failed: {e}")
            self.stats['errors'] += 1
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        return {
            'model_path': self.config['model_path'],
            'device': self.config['device'],
            'classes': self.config['publaynet_classes'],
            'configuration': {
                'confidence_threshold': self.config['confidence_threshold'],
                'iou_threshold': self.config['iou_threshold'],
                'max_detections': self.config['max_detections'],
                'image_size': self.config['image_size']
            },
            'statistics': self.stats.copy(),
            'gpu_available': torch.cuda.is_available(),
            'gpu_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }

class YOLOv8Service:
    """
    Thin wrapper for YOLOv8DocumentAnalyzer for local (non-API) use, loading config from config.ini.
    """
    def __init__(self, config_path='config.ini'):
        config = configparser.ConfigParser()
        with open(config_path, encoding='utf-8') as f:
            config.read_file(f)
        yolo_cfg = config['YOLOv8']
        model_path = yolo_cfg.get('model_path', 'yolov8n.pt')
        self.conf_thres = float(yolo_cfg.get('confidence_threshold', 0.5))
        self.iou_thres = float(yolo_cfg.get('iou_threshold', 0.4))
        self.analyzer = YOLOv8DocumentAnalyzer(model_path)
        # Override thresholds if needed
        self.analyzer.config['confidence_threshold'] = self.conf_thres
        self.analyzer.config['iou_threshold'] = self.iou_thres
        # Log only the model actually being used
        logger = logging.getLogger(__name__)
        logger.info(f"✅ YOLOv8Service using model: {model_path}")

    def detect(self, image):
        """
        Detect layout elements in a PIL image. Returns list of detection dicts.
        """
        return self.analyzer.detect_layout_elements(image)

# Initialize the analyzer
try:
    # Try to load custom fine-tuned model first
    custom_model_path = "models/yolov8_publaynet_finetuned.pt"
    if os.path.exists(custom_model_path):
        analyzer = YOLOv8DocumentAnalyzer(custom_model_path)
        logger.info("🎯 Using fine-tuned PubLayNet model for supreme accuracy")
    else:
        analyzer = YOLOv8DocumentAnalyzer()
        logger.warning("⚠️ Using pre-trained model - consider fine-tuning on PubLayNet for best results")
        
except Exception as e:
    logger.error(f"❌ Failed to initialize YOLOv8 analyzer: {e}")
    analyzer = None

# Create FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="YOLOv8 Document Layout Analysis Service",
        description="State-of-the-art visual content detection for PDF documents",
        version="1.0.0"
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        if analyzer is None:
            raise HTTPException(status_code=503, detail="YOLOv8 analyzer not available")
        
        return {
            "status": "healthy",
            "service": "YOLOv8 Document Layout Analysis",
            "model_loaded": True,
            "device": analyzer.config['device']
        }
    
    @app.post("/predict/layout")
    async def predict_layout(file: UploadFile = File(...)):
        """
        Detect document layout elements in uploaded image.
        
        Returns detected figures, tables, text blocks, titles, and lists
        with bounding boxes and confidence scores.
        """
        if analyzer is None:
            raise HTTPException(status_code=503, detail="YOLOv8 analyzer not available")
        
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            # Read and process image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Detect layout elements
            detections = analyzer.detect_layout_elements(image)
            
            return {
                "detections": detections,
                "image_size": image.size,
                "total_detections": len(detections),
                "processing_info": {
                    "model": "YOLOv8",
                    "device": analyzer.config['device'],
                    "confidence_threshold": analyzer.config['confidence_threshold']
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    @app.get("/model/info")
    async def get_model_info():
        """Get model information and performance statistics"""
        if analyzer is None:
            raise HTTPException(status_code=503, detail="YOLOv8 analyzer not available")
        
        return analyzer.get_model_info()
    
    @app.get("/model/classes")
    async def get_supported_classes():
        """Get list of supported document layout classes"""
        if analyzer is None:
            raise HTTPException(status_code=503, detail="YOLOv8 analyzer not available")
        
        return {
            "classes": list(analyzer.config['publaynet_classes'].values()),
            "class_mapping": analyzer.config['publaynet_classes'],
            "description": "Document layout elements detectable by the model"
        }

else:
    logger.error("❌ FastAPI not available - cannot create web service")
    app = None

def run_service(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the YOLOv8 microservice"""
    if not FASTAPI_AVAILABLE:
        logger.error("❌ FastAPI not available - cannot run service")
        return
    
    if analyzer is None:
        logger.error("❌ YOLOv8 analyzer not initialized - cannot run service")
        return
    
    logger.info(f"🚀 Starting YOLOv8 Document Layout Analysis Service...")
    logger.info(f"   🌐 URL: http://{host}:{port}")
    logger.info(f"   📚 API docs: http://{host}:{port}/docs")
    logger.info(f"   🎯 Model: {analyzer.config['model_path']}")
    logger.info(f"   📱 Device: {analyzer.config['device']}")
    
    uvicorn.run(
        "yolov8_service:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    # Run the service
    run_service(host="127.0.0.1", port=8000, reload=False)
