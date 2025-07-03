#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX Runtime 기반 얼굴 검출 엔진

새로운 모델 구조에 맞춰 ONNX Runtime을 사용한 얼굴 검출 엔진입니다.
- 1차 선택: RetinaFace MobileNet0.25
- 2차 선택: MobileFaceNet
- 백업: OpenCV Haar Cascade
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

from shared.vision_core.detection.base_detector import BaseDetector
from domains.face_recognition.core.value_objects.bounding_box import BoundingBox
from domains.face_recognition.core.value_objects.confidence_score import ConfidenceScore
from domains.face_recognition.infrastructure.models.retinaface_detector import RetinaFaceDetector
from domains.face_recognition.infrastructure.detection_engines.opencv_detection_engine import OpenCVDetectionEngine

logger = logging.getLogger(__name__)

class ONNXDetectionEngine(BaseDetector):
    """ONNX Runtime 기반 얼굴 검출 엔진"""
    
    def __init__(self, config: Dict):
        """
        ONNX 검출 엔진 초기화
        
        Args:
            config: 설정 딕셔너리 (models 섹션 포함)
        """
        super().__init__()
        
        self.config = config
        self.models = {}
        self.current_model = None
        self.fallback_order = []
        
        # 모델 초기화
        self._initialize_models()
        
        logger.info("ONNX Detection Engine initialized")
    
    def _initialize_models(self):
        """모델들을 우선순위별로 초기화"""
        try:
            models_config = self.config.get("models", {})
            detection_config = models_config.get("detection", {})
            
            # 1차 선택: RetinaFace MobileNet0.25
            if "primary" in detection_config:
                primary_config = detection_config["primary"]
                model_path = primary_config.get("path")
                
                if Path(model_path).exists():
                    self.models["primary"] = RetinaFaceDetector(
                        model_path=model_path,
                        device=primary_config.get("device", "auto")
                    )
                    self.fallback_order.append("primary")
                    logger.info("Primary model (RetinaFace) loaded successfully")
                else:
                    logger.warning(f"Primary model not found: {model_path}")
            
            # 2차 선택: MobileFaceNet
            if "secondary" in detection_config:
                secondary_config = detection_config["secondary"]
                model_path = secondary_config.get("path")
                
                if Path(model_path).exists():
                    # MobileFaceNet은 검출용이 아닌 인식용이므로 여기서는 건너뜀
                    logger.info("Secondary model (MobileFaceNet) - recognition only, skipping for detection")
                else:
                    logger.warning(f"Secondary model not found: {model_path}")
            
            # 백업: OpenCV Haar Cascade
            if "backup" in detection_config:
                backup_config = detection_config["backup"]
                model_path = backup_config.get("path")
                
                if Path(model_path).exists():
                    self.models["backup"] = OpenCVDetectionEngine(
                        cascade_path=model_path,
                        config=backup_config
                    )
                    self.fallback_order.append("backup")
                    logger.info("Backup model (OpenCV Haar Cascade) loaded successfully")
                else:
                    logger.warning(f"Backup model not found: {model_path}")
            
            # 현재 모델을 첫 번째 사용 가능한 모델로 설정
            if self.fallback_order:
                self.current_model = self.fallback_order[0]
                logger.info(f"Current model set to: {self.current_model}")
            else:
                logger.error("No models available for detection")
                
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise
    
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """
        이미지에서 얼굴 검출 (폴백 전략 적용)
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            검출된 얼굴들의 바운딩 박스 리스트
        """
        if not self.fallback_order:
            logger.error("No detection models available")
            return []
        
        # 폴백 순서대로 모델 시도
        for model_name in self.fallback_order:
            try:
                logger.debug(f"Trying detection with model: {model_name}")
                
                if model_name in self.models:
                    model = self.models[model_name]
                    bounding_boxes = model.detect(image)
                    
                    if bounding_boxes:
                        logger.info(f"Successfully detected {len(bounding_boxes)} faces using {model_name}")
                        self.current_model = model_name
                        return bounding_boxes
                    else:
                        logger.debug(f"No faces detected with {model_name}, trying next model")
                else:
                    logger.warning(f"Model {model_name} not available")
                    
            except Exception as e:
                logger.warning(f"Detection failed with {model_name}: {str(e)}")
                continue
        
        logger.warning("All detection models failed")
        return []
    
    def detect_with_model(self, image: np.ndarray, model_name: str) -> List[BoundingBox]:
        """
        특정 모델로 얼굴 검출
        
        Args:
            image: 입력 이미지 (BGR)
            model_name: 사용할 모델명 (primary, backup)
            
        Returns:
            검출된 얼굴들의 바운딩 박스 리스트
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not available")
            return []
        
        try:
            model = self.models[model_name]
            bounding_boxes = model.detect(image)
            
            logger.info(f"Detected {len(bounding_boxes)} faces using {model_name}")
            return bounding_boxes
            
        except Exception as e:
            logger.error(f"Detection failed with {model_name}: {str(e)}")
            return []
    
    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        return list(self.models.keys())
    
    def get_current_model(self) -> str:
        """현재 사용 중인 모델명 반환"""
        return self.current_model
    
    def set_current_model(self, model_name: str) -> bool:
        """
        현재 모델 설정
        
        Args:
            model_name: 설정할 모델명
            
        Returns:
            설정 성공 여부
        """
        if model_name in self.models:
            self.current_model = model_name
            logger.info(f"Current model set to: {model_name}")
            return True
        else:
            logger.error(f"Model {model_name} not available")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        모델 정보 반환
        
        Args:
            model_name: 모델명
            
        Returns:
            모델 정보 딕셔너리
        """
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        if hasattr(model, 'get_model_info'):
            return model.get_model_info()
        else:
            return {
                "name": model_name,
                "type": "detection",
                "available": True
            }
    
    def benchmark_models(self, test_image: np.ndarray) -> Dict[str, Dict]:
        """
        모든 모델의 성능 벤치마크
        
        Args:
            test_image: 테스트 이미지
            
        Returns:
            각 모델의 성능 정보
        """
        import time
        
        results = {}
        
        for model_name in self.models:
            try:
                model = self.models[model_name]
                
                # 워밍업
                for _ in range(3):
                    _ = model.detect(test_image)
                
                # 실제 측정
                start_time = time.time()
                bounding_boxes = model.detect(test_image)
                end_time = time.time()
                
                processing_time = (end_time - start_time) * 1000  # ms
                
                results[model_name] = {
                    "detection_count": len(bounding_boxes),
                    "processing_time_ms": processing_time,
                    "fps": 1000 / processing_time if processing_time > 0 else 0,
                    "success": True
                }
                
            except Exception as e:
                results[model_name] = {
                    "detection_count": 0,
                    "processing_time_ms": 0,
                    "fps": 0,
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def validate_image(self, image: np.ndarray) -> bool:
        """이미지 유효성 검사"""
        if image is None:
            return False
        
        if len(image.shape) != 3:
            return False
        
        if image.shape[2] != 3:
            return False
        
        if image.size == 0:
            return False
        
        return True
    
    def __del__(self):
        """소멸자 - 리소스 정리"""
        for model_name, model in self.models.items():
            if hasattr(model, '__del__'):
                try:
                    model.__del__()
                except:
                    pass 