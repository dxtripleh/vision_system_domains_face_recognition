#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RetinaFace Detection Engine.

RetinaFace 기반 얼굴 검출 엔진입니다.
"""

import numpy as np
from typing import List, Dict, Any, Optional

from common.logging import get_logger
from ..models.retinaface_detector import RetinaFaceDetector

logger = get_logger(__name__)


class RetinaFaceDetectionEngine:
    """RetinaFace 기반 검출 엔진"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        엔진 초기화
        
        Args:
            config: 검출 엔진 설정
        """
        self.config = config
        self.model_path = config.get('model_path', 'models/weights/face_detection_retinaface_mnet025_20250628.onnx')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.4)
        self.input_size = config.get('input_size', (640, 640))
        self.use_gpu = config.get('use_gpu', True)
        
        # RetinaFace 검출기 초기화
        self.detector = RetinaFaceDetector(
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            input_size=self.input_size,
            use_gpu=self.use_gpu
        )
        
        # 모델 로드
        self.is_loaded = self.detector.load_model()
        if not self.is_loaded:
            logger.warning("RetinaFace model loading failed, engine may not work properly")
        
        logger.info("RetinaFace detection engine initialized")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        얼굴 검출 수행
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            List[Dict[str, Any]]: 검출 결과
        """
        if not self.is_loaded:
            logger.error("RetinaFace model not loaded")
            return []
        
        if image is None or image.size == 0:
            return []
        
        try:
            # RetinaFace 검출 수행
            detections = self.detector.detect(image)
            
            # 결과 형식 표준화
            standardized_detections = []
            for detection in detections:
                standardized_detection = {
                    'bbox': detection.get('bbox', [0, 0, 0, 0]),
                    'confidence': detection.get('confidence', 0.0),
                    'landmarks': detection.get('landmarks'),
                    'quality_score': self._calculate_quality_score(detection, image.shape)
                }
                standardized_detections.append(standardized_detection)
            
            logger.debug(f"RetinaFace detected {len(standardized_detections)} faces")
            return standardized_detections
            
        except Exception as e:
            logger.error(f"RetinaFace detection error: {str(e)}")
            return []
    
    def _calculate_quality_score(self, detection: Dict[str, Any], image_shape: tuple) -> float:
        """얼굴 품질 점수 계산"""
        try:
            confidence = detection.get('confidence', 0.0)
            bbox = detection.get('bbox', [0, 0, 0, 0])
            landmarks = detection.get('landmarks')
            
            # 기본 신뢰도 점수
            confidence_score = confidence
            
            # 크기 점수
            x, y, w, h = bbox
            face_area = w * h
            image_area = image_shape[0] * image_shape[1]
            size_ratio = face_area / image_area if image_area > 0 else 0
            size_score = min(size_ratio * 10, 1.0)  # 10% 크기일 때 만점
            
            # 랜드마크 품질 점수 (있는 경우)
            landmark_score = 1.0
            if landmarks and len(landmarks) >= 10:  # 5개 랜드마크 * 2 (x, y)
                # 랜드마크가 바운딩 박스 내에 있는지 확인
                valid_landmarks = 0
                total_landmarks = len(landmarks) // 2
                
                for i in range(0, len(landmarks), 2):
                    lx, ly = landmarks[i], landmarks[i+1]
                    if x <= lx <= x + w and y <= ly <= y + h:
                        valid_landmarks += 1
                
                landmark_score = valid_landmarks / total_landmarks if total_landmarks > 0 else 0.5
            
            # 전체 품질 점수 (가중 평균)
            quality_score = (
                confidence_score * 0.5 + 
                size_score * 0.3 + 
                landmark_score * 0.2
            )
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Quality score calculation error: {str(e)}")
            return 0.5  # 기본값
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 설정"""
        self.confidence_threshold = threshold
        if self.detector:
            self.detector.set_confidence_threshold(threshold)
    
    def get_engine_info(self) -> Dict[str, Any]:
        """엔진 정보 반환"""
        return {
            'engine_type': 'retinaface',
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'input_size': self.input_size,
            'use_gpu': self.use_gpu,
            'is_loaded': self.is_loaded
        } 