#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenCV Detection Engine.

OpenCV 기반 얼굴 검출 엔진입니다.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from common.logging import get_logger

logger = get_logger(__name__)


class OpenCVDetectionEngine:
    """OpenCV Haar Cascade 기반 검출 엔진"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        엔진 초기화
        
        Args:
            config: 검출 엔진 설정
        """
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        self.scale_factor = config.get('scale_factor', 1.05)  # 더 민감하게
        self.min_neighbors = config.get('min_neighbors', 2)   # 더 민감하게
        self.min_size = config.get('min_size', (20, 20))      # 더 작은 얼굴도 검출
        self.max_size = config.get('max_size', ())
        
        # Cascade 파일 로드
        cascade_path = config.get('cascade_path')
        if cascade_path and Path(cascade_path).exists():
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            # 기본 cascade 파일 사용
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load OpenCV face cascade")
        
        logger.info("OpenCV detection engine initialized")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        얼굴 검출 수행
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            List[Dict[str, Any]]: 검출 결과
        """
        if image is None or image.size == 0:
            return []
        
        try:
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 히스토그램 평활화 제거 (디버깅 결과 평활화가 검출을 방해함)
            # gray = cv2.equalizeHist(gray)
            
            # 얼굴 검출 (maxSize 파라미터 제거)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # 결과 형식 변환
            detections = []
            for (x, y, w, h) in faces:
                detection = {
                    'bbox': [x, y, w, h],
                    'confidence': 0.8,  # OpenCV는 신뢰도를 제공하지 않으므로 고정값
                    'landmarks': None,
                    'quality_score': self._calculate_quality_score(x, y, w, h, image.shape)
                }
                detections.append(detection)
            
            logger.debug(f"OpenCV detected {len(detections)} faces")
            return detections
            
        except Exception as e:
            logger.error(f"OpenCV detection error: {str(e)}")
            return []
    
    def _calculate_quality_score(self, x: int, y: int, w: int, h: int, 
                                image_shape: tuple) -> float:
        """얼굴 품질 점수 계산"""
        # 이미지 내 위치 점수 (중앙에 가까울수록 높음)
        img_h, img_w = image_shape[:2]
        center_x, center_y = x + w/2, y + h/2
        distance_from_center = np.sqrt(
            ((center_x - img_w/2) / img_w)**2 + 
            ((center_y - img_h/2) / img_h)**2
        )
        position_score = 1.0 - min(distance_from_center, 1.0)
        
        # 크기 점수 (적절한 크기일수록 높음)
        face_area = w * h
        image_area = img_w * img_h
        size_ratio = face_area / image_area
        size_score = min(size_ratio * 10, 1.0)  # 10% 크기일 때 만점
        
        # 종횡비 점수 (1:1에 가까울수록 높음)
        aspect_ratio = w / h if h > 0 else 0
        aspect_score = 1.0 - abs(aspect_ratio - 1.0)
        aspect_score = max(0.0, aspect_score)
        
        # 전체 품질 점수
        quality_score = (position_score * 0.4 + size_score * 0.4 + aspect_score * 0.2)
        return min(max(quality_score, 0.0), 1.0)
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 설정"""
        self.confidence_threshold = threshold
    
    def get_engine_info(self) -> Dict[str, Any]:
        """엔진 정보 반환"""
        return {
            'engine_type': 'opencv',
            'confidence_threshold': self.confidence_threshold,
            'scale_factor': self.scale_factor,
            'min_neighbors': self.min_neighbors,
            'min_size': self.min_size,
            'max_size': self.max_size
        } 