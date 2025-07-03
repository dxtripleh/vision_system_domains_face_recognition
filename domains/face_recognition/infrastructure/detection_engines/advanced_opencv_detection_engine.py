#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced OpenCV Detection Engine.

여러 전처리 방법과 검출 파라미터를 조합한 고급 얼굴 검출 엔진입니다.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from common.logging import get_logger

logger = get_logger(__name__)


class AdvancedOpenCVDetectionEngine:
    """고급 OpenCV 얼굴 검출 엔진"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        엔진 초기화
        
        Args:
            config: 검출 엔진 설정
        """
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        
        # 다중 검출 설정
        self.detection_configs = [
            {'scale': 1.02, 'neighbors': 1, 'min_size': (15, 15), 'name': '극도 민감'},
            {'scale': 1.05, 'neighbors': 1, 'min_size': (20, 20), 'name': '매우 민감'},
            {'scale': 1.05, 'neighbors': 2, 'min_size': (20, 20), 'name': '민감'},
        ]
        
        # Cascade 로드
        cascade_path = 'models/weights/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load OpenCV face cascade")
        
        logger.info("Advanced OpenCV detection engine initialized")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """고급 얼굴 검출 수행"""
        if image is None or image.size == 0:
            return []
        
        try:
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 다중 전처리 및 검출
            all_detections = []
            
            # 1. 원본 이미지로 검출
            for config in self.detection_configs:
                detections = self._detect_with_config(gray, config, 'original')
                all_detections.extend(detections)
            
            # 2. Median Blur 적용
            median_blur = cv2.medianBlur(gray, 3)
            for config in self.detection_configs:
                detections = self._detect_with_config(median_blur, config, 'median_blur')
                all_detections.extend(detections)
            
            # 3. CLAHE 적용
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(gray)
            for config in self.detection_configs:
                detections = self._detect_with_config(clahe_img, config, 'clahe')
                all_detections.extend(detections)
            
            # 4. 이미지 확대 (1.25x)
            h, w = gray.shape
            resized = cv2.resize(gray, (int(w*1.25), int(h*1.25)))
            for config in self.detection_configs:
                detections = self._detect_with_config(resized, config, 'resized_1.25x')
                # 좌표를 원본 크기로 변환
                for det in detections:
                    det['bbox'] = [int(x/1.25) for x in det['bbox']]
                all_detections.extend(detections)
            
            # 중복 제거 및 최적화
            final_detections = self._merge_detections(all_detections)
            
            logger.debug(f"Advanced OpenCV detected {len(final_detections)} faces")
            return final_detections
            
        except Exception as e:
            logger.error(f"Advanced OpenCV detection error: {str(e)}")
            return []
    
    def _detect_with_config(self, gray: np.ndarray, config: Dict[str, Any], method: str) -> List[Dict[str, Any]]:
        """특정 설정으로 검출"""
        try:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=config['scale'],
                minNeighbors=config['neighbors'],
                minSize=config['min_size'],
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            detections = []
            for (x, y, w, h) in faces:
                detection = {
                    'bbox': [x, y, w, h],
                    'confidence': self._calculate_confidence(config, method),
                    'landmarks': None,
                    'quality_score': self._calculate_quality_score(x, y, w, h, gray.shape),
                    'method': f"{method}_{config['name']}"
                }
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.warning(f"Detection failed with {method}_{config['name']}: {e}")
            return []
    
    def _calculate_confidence(self, config: Dict[str, Any], method: str) -> float:
        """신뢰도 계산"""
        base_confidence = 0.8
        
        if config['name'] == '극도 민감':
            base_confidence *= 0.6
        elif config['name'] == '매우 민감':
            base_confidence *= 0.8
        elif config['name'] == '민감':
            base_confidence *= 0.9
        
        if method == 'median_blur':
            base_confidence *= 0.9
        elif method == 'clahe':
            base_confidence *= 0.95
        elif method == 'resized_1.25x':
            base_confidence *= 0.85
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _calculate_quality_score(self, x: int, y: int, w: int, h: int, image_shape: tuple) -> float:
        """얼굴 품질 점수 계산"""
        img_h, img_w = image_shape[:2]
        center_x, center_y = x + w/2, y + h/2
        distance_from_center = np.sqrt(
            ((center_x - img_w/2) / img_w)**2 + 
            ((center_y - img_h/2) / img_h)**2
        )
        position_score = 1.0 - min(distance_from_center, 1.0)
        
        face_area = w * h
        image_area = img_w * img_h
        size_ratio = face_area / image_area
        size_score = min(size_ratio * 10, 1.0)
        
        aspect_ratio = w / h if h > 0 else 0
        aspect_score = 1.0 - abs(aspect_ratio - 1.0)
        aspect_score = max(0.0, aspect_score)
        
        quality_score = (position_score * 0.4 + size_score * 0.4 + aspect_score * 0.2)
        return min(max(quality_score, 0.0), 1.0)
    
    def _merge_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 검출 제거 및 최적화"""
        if not detections:
            return []
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        for detection in detections:
            if detection['confidence'] < self.confidence_threshold:
                continue
            
            is_duplicate = False
            for existing in merged:
                if self._calculate_iou(detection['bbox'], existing['bbox']) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(detection)
        
        return merged
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """IoU 계산"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 설정"""
        self.confidence_threshold = threshold
    
    def get_engine_info(self) -> Dict[str, Any]:
        """엔진 정보 반환"""
        return {
            'engine_type': 'advanced_opencv',
            'confidence_threshold': self.confidence_threshold,
            'detection_configs': [c['name'] for c in self.detection_configs]
        } 