#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Detector Interface.

얼굴 검출을 위한 공통 인터페이스와 유틸리티를 제공합니다.
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from common.logging import get_logger

logger = get_logger(__name__)


class FaceDetectorInterface(ABC):
    """얼굴 검출기 인터페이스"""
    
    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """얼굴 검출 수행"""
        pass
    
    @abstractmethod
    def set_confidence_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 설정"""
        pass


class FaceDetector:
    """통합 얼굴 검출기"""
    
    def __init__(self, detector_type: str = "opencv", **kwargs):
        """
        검출기 초기화
        
        Args:
            detector_type: 검출기 타입 ('opencv', 'retinaface', 'mtcnn')
            **kwargs: 검출기별 추가 설정
        """
        self.detector_type = detector_type
        self.detector = None
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        
        self._initialize_detector(kwargs)
        
    def _initialize_detector(self, config: Dict[str, Any]) -> None:
        """검출기 초기화"""
        if self.detector_type == "opencv":
            self.detector = OpenCVFaceDetector(config)
        elif self.detector_type == "retinaface":
            self.detector = RetinaFaceDetectorWrapper(config)
        elif self.detector_type == "mtcnn":
            self.detector = MTCNNDetectorWrapper(config)
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")
        
        logger.info(f"Face detector initialized: {self.detector_type}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        얼굴 검출 수행
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            List[Dict[str, Any]]: 검출된 얼굴 정보
        """
        return self.detector.detect_faces(image)
    
    def detect_largest_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        가장 큰 얼굴 하나만 검출
        
        Args:
            image: 입력 이미지
            
        Returns:
            Optional[Dict[str, Any]]: 가장 큰 얼굴 정보
        """
        faces = self.detect_faces(image)
        
        if not faces:
            return None
        
        # 면적 기준으로 가장 큰 얼굴 선택
        largest_face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
        return largest_face
    
    def extract_face_region(self, image: np.ndarray, bbox: List[int], 
                          margin: float = 0.2) -> np.ndarray:
        """
        얼굴 영역 추출
        
        Args:
            image: 원본 이미지
            bbox: 바운딩 박스 [x, y, w, h]
            margin: 확장 마진 비율
            
        Returns:
            np.ndarray: 추출된 얼굴 이미지
        """
        x, y, w, h = bbox
        
        # 마진 적용
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        # 확장된 영역 계산
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        # 얼굴 영역 추출
        face_region = image[y1:y2, x1:x2]
        
        return face_region
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 설정"""
        self.confidence_threshold = threshold
        self.detector.set_confidence_threshold(threshold)


class OpenCVFaceDetector(FaceDetectorInterface):
    """OpenCV Haar Cascade 기반 얼굴 검출기"""
    
    def __init__(self, config: Dict[str, Any]):
        """초기화"""
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.scale_factor = config.get('scale_factor', 1.1)
        self.min_neighbors = config.get('min_neighbors', 5)
        self.min_size = config.get('min_size', (30, 30))
        
        # Haar Cascade 분류기 로드
        cascade_path = config.get('cascade_path', 
                                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load cascade classifier: {cascade_path}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """얼굴 검출 수행"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 검출
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        # 결과 포맷 변환
        detections = []
        for (x, y, w, h) in faces:
            detection = {
                'bbox': [x, y, w, h],
                'confidence': 1.0,  # OpenCV는 신뢰도를 제공하지 않음
                'landmarks': None
            }
            detections.append(detection)
        
        return detections
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 설정 (OpenCV는 지원하지 않음)"""
        self.confidence_threshold = threshold


class RetinaFaceDetectorWrapper(FaceDetectorInterface):
    """RetinaFace 검출기 래퍼"""
    
    def __init__(self, config: Dict[str, Any]):
        """초기화"""
        # RetinaFace 검출기 import 및 초기화
        try:
            from ...infrastructure.models.retinaface_detector import RetinaFaceDetector
            
            model_path = config.get('model_path', 'models/weights/face_detection_retinaface_mnet025_20250628.onnx')
            self.detector = RetinaFaceDetector(
                model_path=model_path,
                confidence_threshold=config.get('confidence_threshold', 0.5),
                nms_threshold=config.get('nms_threshold', 0.4),
                use_gpu=config.get('use_gpu', True)
            )
            
            if not self.detector.load_model():
                raise RuntimeError("Failed to load RetinaFace model")
                
        except ImportError as e:
            raise ImportError(f"RetinaFace detector not available: {e}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """얼굴 검출 수행"""
        return self.detector.detect(image)
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 설정"""
        self.detector.set_confidence_threshold(threshold)


class MTCNNDetectorWrapper(FaceDetectorInterface):
    """MTCNN 검출기 래퍼 (향후 구현)"""
    
    def __init__(self, config: Dict[str, Any]):
        """초기화"""
        raise NotImplementedError("MTCNN detector not implemented yet")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """얼굴 검출 수행"""
        raise NotImplementedError("MTCNN detector not implemented yet")
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 설정"""
        raise NotImplementedError("MTCNN detector not implemented yet")


def create_face_detector(detector_type: str = "opencv", **kwargs) -> FaceDetector:
    """
    얼굴 검출기 팩토리 함수
    
    Args:
        detector_type: 검출기 타입
        **kwargs: 검출기별 설정
        
    Returns:
        FaceDetector: 생성된 검출기
    """
    return FaceDetector(detector_type=detector_type, **kwargs) 