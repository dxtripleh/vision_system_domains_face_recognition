#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Detector for Vision Core.

비전 시스템의 공통 검출기 인터페이스입니다.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
import time

from common.logging import get_logger

logger = get_logger(__name__)


class BaseDetector(ABC):
    """기본 검출기 인터페이스"""
    
    def __init__(self, 
                 model_path: str,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 input_size: Tuple[int, int] = (640, 640),
                 use_gpu: bool = True):
        """
        검출기 초기화
        
        Args:
            model_path: 모델 파일 경로
            confidence_threshold: 신뢰도 임계값
            nms_threshold: NMS 임계값
            input_size: 입력 이미지 크기 (width, height)
            use_gpu: GPU 사용 여부
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.use_gpu = use_gpu
        self.is_loaded = False
        self.model = None
        
        logger.info(f"BaseDetector initialized: {self.__class__.__name__}")
    
    @abstractmethod
    def load_model(self) -> bool:
        """모델 로딩"""
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """객체 검출"""
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        pass
    
    @abstractmethod
    def postprocess(self, outputs: Any, original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """검출 결과 후처리"""
        pass
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """배치 검출"""
        results = []
        for image in images:
            result = self.detect(image)
            results.append(result)
        return results
    
    def apply_nms(self, 
                  boxes: List[List[float]], 
                  scores: List[float], 
                  classes: List[int] = None) -> List[int]:
        """Non-Maximum Suppression 적용"""
        if not boxes or not scores:
            return []
        
        # OpenCV의 NMS 사용
        boxes_array = np.array(boxes, dtype=np.float32)
        scores_array = np.array(scores, dtype=np.float32)
        
        indices = cv2.dnn.NMSBoxes(
            boxes_array, 
            scores_array,
            self.confidence_threshold,
            self.nms_threshold
        )
        
        if len(indices) > 0:
            return indices.flatten().tolist()
        else:
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'input_size': self.input_size,
            'use_gpu': self.use_gpu,
            'is_loaded': self.is_loaded,
            'model_type': self.__class__.__name__
        }
    
    def validate_image(self, image: np.ndarray) -> bool:
        """입력 이미지 검증"""
        if image is None:
            return False
        
        if not isinstance(image, np.ndarray):
            return False
        
        if image.size == 0:
            return False
        
        if len(image.shape) not in [2, 3]:
            return False
        
        return True


class FaceDetector(BaseDetector):
    """얼굴 검출기 기본 클래스"""
    
    def __init__(self, 
                 model_path: str,
                 confidence_threshold: float = 0.6,
                 nms_threshold: float = 0.4,
                 input_size: Tuple[int, int] = (640, 640),
                 use_gpu: bool = True):
        super().__init__(model_path, confidence_threshold, nms_threshold, input_size, use_gpu)
        self.face_size_range = (20, 1000)  # 최소/최대 얼굴 크기
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """얼굴 검출 (detect 메서드의 별칭)"""
        return self.detect(image)
    
    def filter_faces_by_size(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """크기 기준으로 얼굴 필터링"""
        filtered_faces = []
        
        for face in faces:
            bbox = face.get('bbox', [])
            if len(bbox) >= 4:
                width = bbox[2] - bbox[0] if len(bbox) == 4 else bbox[2]
                height = bbox[3] - bbox[1] if len(bbox) == 4 else bbox[3]
                
                face_size = min(width, height)
                if self.face_size_range[0] <= face_size <= self.face_size_range[1]:
                    filtered_faces.append(face)
        
        return filtered_faces
    
    def get_largest_face(self, faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """가장 큰 얼굴 반환"""
        if not faces:
            return None
        
        largest_face = None
        largest_area = 0
        
        for face in faces:
            bbox = face.get('bbox', [])
            if len(bbox) >= 4:
                width = bbox[2] - bbox[0] if len(bbox) == 4 else bbox[2]
                height = bbox[3] - bbox[1] if len(bbox) == 4 else bbox[3]
                area = width * height
                
                if area > largest_area:
                    largest_area = area
                    largest_face = face
        
        return largest_face


class OpenCVFaceDetector(FaceDetector):
    """OpenCV Haar Cascade 기반 얼굴 검출기"""
    
    def __init__(self, 
                 cascade_path: str = None,
                 confidence_threshold: float = 0.7,
                 scale_factor: float = 1.1,
                 min_neighbors: int = 5):
        """
        OpenCV 얼굴 검출기 초기화
        
        Args:
            cascade_path: Haar cascade 파일 경로 (None이면 기본값 사용)
            confidence_threshold: 신뢰도 임계값
            scale_factor: 스케일 팩터
            min_neighbors: 최소 인접 객체 수
        """
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        super().__init__(
            model_path=cascade_path,
            confidence_threshold=confidence_threshold,
            input_size=(640, 640),
            use_gpu=False  # OpenCV Haar는 CPU만 지원
        )
        
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.cascade = None
    
    def load_model(self) -> bool:
        """Haar Cascade 모델 로딩"""
        try:
            self.cascade = cv2.CascadeClassifier(self.model_path)
            
            if self.cascade.empty():
                logger.error(f"Failed to load Haar cascade: {self.model_path}")
                return False
            
            self.is_loaded = True
            logger.info("OpenCV Haar cascade loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Haar cascade: {str(e)}")
            return False
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """얼굴 검출"""
        if not self.is_loaded:
            if not self.load_model():
                return []
        
        if not self.validate_image(image):
            return []
        
        try:
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 얼굴 검출
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # 결과 변환
            detected_faces = []
            for (x, y, w, h) in faces:
                face_data = {
                    'bbox': [x, y, x + w, y + h],  # [x1, y1, x2, y2] 형식
                    'confidence': self.confidence_threshold,  # Haar는 신뢰도를 제공하지 않으므로 기본값
                    'landmarks': [],
                    'area': w * h
                }
                detected_faces.append(face_data)
            
            return detected_faces
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 히스토그램 평활화
        gray = cv2.equalizeHist(gray)
        return gray
    
    def postprocess(self, outputs: Any, original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """후처리 (OpenCV는 이미 후처리가 완료된 결과를 반환)"""
        return outputs


def create_face_detector(detector_type: str = "opencv", **kwargs) -> FaceDetector:
    """
    얼굴 검출기 팩토리 함수
    
    Args:
        detector_type: 검출기 타입 ("opencv", "retinaface", "mtcnn" 등)
        **kwargs: 검출기별 설정
        
    Returns:
        FaceDetector: 생성된 얼굴 검출기
    """
    if detector_type.lower() == "opencv":
        return OpenCVFaceDetector(**kwargs)
    elif detector_type.lower() == "retinaface":
        # RetinaFace 구현체를 여기에 추가
        raise NotImplementedError("RetinaFace detector not implemented yet")
    elif detector_type.lower() == "mtcnn":
        # MTCNN 구현체를 여기에 추가
        raise NotImplementedError("MTCNN detector not implemented yet")
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}") 