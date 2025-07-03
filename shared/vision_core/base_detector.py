#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
기본 검출기 인터페이스.

이 모듈은 모든 검출기가 구현해야 하는 기본 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class BaseDetector(ABC):
    """모든 검출기의 기본 인터페이스"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """검출기 초기화
        
        Args:
            model_path: 모델 파일 경로
            confidence_threshold: 검출 신뢰도 임계값
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> bool:
        """모델 로딩
        
        Returns:
            로딩 성공 여부
        """
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """이미지에서 객체 검출
        
        Args:
            image: 입력 이미지 (BGR 형식)
            
        Returns:
            검출 결과 리스트 [
                {
                    'bbox': [x, y, w, h],
                    'confidence': float,
                    'landmarks': [[x1, y1], [x2, y2], ...] (선택적)
                }
            ]
        """
        pass
    
    @abstractmethod
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """배치 이미지에서 객체 검출
        
        Args:
            images: 입력 이미지 리스트
            
        Returns:
            각 이미지별 검출 결과 리스트
        """
        pass
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 설정
        
        Args:
            threshold: 새로운 임계값 (0.0 ~ 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        self.confidence_threshold = threshold
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환
        
        Returns:
            모델 정보 딕셔너리
        """
        return {
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'is_loaded': self.is_loaded,
            'model_type': self.__class__.__name__
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리 (기본 구현)
        
        Args:
            image: 입력 이미지
            
        Returns:
            전처리된 이미지
        """
        # 기본적으로 원본 이미지 반환
        # 서브클래스에서 필요에 따라 오버라이드
        return image
    
    def postprocess_detections(self, raw_detections: Any) -> List[Dict[str, Any]]:
        """검출 결과 후처리 (기본 구현)
        
        Args:
            raw_detections: 모델의 원시 검출 결과
            
        Returns:
            표준화된 검출 결과
        """
        # 서브클래스에서 구현 필요
        raise NotImplementedError("Subclass must implement postprocess_detections")
    
    def validate_image(self, image: np.ndarray) -> bool:
        """입력 이미지 유효성 검사
        
        Args:
            image: 검사할 이미지
            
        Returns:
            이미지 유효성 여부
        """
        if image is None:
            return False
        
        if not isinstance(image, np.ndarray):
            return False
        
        if len(image.shape) != 3:
            return False
        
        if image.shape[2] != 3:
            return False
        
        return True
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        if not self.is_loaded:
            self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        # 필요시 리소스 정리
        pass 