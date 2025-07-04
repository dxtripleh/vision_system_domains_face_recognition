#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
비전 알고리즘 공통 모듈.

이 모듈은 비전 시스템의 공통 알고리즘들을 제공합니다.
크로스 플랫폼 호환성을 고려하여 설계되었습니다.
"""

__version__ = "1.0.0"
__author__ = "Vision System Team"

# 하위 모듈들 import
from .detection.base_detector import BaseDetector
from .recognition.base_recognizer import BaseRecognizer
from .preprocessing.image_processor import ImageProcessor

# 공개 API
__all__ = [
    # 기본 클래스들
    'BaseDetector',
    'BaseRecognizer', 
    'ImageProcessor',
    
    # 유틸리티 함수들
    'get_optimal_device',
    'validate_image',
    'create_processor'
]

def get_optimal_device() -> str:
    """
    현재 하드웨어 환경에 최적화된 디바이스를 반환합니다.
    
    Returns:
        최적 디바이스 ('cpu', 'gpu', 'jetson')
    """
    from common.utils import HardwareDetector
    detector = HardwareDetector()
    return detector.get_optimal_device()

def validate_image(image) -> bool:
    """
    이미지가 유효한지 검증합니다.
    
    Args:
        image: 검증할 이미지
        
    Returns:
        유효성 여부
    """
    if image is None:
        return False
    
    import numpy as np
    if not isinstance(image, np.ndarray):
        return False
    
    if len(image.shape) != 3:
        return False
    
    if image.shape[2] not in [1, 3]:  # 그레이스케일 또는 RGB
        return False
    
    return True

def create_processor(processor_type: str = 'image', **kwargs):
    """
    프로세서 인스턴스를 생성합니다.
    
    Args:
        processor_type: 프로세서 타입 ('image', 'detector', 'recognizer')
        **kwargs: 추가 인자들
        
    Returns:
        프로세서 인스턴스
        
    Raises:
        ValueError: 지원하지 않는 프로세서 타입
    """
    if processor_type == 'image':
        return ImageProcessor(**kwargs)
    elif processor_type == 'detector':
        # BaseDetector는 추상 클래스이므로 구체적인 구현체가 필요
        raise ValueError("detector 타입은 구체적인 구현체를 직접 사용하세요")
    elif processor_type == 'recognizer':
        # BaseRecognizer는 추상 클래스이므로 구체적인 구현체가 필요
        raise ValueError("recognizer 타입은 구체적인 구현체를 직접 사용하세요")
    else:
        raise ValueError(f"지원하지 않는 프로세서 타입: {processor_type}")

# 모듈 정보
MODULE_INFO = {
    'name': 'Vision Core',
    'version': __version__,
    'description': '비전 알고리즘 공통 모듈',
    'modules': {
        'detection': '객체 검출 관련 기능',
        'recognition': '객체 인식 관련 기능', 
        'preprocessing': '이미지 전처리 기능',
        'postprocessing': '후처리 기능 (향후 구현)',
        'tracking': '객체 추적 기능 (향후 구현)',
        'visualization': '시각화 기능 (향후 구현)'
    }
}

def get_module_info() -> dict:
    """모듈 정보를 반환합니다."""
    return MODULE_INFO.copy() 