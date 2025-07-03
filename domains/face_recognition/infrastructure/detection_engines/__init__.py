#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detection Engines Module.

얼굴 검출 엔진들을 관리하는 모듈입니다.
"""

from typing import Dict, Type, Any
from .opencv_detection_engine import OpenCVDetectionEngine
from .retinaface_detection_engine import RetinaFaceDetectionEngine

# 사용 가능한 검출 엔진들
AVAILABLE_ENGINES = {
    'opencv': OpenCVDetectionEngine,
    'retinaface': RetinaFaceDetectionEngine,
}

def create_detection_engine(engine_type: str, config: Dict[str, Any]):
    """
    검출 엔진 팩토리 함수
    
    Args:
        engine_type: 엔진 타입 ('opencv', 'retinaface')
        config: 엔진 설정
        
    Returns:
        검출 엔진 인스턴스
    """
    if engine_type not in AVAILABLE_ENGINES:
        raise ValueError(f"Unknown detection engine: {engine_type}")
    
    engine_class = AVAILABLE_ENGINES[engine_type]
    return engine_class(config)

__all__ = [
    'OpenCVDetectionEngine',
    'RetinaFaceDetectionEngine', 
    'create_detection_engine',
    'AVAILABLE_ENGINES'
] 