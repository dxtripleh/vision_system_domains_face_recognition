#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
이미지 전처리 공통 모듈.

이 모듈은 이미지 전처리 관련 공통 기능을 제공합니다.
크로스 플랫폼 호환성을 고려하여 설계되었습니다.
"""

__version__ = "1.0.0"
__author__ = "Vision System Team"

# 이미지 프로세서 클래스 import
from .image_processor import ImageProcessor

# 공개 API
__all__ = [
    'ImageProcessor'
]

# 모듈 정보
MODULE_INFO = {
    'name': 'Preprocessing',
    'version': __version__,
    'description': '이미지 전처리 공통 모듈',
    'classes': {
        'ImageProcessor': '이미지 전처리 클래스'
    }
}

def get_module_info() -> dict:
    """모듈 정보를 반환합니다."""
    return MODULE_INFO.copy() 