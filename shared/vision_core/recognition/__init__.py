#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
객체 인식 공통 모듈.

이 모듈은 객체 인식 관련 공통 기능을 제공합니다.
크로스 플랫폼 호환성을 고려하여 설계되었습니다.
"""

__version__ = "1.0.0"
__author__ = "Vision System Team"

# 기본 인식기 클래스 import
from .base_recognizer import BaseRecognizer

# 공개 API
__all__ = [
    'BaseRecognizer'
]

# 모듈 정보
MODULE_INFO = {
    'name': 'Recognition',
    'version': __version__,
    'description': '객체 인식 공통 모듈',
    'classes': {
        'BaseRecognizer': '모든 인식기의 기본 클래스'
    }
}

def get_module_info() -> dict:
    """모듈 정보를 반환합니다."""
    return MODULE_INFO.copy() 