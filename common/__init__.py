#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common package.

이 패키지는 범용 유틸리티 및 기능을 포함합니다.
"""

__version__ = "1.0.0"
__author__ = "Vision System Team"

# 공통 모듈 목록
COMMON_MODULES = [
    "config",
    "logging",
    "utils"
]

def get_common_module(module_name: str):
    """공통 모듈을 동적으로 로드합니다."""
    if module_name not in COMMON_MODULES:
        raise ValueError(f"공통 모듈 '{module_name}'을 찾을 수 없습니다.")
    
    try:
        module = __import__(f"common.{module_name}", fromlist=["*"])
        return module
    except ImportError as e:
        raise ImportError(f"공통 모듈 '{module_name}' 로드 실패: {e}") 