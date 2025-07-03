#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared package.

이 패키지는 모든 도메인에서 공통으로 사용되는 모듈을 포함합니다.
"""

__version__ = "1.0.0"
__author__ = "Vision System Team"

# 공유 모듈 목록
SHARED_MODULES = [
    "vision_core",
    "security"
]

def get_shared_module(module_name: str):
    """공유 모듈을 동적으로 로드합니다."""
    if module_name not in SHARED_MODULES:
        raise ValueError(f"공유 모듈 '{module_name}'을 찾을 수 없습니다.")
    
    try:
        module = __import__(f"shared.{module_name}", fromlist=["*"])
        return module
    except ImportError as e:
        raise ImportError(f"공유 모듈 '{module_name}' 로드 실패: {e}") 