#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Domains package.

이 패키지는 비전 시스템의 도메인별 기능을 포함합니다.
각 도메인은 독립적으로 개발되며, 공통 모듈을 통해 상호작용합니다.
"""

__version__ = "1.0.0"
__author__ = "Vision System Team"

# 도메인 목록
AVAILABLE_DOMAINS = [
    "humanoid.face_recognition",
    "factory.defect_detection",  # 향후
    "infrastructure.powerline_inspection"  # 향후
]

def get_domain(domain_name: str):
    """도메인 모듈을 동적으로 로드합니다."""
    try:
        module = __import__(f"domains.{domain_name}", fromlist=["*"])
        return module
    except ImportError as e:
        raise ImportError(f"도메인 '{domain_name}'을 찾을 수 없습니다: {e}")

def list_available_domains():
    """사용 가능한 도메인 목록을 반환합니다."""
    return AVAILABLE_DOMAINS.copy() 