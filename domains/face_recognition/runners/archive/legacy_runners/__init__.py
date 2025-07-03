#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
데이터 수집 관련 실행 스크립트 모듈

이 모듈은 얼굴 인식 시스템의 데이터 수집 관련 실행 스크립트들을 포함합니다.
"""

__version__ = "1.0.0"
__author__ = "Vision System Team"

# 모듈 설명
__all__ = [
    "run_auto_face_collector",
    "run_unified_face_capture", 
    "run_capture_and_register"
]

# 모듈 메타데이터
MODULES = {
    "run_auto_face_collector": "자동 얼굴 수집기",
    "run_unified_face_capture": "통합 얼굴 캡처 시스템",
    "run_capture_and_register": "실시간 캡처 및 등록"
}

# 이 폴더에는 얼굴인식 도메인의 데이터 수집 관련 실행 스크립트들이 있습니다. 