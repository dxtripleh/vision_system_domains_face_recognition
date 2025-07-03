#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴 인식 시스템 실행 스크립트 패키지

이 패키지는 얼굴 인식 시스템의 모든 실행 스크립트들을 기능별로 분류하여 포함합니다.
"""

__version__ = "1.0.0"
__author__ = "Vision System Team"

# 하위 모듈들
__all__ = [
    "data_collection",
    "recognition", 
    "management",
    "training"
]

# 패키지 구조
PACKAGE_STRUCTURE = {
    "data_collection": {
        "description": "데이터 수집 관련 실행 스크립트",
        "scripts": [
            "run_auto_face_collector",
            "run_unified_face_capture",
            "run_capture_and_register"
        ]
    },
    "recognition": {
        "description": "얼굴 인식 관련 실행 스크립트",
        "scripts": [
            "run_realtime_recognition",
            "run_demo",
            "run_advanced_recognition"
        ]
    },
    "management": {
        "description": "시스템 관리 관련 실행 스크립트",
        "scripts": [
            "run_group_manager"
        ]
    },
    "training": {
        "description": "모델 훈련 관련 실행 스크립트",
        "scripts": [
            "run_model_training_pipeline"
        ]
    }
}
