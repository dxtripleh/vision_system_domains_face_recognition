#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Factory Defect Detection Domain.

공장 불량 검출을 위한 도메인입니다.
YOLOv8 기반 ONNX 모델을 사용하여 실시간 불량 검출을 수행합니다.
"""

__version__ = "1.0.0"
__author__ = "Vision System Team"

# 불량 유형 정의
DEFECT_TYPES = {
    'scratch': {'id': 0, 'name': '스크래치', 'color': [0, 0, 255]},
    'dent': {'id': 1, 'name': '함몰', 'color': [0, 255, 0]},
    'crack': {'id': 2, 'name': '균열', 'color': [255, 0, 0]},
    'discoloration': {'id': 3, 'name': '변색', 'color': [255, 255, 0]},
    'contamination': {'id': 4, 'name': '오염', 'color': [255, 0, 255]}
}

# 품질 관리 임계값
QUALITY_THRESHOLDS = {
    'alert_threshold': 0.05,     # 5% 불량률에서 경고
    'critical_threshold': 0.1,   # 10% 불량률에서 심각 알림
    'sampling_rate': 1.0         # 100% 샘플링
}

def get_defect_info(defect_id: int) -> dict:
    """불량 ID에 해당하는 정보를 반환합니다."""
    for defect_type, info in DEFECT_TYPES.items():
        if info['id'] == defect_id:
            return info
    return None 