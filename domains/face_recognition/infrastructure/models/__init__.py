#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴인식 모델 패키지.

이 패키지는 얼굴 검출 및 인식을 위한 AI 모델 구현체들을 포함합니다.
"""

from .retinaface_detector import RetinaFaceDetector

__all__ = [
    "RetinaFaceDetector"
] 