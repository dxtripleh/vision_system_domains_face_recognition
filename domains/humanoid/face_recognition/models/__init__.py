#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴인식 모델 모듈.

이 모듈은 얼굴 검출 및 인식을 위한 모델들을 포함합니다.
"""

from .face_detection_model import FaceDetectionModel
from .face_recognition_model import FaceRecognitionModel

__all__ = [
    'FaceDetectionModel',
    'FaceRecognitionModel'
] 