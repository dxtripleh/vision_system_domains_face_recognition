#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Humanoid Face Recognition Domain.

얼굴인식 도메인입니다. ONNX 기반 얼굴인식 모델을 지원합니다.
"""

__version__ = "1.0.0"
__author__ = "Vision System Team"

# 주요 모듈들 import
from .models import FaceDetectionModel, FaceRecognitionModel
from .services import FaceRecognitionService, IntegratedFaceRecognitionService

__all__ = [
    'FaceDetectionModel',
    'FaceRecognitionModel', 
    'FaceRecognitionService',
    'IntegratedFaceRecognitionService'
] 