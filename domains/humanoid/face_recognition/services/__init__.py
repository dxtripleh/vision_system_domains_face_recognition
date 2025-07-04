#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴인식 서비스 모듈.

이 모듈은 얼굴인식 관련 서비스들을 포함합니다.
"""

from .service import FaceRecognitionService
from .face_recognition_service import FaceRecognitionService as IntegratedFaceRecognitionService

__all__ = [
    'FaceRecognitionService',
    'IntegratedFaceRecognitionService'
] 