#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition Entities

얼굴인식 도메인의 핵심 엔티티들을 정의합니다.
"""

from .face import Face
from .person import Person
from .face_detection_result import FaceDetectionResult

__all__ = ["Face", "Person", "FaceDetectionResult"] 