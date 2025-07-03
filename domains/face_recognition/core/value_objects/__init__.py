#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition Core Value Objects.

얼굴인식 도메인의 값 객체들을 정의합니다.
"""

from .bounding_box import BoundingBox
from .confidence_score import ConfidenceScore
from .face_embedding import FaceEmbedding
from .point import Point

__all__ = [
    'BoundingBox',
    'ConfidenceScore', 
    'FaceEmbedding',
    'Point'
] 