#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition API.

얼굴인식 도메인의 REST API를 제공합니다.
"""

from .face_recognition_api import FaceRecognitionAPI
from .routes import detection_routes, recognition_routes, management_routes

__all__ = [
    "FaceRecognitionAPI",
    "detection_routes",
    "recognition_routes", 
    "management_routes"
] 