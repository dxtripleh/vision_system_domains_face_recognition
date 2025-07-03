#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Repositories package for face recognition domain.

얼굴인식 도메인의 저장소 패키지입니다.
"""

from .face_repository import FaceRepository
from .person_repository import PersonRepository

__all__ = [
    'FaceRepository',
    'PersonRepository'
] 