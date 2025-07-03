#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모델 관리 패키지.

이 패키지는 얼굴인식 모델의 다운로드, 검증, 관리 기능을 제공합니다.
"""

from .download_models import ModelDownloader

__all__ = ["ModelDownloader"] 