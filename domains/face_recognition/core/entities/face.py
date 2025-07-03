#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Entity

얼굴 정보를 나타내는 핵심 엔티티입니다.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from datetime import datetime


@dataclass
class Face:
    """얼굴 엔티티 클래스"""
    
    face_id: str
    person_id: Optional[str]
    embedding: np.ndarray
    confidence: float
    bbox: List[float]  # [x, y, width, height]
    landmarks: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.created_at is None:
            self.created_at = datetime.now()
            
        # 신뢰도 검증
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
            
        # 바운딩 박스 검증
        if len(self.bbox) != 4:
            raise ValueError("Bbox must have exactly 4 values [x, y, width, height]")
    
    def get_bbox_coordinates(self) -> tuple:
        """바운딩 박스 좌표 반환 (x1, y1, x2, y2)"""
        x, y, w, h = self.bbox
        return (x, y, x + w, y + h)
    
    def has_landmarks(self) -> bool:
        """랜드마크 존재 여부 확인"""
        return self.landmarks is not None and len(self.landmarks) > 0
    
    def is_identified(self) -> bool:
        """신원 확인 여부"""
        return self.person_id is not None 