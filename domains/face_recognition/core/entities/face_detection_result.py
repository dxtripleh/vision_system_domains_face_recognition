#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Detection Result Entity.

얼굴 검출 결과를 나타내는 엔티티입니다.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from .face import Face


@dataclass
class FaceDetectionResult:
    """얼굴 검출 결과 엔티티"""
    
    image_id: str
    faces: List[Face]
    processing_time_ms: float
    model_name: str
    image_metadata: Dict[str, Any]
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.created_at is None:
            self.created_at = datetime.now()
    
    @property
    def face_count(self) -> int:
        """검출된 얼굴 수"""
        return len(self.faces)
    
    @property
    def has_faces(self) -> bool:
        """얼굴이 검출되었는지 여부"""
        return len(self.faces) > 0
    
    @property
    def high_confidence_faces(self) -> List[Face]:
        """높은 신뢰도의 얼굴들 (0.8 이상)"""
        return [face for face in self.faces if face.confidence >= 0.8]
    
    def get_largest_face(self) -> Optional[Face]:
        """가장 큰 얼굴 반환"""
        if not self.faces:
            return None
        
        return max(self.faces, key=lambda f: f.bbox[2] * f.bbox[3])
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 형식으로 변환"""
        return {
            'image_id': self.image_id,
            'face_count': self.face_count,
            'processing_time_ms': self.processing_time_ms,
            'model_name': self.model_name,
            'image_metadata': self.image_metadata,
            'faces': [
                {
                    'face_id': face.face_id,
                    'bbox': face.bbox,
                    'confidence': face.confidence,
                    'person_id': face.person_id
                }
                for face in self.faces
            ],
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    def get_identified_faces(self) -> List[Face]:
        """신원이 확인된 얼굴들만 반환"""
        return [face for face in self.faces if face.is_identified()]
    
    def get_unidentified_faces(self) -> List[Face]:
        """신원이 확인되지 않은 얼굴들만 반환"""
        return [face for face in self.faces if not face.is_identified()]
    
    def add_face(self, face: Face) -> None:
        """얼굴 추가"""
        self.faces.append(face)
    
    @classmethod
    def create_empty(cls, image_id: str, model_name: str, model_version: str) -> "FaceDetectionResult":
        """빈 검출 결과 생성"""
        return cls(
            image_id=image_id,
            faces=[],
            processing_time_ms=0.0,
            model_name=model_name,
            image_metadata={}
        )
    
    def get_face_count(self) -> int:
        """검출된 얼굴 개수 반환"""
        return len(self.faces)
    
    def has_faces(self) -> bool:
        """얼굴 검출 여부 확인"""
        return len(self.faces) > 0
    
    def get_high_confidence_faces(self, threshold: float = 0.8) -> List[Face]:
        """높은 신뢰도의 얼굴들만 반환"""
        return [face for face in self.faces if face.confidence >= threshold]
    
    def add_face(self, face: Face) -> None:
        """얼굴 추가"""
        self.faces.append(face)
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            'image_id': self.image_id,
            'face_count': self.get_face_count(),
            'processing_time_ms': self.processing_time_ms,
            'model_name': self.model_name,
            'image_metadata': self.image_metadata,
            'faces': [
                {
                    'face_id': face.face_id,
                    'bbox': face.bbox,
                    'confidence': face.confidence,
                    'person_id': face.person_id
                }
                for face in self.faces
            ],
            'created_at': self.created_at.isoformat() if self.created_at else None
        } 