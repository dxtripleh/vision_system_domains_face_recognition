#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Person Entity

인물 정보를 나타내는 핵심 엔티티입니다.
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
import uuid


@dataclass
class Person:
    """인물 엔티티 클래스"""
    
    person_id: str
    name: str
    face_embeddings: List  # 여러 얼굴 임베딩 저장
    metadata: Optional[dict] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
            
        # 이름 검증
        if not self.name or not self.name.strip():
            raise ValueError("Name cannot be empty")
    
    @classmethod
    def create_new(cls, name: str, metadata: Optional[dict] = None) -> "Person":
        """새로운 인물 생성"""
        person_id = str(uuid.uuid4())
        return cls(
            person_id=person_id,
            name=name.strip(),
            face_embeddings=[],
            metadata=metadata or {}
        )
    
    def add_face_embedding(self, embedding) -> None:
        """얼굴 임베딩 추가"""
        self.face_embeddings.append(embedding)
        self.updated_at = datetime.now()
    
    def get_embedding_count(self) -> int:
        """등록된 얼굴 임베딩 개수 반환"""
        return len(self.face_embeddings)
    
    def has_embeddings(self) -> bool:
        """얼굴 임베딩 존재 여부 확인"""
        return len(self.face_embeddings) > 0 