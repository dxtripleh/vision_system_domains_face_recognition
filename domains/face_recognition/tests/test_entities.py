#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition Entities Unit Tests.

얼굴인식 도메인 엔티티들의 단위 테스트입니다.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock

from domains.face_recognition.core.entities.face import Face
from domains.face_recognition.core.entities.person import Person
from domains.face_recognition.core.entities.face_detection_result import FaceDetectionResult
from domains.face_recognition.core.value_objects.face_embedding import FaceEmbedding
from domains.face_recognition.core.value_objects.bounding_box import BoundingBox
from domains.face_recognition.core.value_objects.confidence_score import ConfidenceScore


class TestFace:
    """Face 엔티티 테스트"""
    
    def test_face_creation_valid(self):
        """유효한 Face 생성 테스트"""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        face = Face(
            face_id="test_face_1",
            person_id="person_1",
            embedding=embedding,
            confidence=0.85,
            bbox=[10, 20, 100, 120],
            landmarks=[50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
        )
        
        assert face.face_id == "test_face_1"
        assert face.person_id == "person_1"
        assert face.confidence == 0.85
        assert face.bbox == [10, 20, 100, 120]
        assert face.is_identified() is True
        assert face.has_landmarks() is True
        assert isinstance(face.created_at, datetime)
    
    def test_face_creation_invalid_confidence(self):
        """잘못된 신뢰도로 Face 생성 시 오류 테스트"""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Face(
                face_id="test_face_1",
                person_id="person_1",
                embedding=embedding,
                confidence=1.5,  # 잘못된 값
                bbox=[10, 20, 100, 120]
            )
    
    def test_face_bbox_coordinates(self):
        """바운딩 박스 좌표 계산 테스트"""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        face = Face(
            face_id="test_face_1",
            person_id="person_1",
            embedding=embedding,
            confidence=0.8,
            bbox=[10, 20, 100, 120]  # x, y, w, h
        )
        
        x1, y1, x2, y2 = face.get_bbox_coordinates()
        assert x1 == 10
        assert y1 == 20
        assert x2 == 110  # 10 + 100
        assert y2 == 140  # 20 + 120


class TestPerson:
    """Person 엔티티 테스트"""
    
    def test_person_creation_valid(self):
        """유효한 Person 생성 테스트"""
        person = Person(
            person_id="person_1",
            name="홍길동",
            face_embeddings=[],
            metadata={"age": 30, "department": "개발팀"}
        )
        
        assert person.person_id == "person_1"
        assert person.name == "홍길동"
        assert person.face_embeddings == []
        assert person.metadata["age"] == 30
        assert isinstance(person.created_at, datetime)
    
    def test_person_create_new(self):
        """새 Person 생성 클래스 메서드 테스트"""
        person = Person.create_new("김철수", {"role": "manager"})
        
        assert len(person.person_id) > 0  # UUID 생성됨
        assert person.name == "김철수"
        assert person.face_embeddings == []
        assert person.metadata["role"] == "manager"


class TestFaceEmbedding:
    """FaceEmbedding 값 객체 테스트"""
    
    def test_face_embedding_creation_valid(self):
        """유효한 FaceEmbedding 생성 테스트"""
        vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        embedding = FaceEmbedding(vector=vector, model_name="arcface")
        
        assert embedding.dimension == 4
        assert embedding.model_name == "arcface"
        assert embedding.norm > 0
    
    def test_face_embedding_normalize(self):
        """FaceEmbedding 정규화 테스트"""
        vector = np.array([3.0, 4.0], dtype=np.float32)  # 크기 5인 벡터
        embedding = FaceEmbedding(vector=vector, model_name="test")
        
        normalized = embedding.normalize()
        assert abs(normalized.norm - 1.0) < 1e-6
        assert normalized.is_normalized is True
    
    def test_face_embedding_similarity(self):
        """FaceEmbedding 유사도 계산 테스트"""
        vector1 = np.array([1.0, 0.0], dtype=np.float32)
        vector2 = np.array([0.0, 1.0], dtype=np.float32)
        vector3 = np.array([1.0, 0.0], dtype=np.float32)
        
        embedding1 = FaceEmbedding(vector=vector1, model_name="test")
        embedding2 = FaceEmbedding(vector=vector2, model_name="test")
        embedding3 = FaceEmbedding(vector=vector3, model_name="test")
        
        # 수직인 벡터들의 코사인 유사도는 0
        similarity_12 = embedding1.similarity_with(embedding2, 'cosine')
        assert abs(similarity_12) < 1e-6
        
        # 동일한 벡터들의 코사인 유사도는 1
        similarity_13 = embedding1.similarity_with(embedding3, 'cosine')
        assert abs(similarity_13 - 1.0) < 1e-6


class TestBoundingBox:
    """BoundingBox 값 객체 테스트"""
    
    def test_bounding_box_creation_valid(self):
        """유효한 BoundingBox 생성 테스트"""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 80
        assert bbox.right == 110
        assert bbox.bottom == 100
        assert bbox.area == 8000
    
    def test_bounding_box_contains_point(self):
        """BoundingBox 내부 점 확인 테스트"""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        
        assert bbox.contains_point(50, 50) is True  # 내부
        assert bbox.contains_point(10, 20) is True  # 경계
        assert bbox.contains_point(5, 50) is False  # 완전히 밖


class TestConfidenceScore:
    """ConfidenceScore 값 객체 테스트"""
    
    def test_confidence_score_creation_valid(self):
        """유효한 ConfidenceScore 생성 테스트"""
        score = ConfidenceScore(value=0.85)
        
        assert score.value == 0.85
        assert score.percentage == 85.0
        assert score.is_high is True
        assert score.confidence_level == "high"
    
    def test_confidence_score_levels(self):
        """ConfidenceScore 레벨 확인 테스트"""
        high_score = ConfidenceScore(value=0.9)
        medium_score = ConfidenceScore(value=0.6)
        low_score = ConfidenceScore(value=0.3)
        
        assert high_score.confidence_level == "high"
        assert medium_score.confidence_level == "medium"
        assert low_score.confidence_level == "low"


if __name__ == "__main__":
    pytest.main([__file__]) 