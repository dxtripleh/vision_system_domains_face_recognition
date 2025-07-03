#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition Services Unit Tests.

얼굴인식 서비스들의 단위 테스트입니다.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService
from domains.face_recognition.core.services.face_detection_service import FaceDetectionService
from domains.face_recognition.core.services.face_matching_service import FaceMatchingService
from domains.face_recognition.core.entities.face import Face
from domains.face_recognition.core.entities.person import Person
from domains.face_recognition.core.value_objects.face_embedding import FaceEmbedding


class TestFaceRecognitionService:
    """FaceRecognitionService 테스트"""
    
    @pytest.fixture
    def mock_recognizer(self):
        """Mock 인식기"""
        recognizer = Mock()
        recognizer.extract_embedding.return_value = FaceEmbedding(
            vector=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            model_name="mock"
        )
        return recognizer
    
    @pytest.fixture
    def mock_person_repository(self):
        """Mock 인물 저장소"""
        repo = Mock()
        repo.find_all.return_value = []
        repo.save.return_value = True
        return repo
    
    @pytest.fixture
    def mock_face_repository(self):
        """Mock 얼굴 저장소"""
        repo = Mock()
        repo.find_by_person_id.return_value = []
        repo.save.return_value = True
        return repo
    
    @pytest.fixture
    def service(self, mock_recognizer, mock_person_repository, mock_face_repository):
        """FaceRecognitionService 인스턴스"""
        return FaceRecognitionService(
            recognizer=mock_recognizer,
            person_repository=mock_person_repository,
            face_repository=mock_face_repository,
            use_mock=True
        )
    
    def test_extract_embedding_valid_image(self, service):
        """유효한 이미지에서 임베딩 추출 테스트"""
        # 가짜 얼굴 이미지 생성 (100x100 RGB)
        face_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        embedding = service.extract_embedding(face_image)
        
        assert embedding is not None
        assert isinstance(embedding, FaceEmbedding)
        assert embedding.dimension == 4
    
    def test_extract_embedding_invalid_image(self, service):
        """잘못된 이미지에서 임베딩 추출 시 오류 테스트"""
        with pytest.raises(ValueError, match="유효하지 않은 얼굴 이미지입니다"):
            service.extract_embedding(None)
    
    def test_register_person_success(self, service):
        """인물 등록 성공 테스트"""
        face_images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            for _ in range(2)
        ]
        
        person_id = service.register_person(
            name="테스트 사용자",
            face_images=face_images,
            metadata={"department": "개발팀"}
        )
        
        assert person_id is not None
        assert len(person_id) > 0
        
        # 저장소 호출 확인
        service.person_repository.save.assert_called_once()
    
    def test_register_person_empty_name(self, service):
        """빈 이름으로 인물 등록 시 오류 테스트"""
        face_images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)]
        
        with pytest.raises(ValueError, match="이름이 필요합니다"):
            service.register_person("", face_images)


class TestFaceDetectionService:
    """FaceDetectionService 테스트"""
    
    @pytest.fixture
    def service(self):
        """FaceDetectionService 인스턴스"""
        return FaceDetectionService(use_mock=True)
    
    def test_detect_faces_valid_image(self, service):
        """유효한 이미지에서 얼굴 검출 테스트"""
        # 가짜 이미지 생성 (640x480 RGB)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = service.detect_faces(test_image)
        
        assert result is not None
        assert result.image_id is not None
        assert result.processing_time_ms >= 0
        assert result.model_name == "opencv_haarcascade"
        assert isinstance(result.faces, list)
    
    def test_detect_faces_invalid_image(self, service):
        """잘못된 이미지에서 얼굴 검출 시 오류 테스트"""
        with pytest.raises(ValueError, match="유효하지 않은 이미지입니다"):
            service.detect_faces(None)


class TestFaceMatchingService:
    """FaceMatchingService 테스트"""
    
    @pytest.fixture
    def service(self):
        """FaceMatchingService 인스턴스"""
        return FaceMatchingService()
    
    @pytest.fixture
    def sample_faces(self):
        """테스트용 얼굴 데이터"""
        face1 = Face(
            face_id="face_1",
            person_id=None,
            embedding=FaceEmbedding(
                vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                model_name="test"
            ),
            confidence=0.9,
            bbox=[10, 10, 50, 50]
        )
        
        face2 = Face(
            face_id="face_2", 
            person_id=None,
            embedding=FaceEmbedding(
                vector=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                model_name="test"
            ),
            confidence=0.8,
            bbox=[60, 60, 50, 50]
        )
        
        return face1, face2
    
    def test_compare_faces_identical(self, service, sample_faces):
        """동일한 얼굴 비교 테스트"""
        face1, _ = sample_faces
        
        # 같은 얼굴과 비교
        is_same, similarity, confidence_level = service.compare_faces(face1, face1)
        
        assert is_same is True
        assert similarity == 1.0
        assert confidence_level == "high"
    
    def test_compare_faces_different(self, service, sample_faces):
        """다른 얼굴 비교 테스트"""
        face1, face2 = sample_faces
        
        # 수직인 벡터들 (코사인 유사도 0)
        is_same, similarity, confidence_level = service.compare_faces(face1, face2)
        
        assert is_same is False
        assert abs(similarity) < 1e-6
        assert confidence_level == "low"


if __name__ == "__main__":
    pytest.main([__file__]) 