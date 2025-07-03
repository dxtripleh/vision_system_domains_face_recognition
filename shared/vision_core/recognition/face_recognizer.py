#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognizer Interface.

얼굴 인식을 위한 공통 인터페이스와 유틸리티를 제공합니다.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from common.logging import get_logger

logger = get_logger(__name__)


class FaceRecognizerInterface(ABC):
    """얼굴 인식기 인터페이스"""
    
    @abstractmethod
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 임베딩 추출"""
        pass
    
    @abstractmethod
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """임베딩 유사도 계산"""
        pass


class FaceRecognizer:
    """통합 얼굴 인식기"""
    
    def __init__(self, recognizer_type: str = "arcface", **kwargs):
        """
        인식기 초기화
        
        Args:
            recognizer_type: 인식기 타입 ('arcface', 'facenet', 'mobilefacenet')
            **kwargs: 인식기별 추가 설정
        """
        self.recognizer_type = recognizer_type
        self.recognizer = None
        
        self._initialize_recognizer(kwargs)
        
    def _initialize_recognizer(self, config: Dict[str, Any]) -> None:
        """인식기 초기화"""
        if self.recognizer_type == "arcface":
            self.recognizer = ArcFaceRecognizerWrapper(config)
        elif self.recognizer_type == "facenet":
            self.recognizer = FaceNetRecognizerWrapper(config)
        elif self.recognizer_type == "mobilefacenet":
            self.recognizer = MobileFaceNetRecognizerWrapper(config)
        else:
            raise ValueError(f"Unknown recognizer type: {self.recognizer_type}")
        
        logger.info(f"Face recognizer initialized: {self.recognizer_type}")
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        얼굴 임베딩 추출
        
        Args:
            face_image: 입력 얼굴 이미지 (BGR)
            
        Returns:
            Optional[np.ndarray]: 추출된 임베딩
        """
        return self.recognizer.extract_embedding(face_image)
    
    def extract_embeddings_batch(self, face_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        배치 얼굴 임베딩 추출
        
        Args:
            face_images: 입력 얼굴 이미지 리스트
            
        Returns:
            List[Optional[np.ndarray]]: 추출된 임베딩 리스트
        """
        embeddings = []
        for face_image in face_images:
            embedding = self.extract_embedding(face_image)
            embeddings.append(embedding)
        
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        임베딩 유사도 계산
        
        Args:
            embedding1: 첫 번째 임베딩
            embedding2: 두 번째 임베딩
            
        Returns:
            float: 유사도 점수
        """
        return self.recognizer.compute_similarity(embedding1, embedding2)
    
    def identify_face(self, 
                     query_embedding: np.ndarray,
                     reference_embeddings: List[Tuple[str, np.ndarray]],
                     threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        """
        얼굴 식별 (1:N 매칭)
        
        Args:
            query_embedding: 질의 임베딩
            reference_embeddings: 참조 임베딩 리스트 [(person_id, embedding), ...]
            threshold: 유사도 임계값
            
        Returns:
            Optional[Tuple[str, float]]: (person_id, similarity) 또는 None
        """
        best_match = None
        best_similarity = 0.0
        
        for person_id, ref_embedding in reference_embeddings:
            similarity = self.compute_similarity(query_embedding, ref_embedding)
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = (person_id, similarity)
        
        return best_match
    
    def verify_face(self,
                   embedding1: np.ndarray,
                   embedding2: np.ndarray,
                   threshold: float = 0.6) -> Tuple[bool, float]:
        """
        얼굴 검증 (1:1 매칭)
        
        Args:
            embedding1: 첫 번째 임베딩
            embedding2: 두 번째 임베딩
            threshold: 유사도 임계값
            
        Returns:
            Tuple[bool, float]: (검증 결과, 유사도)
        """
        similarity = self.compute_similarity(embedding1, embedding2)
        is_same = similarity >= threshold
        
        return is_same, similarity


class ArcFaceRecognizerWrapper(FaceRecognizerInterface):
    """ArcFace 인식기 래퍼"""
    
    def __init__(self, config: Dict[str, Any]):
        """초기화"""
        try:
            # 의존성 주입을 통한 모델 로딩 (의존성 규칙 준수)
            model_path = config.get('model_path', 'models/weights/face_recognition_arcface_r50_20250628.onnx')
            
            # 외부에서 주입된 recognizer 사용
            if 'recognizer_instance' in config:
                self.recognizer = config['recognizer_instance']
            else:
                # 기본 구현 (향후 개선)
                logger.warning("Using fallback recognizer implementation")
                self.recognizer = self._create_fallback_recognizer(config)
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ArcFace recognizer: {e}")
    
    def _create_fallback_recognizer(self, config: Dict[str, Any]):
        """기본 인식기 생성 (의존성 규칙 준수)"""
        # 기본 구현 - 실제로는 외부에서 주입받아야 함
        class FallbackRecognizer:
            def extract_embedding(self, face_image):
                # 기본 임베딩 생성 (실제로는 모델 사용)
                return np.random.rand(512)  # 임시 구현
                
            def load_model(self):
                return True
        
        return FallbackRecognizer()
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 임베딩 추출"""
        embedding_obj = self.recognizer.extract_embedding(face_image)
        return embedding_obj.vector if embedding_obj else None
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """임베딩 유사도 계산"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # 코사인 유사도 계산
        dot_product = np.dot(embedding1, embedding2)
        norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)


class FaceNetRecognizerWrapper(FaceRecognizerInterface):
    """FaceNet 인식기 래퍼 (향후 구현)"""
    
    def __init__(self, config: Dict[str, Any]):
        """초기화"""
        raise NotImplementedError("FaceNet recognizer not implemented yet")
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 임베딩 추출"""
        raise NotImplementedError("FaceNet recognizer not implemented yet")
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """임베딩 유사도 계산"""
        raise NotImplementedError("FaceNet recognizer not implemented yet")


class MobileFaceNetRecognizerWrapper(FaceRecognizerInterface):
    """MobileFaceNet 인식기 래퍼 (향후 구현)"""
    
    def __init__(self, config: Dict[str, Any]):
        """초기화"""
        raise NotImplementedError("MobileFaceNet recognizer not implemented yet")
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 임베딩 추출"""
        raise NotImplementedError("MobileFaceNet recognizer not implemented yet")
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """임베딩 유사도 계산"""
        raise NotImplementedError("MobileFaceNet recognizer not implemented yet")


def create_face_recognizer(recognizer_type: str = "arcface", **kwargs) -> FaceRecognizer:
    """
    얼굴 인식기 팩토리 함수
    
    Args:
        recognizer_type: 인식기 타입
        **kwargs: 인식기별 설정
        
    Returns:
        FaceRecognizer: 생성된 인식기
    """
    return FaceRecognizer(recognizer_type=recognizer_type, **kwargs) 