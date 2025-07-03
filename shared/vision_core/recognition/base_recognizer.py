#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Recognizer for Vision Core.

비전 시스템의 공통 인식기 인터페이스입니다.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
import time

from common.logging import get_logger

logger = get_logger(__name__)


class BaseRecognizer(ABC):
    """기본 인식기 인터페이스"""
    
    def __init__(self, 
                 model_path: str,
                 input_size: Tuple[int, int] = (112, 112),
                 embedding_size: int = 512,
                 use_gpu: bool = True):
        """
        인식기 초기화
        
        Args:
            model_path: 모델 파일 경로
            input_size: 입력 이미지 크기 (width, height)
            embedding_size: 임베딩 벡터 차원
            use_gpu: GPU 사용 여부
        """
        self.model_path = model_path
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.use_gpu = use_gpu
        self.is_loaded = False
        self.model = None
        
        logger.info(f"BaseRecognizer initialized: {self.__class__.__name__}")
    
    @abstractmethod
    def load_model(self) -> bool:
        """모델 로딩"""
        pass
    
    @abstractmethod
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """단일 얼굴에서 임베딩 추출"""
        pass
    
    @abstractmethod
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """얼굴 이미지 전처리"""
        pass
    
    def extract_embeddings_batch(self, face_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """배치 임베딩 추출"""
        embeddings = []
        for face_image in face_images:
            embedding = self.extract_embedding(face_image)
            embeddings.append(embedding)
        return embeddings
    
    def compute_similarity(self, 
                          embedding1: np.ndarray, 
                          embedding2: np.ndarray,
                          metric: str = 'cosine') -> float:
        """임베딩 간 유사도 계산"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        if embedding1.shape != embedding2.shape:
            logger.warning(f"Embedding shape mismatch: {embedding1.shape} vs {embedding2.shape}")
            return 0.0
        
        if metric == 'cosine':
            # 코사인 유사도
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        
        elif metric == 'euclidean':
            # 유클리드 거리 (거리이므로 유사도로 변환)
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(1.0 / (1.0 + distance))
        
        elif metric == 'dot':
            # 내적
            return float(np.dot(embedding1, embedding2))
        
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    def identify(self, 
                query_embedding: np.ndarray,
                reference_embeddings: List[Tuple[str, np.ndarray]],
                threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        """
        1:N 얼굴 식별
        
        Args:
            query_embedding: 쿼리 임베딩
            reference_embeddings: (ID, 임베딩) 튜플의 리스트
            threshold: 유사도 임계값
            
        Returns:
            Optional[Tuple[str, float]]: (매칭된 ID, 유사도) 또는 None
        """
        if query_embedding is None:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for person_id, ref_embedding in reference_embeddings:
            similarity = self.compute_similarity(query_embedding, ref_embedding)
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = (person_id, similarity)
        
        return best_match
    
    def verify(self,
              embedding1: np.ndarray,
              embedding2: np.ndarray,
              threshold: float = 0.6) -> Tuple[bool, float]:
        """
        1:1 얼굴 검증
        
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
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """임베딩 L2 정규화"""
        if embedding is None:
            return None
        
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        
        return embedding / norm
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_path': self.model_path,
            'input_size': self.input_size,
            'embedding_size': self.embedding_size,
            'use_gpu': self.use_gpu,
            'is_loaded': self.is_loaded,
            'model_type': self.__class__.__name__
        }
    
    def validate_face_image(self, face_image: np.ndarray) -> bool:
        """얼굴 이미지 검증"""
        if face_image is None:
            return False
        
        if not isinstance(face_image, np.ndarray):
            return False
        
        if face_image.size == 0:
            return False
        
        if len(face_image.shape) not in [2, 3]:
            return False
        
        # 최소 크기 확인
        min_size = 32
        if face_image.shape[0] < min_size or face_image.shape[1] < min_size:
            return False
        
        return True


class FaceRecognizer(BaseRecognizer):
    """얼굴 인식기 기본 클래스"""
    
    def __init__(self, 
                 model_path: str,
                 input_size: Tuple[int, int] = (112, 112),
                 embedding_size: int = 512,
                 use_gpu: bool = True):
        super().__init__(model_path, input_size, embedding_size, use_gpu)
        self.mean = [127.5, 127.5, 127.5]  # 기본 정규화 값
        self.std = [127.5, 127.5, 127.5]
    
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """얼굴 이미지 전처리 (기본 구현)"""
        if not self.validate_face_image(face_image):
            raise ValueError("Invalid face image")
        
        # 크기 조정
        resized = cv2.resize(face_image, self.input_size)
        
        # 정규화 (0-255 -> -1~1)
        normalized = (resized.astype(np.float32) - 127.5) / 127.5
        
        # 차원 변경 (H, W, C) -> (C, H, W)
        if len(normalized.shape) == 3:
            preprocessed = np.transpose(normalized, (2, 0, 1))
        else:
            preprocessed = np.expand_dims(normalized, axis=0)
        
        # 배치 차원 추가 (C, H, W) -> (1, C, H, W)
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        return preprocessed


class ArcFaceRecognizer(FaceRecognizer):
    """ArcFace 기반 얼굴 인식기"""
    
    def __init__(self, 
                 model_path: str,
                 input_size: Tuple[int, int] = (112, 112),
                 embedding_size: int = 512,
                 use_gpu: bool = True):
        super().__init__(model_path, input_size, embedding_size, use_gpu)
        self.net = None
    
    def load_model(self) -> bool:
        """ArcFace 모델 로딩"""
        try:
            if not self.model_path or not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # OpenCV DNN으로 모델 로딩
            if self.model_path.endswith('.onnx'):
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
            elif self.model_path.endswith('.pb'):
                self.net = cv2.dnn.readNetFromTensorflow(self.model_path)
            else:
                logger.error(f"Unsupported model format: {self.model_path}")
                return False
            
            # GPU 설정
            if self.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info("Using GPU backend for face recognition")
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logger.info("Using CPU backend for face recognition")
            
            self.is_loaded = True
            logger.info("ArcFace model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ArcFace model: {str(e)}")
            return False
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """ArcFace로 임베딩 추출"""
        if not self.is_loaded:
            if not self.load_model():
                return None
        
        if not self.validate_face_image(face_image):
            return None
        
        try:
            # 이미지 전처리
            blob = self.preprocess(face_image)
            
            # 모델 추론
            self.net.setInput(blob)
            output = self.net.forward()
            
            # 임베딩 벡터 추출
            embedding = output[0].flatten()
            
            # L2 정규화
            embedding = self.normalize_embedding(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {str(e)}")
            return None


class MockFaceRecognizer(FaceRecognizer):
    """Mock 얼굴 인식기 (테스트용)"""
    
    def __init__(self, 
                 embedding_size: int = 512,
                 use_deterministic: bool = True):
        super().__init__(
            model_path="mock_model",
            input_size=(112, 112),
            embedding_size=embedding_size,
            use_gpu=False
        )
        self.use_deterministic = use_deterministic
        
    def load_model(self) -> bool:
        """Mock 모델 로딩 (항상 성공)"""
        self.is_loaded = True
        logger.info("Mock face recognizer loaded")
        return True
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Mock 임베딩 생성"""
        if not self.validate_face_image(face_image):
            return None
        
        if self.use_deterministic:
            # 이미지 내용 기반 결정론적 임베딩
            hash_value = hash(face_image.tobytes()) % (2**32)
            np.random.seed(hash_value)
        
        # 랜덤 임베딩 생성
        embedding = np.random.randn(self.embedding_size).astype(np.float32)
        
        # L2 정규화
        embedding = self.normalize_embedding(embedding)
        
        return embedding


def create_face_recognizer(recognizer_type: str = "arcface", **kwargs) -> FaceRecognizer:
    """
    얼굴 인식기 팩토리 함수
    
    Args:
        recognizer_type: 인식기 타입 ("arcface", "facenet", "mock" 등)
        **kwargs: 인식기별 설정
        
    Returns:
        FaceRecognizer: 생성된 얼굴 인식기
    """
    if recognizer_type.lower() == "arcface":
        return ArcFaceRecognizer(**kwargs)
    elif recognizer_type.lower() == "mock":
        return MockFaceRecognizer(**kwargs)
    elif recognizer_type.lower() == "facenet":
        raise NotImplementedError("FaceNet recognizer not implemented yet")
    else:
        raise ValueError(f"Unsupported recognizer type: {recognizer_type}")


# 필요한 import 추가
import os 