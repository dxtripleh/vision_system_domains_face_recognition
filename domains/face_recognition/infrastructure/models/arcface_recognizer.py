#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ArcFace 얼굴 인식기.

이 모듈은 ArcFace 모델을 사용한 고성능 얼굴 인식기를 구현합니다.
얼굴 이미지에서 특징 벡터(임베딩)를 추출하는 기능을 제공합니다.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import get_logger
from ...core.value_objects.face_embedding import FaceEmbedding

logger = get_logger(__name__)


class ArcFaceRecognizer:
    """ArcFace 기반 얼굴 인식기"""
    
    def __init__(self, 
                 model_path: str,
                 input_size: Tuple[int, int] = (112, 112),
                 use_gpu: bool = True):
        """ArcFace 인식기 초기화
        
        Args:
            model_path: 모델 파일 경로
            input_size: 입력 이미지 크기 (width, height)
            use_gpu: GPU 사용 여부
        """
        self.model_path = model_path
        self.input_size = input_size
        self.use_gpu = use_gpu
        self.net = None
        self.is_loaded = False
        
        logger.info(f"ArcFace recognizer initialized with model: {model_path}")
    
    def load_model(self) -> bool:
        """ArcFace 모델 로딩
        
        Returns:
            bool: 로딩 성공 여부
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # OpenCV DNN으로 모델 로딩
            if self.model_path.endswith('.onnx'):
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
            elif self.model_path.endswith('.pb'):
                self.net = cv2.dnn.readNetFromTensorflow(self.model_path)
            elif self.model_path.endswith('.caffemodel'):
                # Caffe 모델의 경우 prototxt 파일도 필요
                prototxt_path = self.model_path.replace('.caffemodel', '.prototxt')
                if not os.path.exists(prototxt_path):
                    logger.error(f"Prototxt file not found: {prototxt_path}")
                    return False
                self.net = cv2.dnn.readNetFromCaffe(prototxt_path, self.model_path)
            else:
                logger.error(f"Unsupported model format: {self.model_path}")
                return False
            
            # GPU 설정
            if self.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info("Using GPU backend for inference")
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logger.info("Using CPU backend for inference")
            
            self.is_loaded = True
            logger.info("ArcFace model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ArcFace model: {str(e)}")
            return False
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """얼굴 이미지 전처리
        
        Args:
            face_image: 입력 얼굴 이미지 (BGR)
            
        Returns:
            np.ndarray: 전처리된 이미지
        """
        # 크기 조정
        resized = cv2.resize(face_image, self.input_size)
        
        # 정규화 (0-255 -> -1~1)
        normalized = (resized.astype(np.float32) - 127.5) / 127.5
        
        # 차원 변경 (H, W, C) -> (1, C, H, W)
        blob = np.transpose(normalized, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        
        return blob
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[FaceEmbedding]:
        """단일 얼굴 이미지에서 임베딩 추출
        
        Args:
            face_image: 입력 얼굴 이미지 (BGR)
            
        Returns:
            Optional[FaceEmbedding]: 추출된 임베딩 (실패시 None)
        """
        if not self.is_loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        
        if not self._validate_face_image(face_image):
            logger.error("Invalid face image")
            return None
        
        try:
            # 이미지 전처리
            blob = self.preprocess_face(face_image)
            
            # 모델 추론
            self.net.setInput(blob)
            output = self.net.forward()
            
            # 임베딩 벡터 추출
            embedding_vector = output[0].flatten()
            
            # L2 정규화
            norm = np.linalg.norm(embedding_vector)
            if norm > 0:
                embedding_vector = embedding_vector / norm
            
            # FaceEmbedding 객체 생성
            embedding = FaceEmbedding(
                vector=embedding_vector,
                model_name="arcface"
            )
            
            logger.debug(f"Embedding extracted: dimension={embedding.dimension}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {str(e)}")
            return None
    
    def extract_embeddings_batch(self, face_images: List[np.ndarray]) -> List[Optional[FaceEmbedding]]:
        """배치 얼굴 이미지에서 임베딩 추출
        
        Args:
            face_images: 입력 얼굴 이미지 리스트
            
        Returns:
            List[Optional[FaceEmbedding]]: 추출된 임베딩 리스트
        """
        embeddings = []
        
        for i, face_image in enumerate(face_images):
            try:
                embedding = self.extract_embedding(face_image)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error extracting embedding for image {i}: {str(e)}")
                embeddings.append(None)
        
        logger.debug(f"Batch embedding extraction completed: {len(face_images)} images")
        return embeddings
    
    def compute_similarity(self, 
                         embedding1: FaceEmbedding, 
                         embedding2: FaceEmbedding,
                         metric: str = 'cosine') -> float:
        """두 임베딩 간의 유사도 계산
        
        Args:
            embedding1: 첫 번째 임베딩
            embedding2: 두 번째 임베딩
            metric: 유사도 메트릭 ('cosine', 'euclidean', 'dot')
            
        Returns:
            float: 유사도 점수
        """
        try:
            return embedding1.similarity_with(embedding2, metric)
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def identify_face(self, 
                     query_embedding: FaceEmbedding,
                     reference_embeddings: List[Tuple[str, FaceEmbedding]],
                     threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        """얼굴 식별 (1:N 매칭)
        
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
        
        if best_match:
            logger.debug(f"Face identified: {best_match[0]} (similarity: {best_match[1]:.3f})")
        else:
            logger.debug("No matching face found")
        
        return best_match
    
    def verify_face(self,
                   embedding1: FaceEmbedding,
                   embedding2: FaceEmbedding,
                   threshold: float = 0.6) -> Tuple[bool, float]:
        """얼굴 검증 (1:1 매칭)
        
        Args:
            embedding1: 첫 번째 임베딩
            embedding2: 두 번째 임베딩
            threshold: 유사도 임계값
            
        Returns:
            Tuple[bool, float]: (검증 결과, 유사도)
        """
        similarity = self.compute_similarity(embedding1, embedding2)
        is_same = similarity >= threshold
        
        logger.debug(f"Face verification: {'MATCH' if is_same else 'NO_MATCH'} (similarity: {similarity:.3f})")
        
        return is_same, similarity
    
    def _validate_face_image(self, face_image: np.ndarray) -> bool:
        """얼굴 이미지 유효성 검사
        
        Args:
            face_image: 검사할 얼굴 이미지
            
        Returns:
            bool: 이미지 유효성 여부
        """
        if face_image is None:
            return False
        
        if not isinstance(face_image, np.ndarray):
            return False
        
        if len(face_image.shape) != 3 or face_image.shape[2] != 3:
            return False
        
        if face_image.shape[0] < 32 or face_image.shape[1] < 32:
            logger.warning("Face image too small (minimum 32x32)")
            return False
        
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환
        
        Returns:
            Dict[str, Any]: 모델 정보
        """
        return {
            'model_path': self.model_path,
            'input_size': self.input_size,
            'use_gpu': self.use_gpu,
            'is_loaded': self.is_loaded,
            'model_type': 'ArcFace',
            'embedding_dimension': 512  # ArcFace 기본 차원
        }
    
    def set_gpu_usage(self, use_gpu: bool) -> None:
        """GPU 사용 설정 변경
        
        Args:
            use_gpu: GPU 사용 여부
        """
        if self.use_gpu != use_gpu:
            self.use_gpu = use_gpu
            if self.is_loaded:
                logger.info("GPU setting changed. Model needs to be reloaded.")
                self.is_loaded = False
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        if not self.is_loaded:
            self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        # 필요시 리소스 정리
        pass


def create_arcface_recognizer(model_path: str, 
                            input_size: Tuple[int, int] = (112, 112),
                            use_gpu: bool = True) -> ArcFaceRecognizer:
    """ArcFace 인식기 팩토리 함수
    
    Args:
        model_path: 모델 파일 경로
        input_size: 입력 이미지 크기
        use_gpu: GPU 사용 여부
        
    Returns:
        ArcFaceRecognizer: 생성된 인식기
    """
    recognizer = ArcFaceRecognizer(
        model_path=model_path,
        input_size=input_size,
        use_gpu=use_gpu
    )
    
    recognizer.load_model()
    
    return recognizer


# 모델별 사전 정의된 설정
ARCFACE_MODELS = {
    'arcface_r100_buffalo_l': {
        'input_size': (112, 112),
        'embedding_size': 512,
        'description': 'ArcFace ResNet100 - 높은 정확도'
    },
    'arcface_r50_buffalo_s': {
        'input_size': (112, 112), 
        'embedding_size': 512,
        'description': 'ArcFace ResNet50 - 균형잡힌 성능'
    },
    'mobilefacenet': {
        'input_size': (112, 112),
        'embedding_size': 128,
        'description': 'MobileFaceNet - 경량화 모델'
    }
} 