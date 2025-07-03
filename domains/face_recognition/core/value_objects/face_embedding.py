#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Embedding Value Object.

얼굴 임베딩 벡터를 나타내는 값 객체입니다.
"""

import numpy as np
from typing import Dict, Any, List, Union
from dataclasses import dataclass
import hashlib
import base64


@dataclass(frozen=True)
class FaceEmbedding:
    """
    얼굴 임베딩 값 객체
    
    얼굴의 특징을 나타내는 고차원 벡터를 안전하게 관리합니다.
    """
    
    vector: np.ndarray
    model_name: str = "unknown"
    
    def __post_init__(self):
        """초기화 후 검증"""
        if not isinstance(self.vector, np.ndarray):
            raise TypeError(f"Embedding vector must be numpy array, got {type(self.vector)}")
        
        if self.vector.ndim != 1:
            raise ValueError(f"Embedding must be 1-dimensional, got shape {self.vector.shape}")
        
        if self.vector.size == 0:
            raise ValueError("Embedding vector cannot be empty")
        
        if not np.isfinite(self.vector).all():
            raise ValueError("Embedding vector must contain only finite values")
        
        # numpy array를 immutable하게 만들기 (frozen dataclass에서는 설정하지 않음)
        # self.vector.flags.writeable = False
    
    @property
    def dimension(self) -> int:
        """임베딩 벡터의 차원"""
        return self.vector.size
    
    @property
    def norm(self) -> float:
        """L2 노름 (벡터의 크기)"""
        return float(np.linalg.norm(self.vector))
    
    @property
    def is_normalized(self) -> bool:
        """정규화된 벡터인지 확인 (L2 norm ≈ 1.0)"""
        return abs(self.norm - 1.0) < 1e-6
    
    @property
    def hash(self) -> str:
        """임베딩의 해시값 (익명화용)"""
        vector_bytes = self.vector.tobytes()
        hash_obj = hashlib.sha256(vector_bytes)
        return hash_obj.hexdigest()
    
    def normalize(self) -> 'FaceEmbedding':
        """
        임베딩을 L2 정규화
        
        Returns:
            FaceEmbedding: 정규화된 임베딩
        """
        norm = self.norm
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")
        
        normalized_vector = self.vector / norm
        return FaceEmbedding(vector=normalized_vector, model_name=self.model_name)
    
    def similarity_with(self, other: 'FaceEmbedding', metric: str = 'cosine') -> float:
        """
        다른 임베딩과의 유사도 계산
        
        Args:
            other: 비교할 임베딩
            metric: 유사도 메트릭 ('cosine', 'euclidean', 'dot')
            
        Returns:
            float: 유사도 값
        """
        if self.dimension != other.dimension:
            raise ValueError(f"Dimension mismatch: {self.dimension} vs {other.dimension}")
        
        if metric == 'cosine':
            # 코사인 유사도
            dot_product = np.dot(self.vector, other.vector)
            norm_product = self.norm * other.norm
            if norm_product == 0:
                return 0.0
            return float(dot_product / norm_product)
        
        elif metric == 'euclidean':
            # 유클리드 거리 (거리이므로 유사도로 변환)
            distance = np.linalg.norm(self.vector - other.vector)
            return float(1.0 / (1.0 + distance))
        
        elif metric == 'dot':
            # 내적 (정규화된 벡터라면 코사인 유사도와 동일)
            return float(np.dot(self.vector, other.vector))
        
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def is_similar_to(self, other: 'FaceEmbedding', threshold: float = 0.8) -> bool:
        """
        다른 임베딩과 유사한지 확인
        
        Args:
            other: 비교할 임베딩
            threshold: 유사도 임계값
            
        Returns:
            bool: 임계값 이상이면 True
        """
        similarity = self.similarity_with(other, 'cosine')
        return similarity >= threshold
    
    def to_list(self) -> List[float]:
        """
        리스트 형식으로 변환
        
        Returns:
            List[float]: 임베딩 벡터의 리스트
        """
        return self.vector.tolist()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        딕셔너리 형식으로 변환
        
        Returns:
            Dict[str, Any]: 임베딩 정보
        """
        return {
            'vector': self.to_list(),
            'model_name': self.model_name,
            'dimension': self.dimension,
            'norm': self.norm,
            'is_normalized': self.is_normalized,
            'hash': self.hash
        }
    
    @classmethod
    def from_list(cls, vector_list: List[float], model_name: str = "unknown") -> 'FaceEmbedding':
        """
        리스트에서 임베딩 생성
        
        Args:
            vector_list: 임베딩 벡터 리스트
            model_name: 모델명
            
        Returns:
            FaceEmbedding: 생성된 임베딩
        """
        vector = np.array(vector_list, dtype=np.float32)
        return cls(vector=vector, model_name=model_name)
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"FaceEmbedding(dim={self.dimension}, model={self.model_name})"
    
    def __repr__(self) -> str:
        """개발자용 문자열 표현"""
        return f"FaceEmbedding(vector=array({self.dimension}), model_name='{self.model_name}', norm={self.norm:.3f})"
