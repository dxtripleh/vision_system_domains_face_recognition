#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Recognizer Module.

모든 인식기의 기본 클래스입니다.
크로스 플랫폼 호환성을 고려하여 설계되었습니다.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
import numpy as np

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import get_logger
from common.config import load_config
from common.utils import HardwareDetector, ValidationUtils

logger = get_logger(__name__)


class BaseRecognizer(ABC):
    """모든 인식기의 기본 클래스.
    
    이 클래스는 얼굴 인식, 객체 인식 등 모든 인식 기능의 기본 인터페이스를 제공합니다.
    크로스 플랫폼 호환성을 보장하며, 하드웨어 환경에 따른 최적화를 지원합니다.
    """
    
    def __init__(
        self, 
        model_path: Optional[Union[str, Path]] = None,
        device: str = 'auto',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        기본 인식기 초기화.
        
        Args:
            model_path: 모델 파일 경로 (None이면 자동 감지)
            device: 실행 디바이스 ('cpu', 'gpu', 'auto')
            config: 설정 딕셔너리
            
        Raises:
            FileNotFoundError: 모델 파일을 찾을 수 없는 경우
            RuntimeError: 하드웨어 연결 실패 시
        """
        # 하드웨어 감지기 초기화
        self.hardware_detector = HardwareDetector()
        
        # 설정 로드
        self.config = config or self._load_default_config()
        
        # 디바이스 설정
        self.device = self._get_optimal_device(device)
        
        # 모델 경로 설정
        self.model_path = self._resolve_model_path(model_path)
        
        # 모델 로드
        self.model = None
        self._load_model()
        
        # 임베딩 데이터베이스
        self.embedding_db = {}
        
        # 성능 모니터링
        self.performance_stats = {
            'total_recognitions': 0,
            'total_time': 0.0,
            'avg_recognition_time': 0.0,
            'total_embeddings': 0
        }
        
        logger.info(f"Base Recognizer 초기화 완료: {self.__class__.__name__}")
        logger.info(f"디바이스: {self.device}")
        logger.info(f"모델 경로: {self.model_path}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """기본 설정을 로드합니다."""
        try:
            config = load_config('recognition')
            return config.get('base_recognizer', {})
        except Exception as e:
            logger.warning(f"기본 설정 로드 실패, 기본값 사용: {e}")
            return {
                'similarity_threshold': 0.6,
                'embedding_dim': 512,
                'normalize_embeddings': True,
                'input_size': (112, 112),
                'max_embeddings_per_identity': 10
            }
    
    def _get_optimal_device(self, device: str) -> str:
        """최적의 실행 디바이스를 결정합니다."""
        if device == 'auto':
            return self.hardware_detector.get_optimal_device()
        elif device in ['cpu', 'gpu', 'jetson']:
            return device
        else:
            logger.warning(f"알 수 없는 디바이스: {device}, CPU 사용")
            return 'cpu'
    
    def _resolve_model_path(self, model_path: Optional[Union[str, Path]]) -> Path:
        """모델 경로를 해결합니다."""
        if model_path is None:
            # 기본 모델 경로 자동 감지
            models_dir = project_root / "models" / "weights"
            model_name = self._get_default_model_name()
            model_path = models_dir / model_name
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"기본 모델 파일을 찾을 수 없습니다: {model_path}\n"
                    f"모델을 다운로드하거나 올바른 경로를 지정하세요."
                )
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        return model_path
    
    def _get_default_model_name(self) -> str:
        """기본 모델 파일명을 반환합니다."""
        # 하위 클래스에서 오버라이드
        return "base_recognizer.onnx"
    
    @abstractmethod
    def _load_model(self):
        """모델을 로드합니다. 하위 클래스에서 구현해야 합니다."""
        pass
    
    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리를 수행합니다. 하위 클래스에서 구현해야 합니다."""
        pass
    
    @abstractmethod
    def _extract_embedding(self, processed_image: np.ndarray) -> np.ndarray:
        """특징 벡터(임베딩)를 추출합니다. 하위 클래스에서 구현해야 합니다."""
        pass
    
    def extract_embedding(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        이미지에서 특징 벡터를 추출합니다.
        
        Args:
            image: 입력 이미지 (numpy 배열)
            
        Returns:
            특징 벡터 (numpy 배열)
            
        Raises:
            ValueError: 입력 이미지가 유효하지 않은 경우
            RuntimeError: 모델 추론 실패 시
        """
        start_time = time.time()
        
        try:
            # 입력 검증
            if not self._validate_input(image):
                raise ValueError("입력 이미지가 유효하지 않습니다")
            
            # 전처리
            processed_image = self._preprocess(image)
            
            # 임베딩 추출
            embedding = self._extract_embedding(processed_image)
            
            # 정규화 (설정에 따라)
            if self.config.get('normalize_embeddings', True):
                embedding = self._normalize_embedding(embedding)
            
            # 성능 통계 업데이트
            inference_time = time.time() - start_time
            self._update_performance_stats(inference_time, embedding_extraction=True)
            
            logger.debug(f"임베딩 추출 완료: {embedding.shape}, {inference_time:.3f}초")
            
            return embedding
            
        except Exception as e:
            logger.error(f"임베딩 추출 실패: {str(e)}")
            raise RuntimeError(f"임베딩 추출 중 오류 발생: {str(e)}")
    
    def recognize(
        self, 
        image: np.ndarray,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        이미지를 인식합니다.
        
        Args:
            image: 입력 이미지 (numpy 배열)
            similarity_threshold: 유사도 임계값 (None이면 설정값 사용)
            
        Returns:
            인식 결과 딕셔너리:
            - identity: 인식된 신원 (없으면 None)
            - confidence: 신뢰도 점수
            - similarity: 최고 유사도 점수
            - embedding: 추출된 임베딩
            
        Raises:
            ValueError: 입력 이미지가 유효하지 않은 경우
            RuntimeError: 인식 실패 시
        """
        start_time = time.time()
        
        try:
            # 임베딩 추출
            embedding = self.extract_embedding(image)
            
            # 데이터베이스에서 매칭
            threshold = similarity_threshold or self.config.get('similarity_threshold', 0.6)
            match_result = self._find_best_match(embedding, threshold)
            
            # 성능 통계 업데이트
            inference_time = time.time() - start_time
            self._update_performance_stats(inference_time)
            
            result = {
                'identity': match_result.get('identity'),
                'confidence': match_result.get('confidence', 0.0),
                'similarity': match_result.get('similarity', 0.0),
                'embedding': embedding
            }
            
            logger.debug(f"인식 완료: {result['identity']}, {result['similarity']:.3f}, {inference_time:.3f}초")
            
            return result
            
        except Exception as e:
            logger.error(f"인식 실패: {str(e)}")
            raise RuntimeError(f"인식 처리 중 오류 발생: {str(e)}")
    
    def add_identity(
        self, 
        identity: str, 
        images: List[np.ndarray]
    ) -> bool:
        """
        새로운 신원을 데이터베이스에 추가합니다.
        
        Args:
            identity: 신원 식별자
            images: 해당 신원의 이미지 리스트
            
        Returns:
            추가 성공 여부
            
        Raises:
            ValueError: 입력이 유효하지 않은 경우
        """
        try:
            if not identity or not images:
                raise ValueError("신원과 이미지가 필요합니다")
            
            embeddings = []
            max_embeddings = self.config.get('max_embeddings_per_identity', 10)
            
            for image in images[:max_embeddings]:
                embedding = self.extract_embedding(image)
                embeddings.append(embedding)
            
            # 평균 임베딩 계산
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                self.embedding_db[identity] = {
                    'embedding': avg_embedding,
                    'count': len(embeddings),
                    'added_time': time.time()
                }
                
                logger.info(f"신원 추가 완료: {identity}, 임베딩 {len(embeddings)}개")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"신원 추가 실패: {str(e)}")
            return False
    
    def remove_identity(self, identity: str) -> bool:
        """
        신원을 데이터베이스에서 제거합니다.
        
        Args:
            identity: 제거할 신원 식별자
            
        Returns:
            제거 성공 여부
        """
        try:
            if identity in self.embedding_db:
                del self.embedding_db[identity]
                logger.info(f"신원 제거 완료: {identity}")
                return True
            else:
                logger.warning(f"존재하지 않는 신원: {identity}")
                return False
                
        except Exception as e:
            logger.error(f"신원 제거 실패: {str(e)}")
            return False
    
    def get_identities(self) -> List[str]:
        """등록된 모든 신원을 반환합니다."""
        return list(self.embedding_db.keys())
    
    def get_database_info(self) -> Dict[str, Any]:
        """데이터베이스 정보를 반환합니다."""
        return {
            'total_identities': len(self.embedding_db),
            'identities': list(self.embedding_db.keys()),
            'total_embeddings': sum(info['count'] for info in self.embedding_db.values())
        }
    
    def _validate_input(self, image: np.ndarray) -> bool:
        """입력 이미지를 검증합니다."""
        if image is None:
            return False
        
        if not isinstance(image, np.ndarray):
            return False
        
        if len(image.shape) != 3:
            return False
        
        if image.shape[2] not in [1, 3]:  # 그레이스케일 또는 RGB
            return False
        
        return True
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """임베딩을 정규화합니다."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def _find_best_match(
        self, 
        query_embedding: np.ndarray, 
        threshold: float
    ) -> Dict[str, Any]:
        """가장 유사한 신원을 찾습니다."""
        if not self.embedding_db:
            return {'identity': None, 'confidence': 0.0, 'similarity': 0.0}
        
        best_match = {'identity': None, 'confidence': 0.0, 'similarity': 0.0}
        
        for identity, info in self.embedding_db.items():
            db_embedding = info['embedding']
            
            # 코사인 유사도 계산
            similarity = self._calculate_similarity(query_embedding, db_embedding)
            
            if similarity > best_match['similarity'] and similarity >= threshold:
                best_match = {
                    'identity': identity,
                    'confidence': similarity,
                    'similarity': similarity
                }
        
        return best_match
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """두 임베딩 간의 유사도를 계산합니다."""
        # 코사인 유사도
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        return 0.0
    
    def _update_performance_stats(self, inference_time: float, embedding_extraction: bool = False):
        """성능 통계를 업데이트합니다."""
        if embedding_extraction:
            self.performance_stats['total_embeddings'] += 1
        else:
            self.performance_stats['total_recognitions'] += 1
            self.performance_stats['total_time'] += inference_time
            self.performance_stats['avg_recognition_time'] = (
                self.performance_stats['total_time'] / self.performance_stats['total_recognitions']
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계를 반환합니다."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """성능 통계를 초기화합니다."""
        self.performance_stats = {
            'total_recognitions': 0,
            'total_time': 0.0,
            'avg_recognition_time': 0.0,
            'total_embeddings': 0
        }
        logger.info("성능 통계 초기화 완료")
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보를 반환합니다."""
        return {
            'model_path': str(self.model_path),
            'device': self.device,
            'config': self.config,
            'performance_stats': self.performance_stats,
            'database_info': self.get_database_info()
        }
    
    def save_database(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """임베딩 데이터베이스를 파일로 저장합니다."""
        try:
            if file_path is None:
                file_path = project_root / "data" / "domains" / "humanoid" / "face_recognition" / "embeddings_db.pkl"
            
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(self.embedding_db, f)
            
            logger.info(f"데이터베이스 저장 완료: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"데이터베이스 저장 실패: {str(e)}")
            return False
    
    def load_database(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """임베딩 데이터베이스를 파일에서 로드합니다."""
        try:
            if file_path is None:
                file_path = project_root / "data" / "domains" / "humanoid" / "face_recognition" / "embeddings_db.pkl"
            
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning(f"데이터베이스 파일이 존재하지 않습니다: {file_path}")
                return False
            
            import pickle
            with open(file_path, 'rb') as f:
                self.embedding_db = pickle.load(f)
            
            logger.info(f"데이터베이스 로드 완료: {file_path}, {len(self.embedding_db)}개 신원")
            return True
            
        except Exception as e:
            logger.error(f"데이터베이스 로드 실패: {str(e)}")
            return False
    
    def __enter__(self):
        """컨텍스트 매니저 진입."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료."""
        self.cleanup()
    
    def cleanup(self):
        """리소스를 정리합니다."""
        try:
            if hasattr(self, 'model') and self.model is not None:
                # 모델별 정리 로직 (하위 클래스에서 오버라이드 가능)
                self._cleanup_model()
                self.model = None
            
            # 데이터베이스 자동 저장
            self.save_database()
            
            logger.info(f"{self.__class__.__name__} 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"리소스 정리 중 오류: {str(e)}")
    
    def _cleanup_model(self):
        """모델 리소스를 정리합니다. 하위 클래스에서 오버라이드 가능."""
        pass 