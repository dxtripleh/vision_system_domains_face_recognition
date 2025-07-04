#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Detector Module.

모든 검출기의 기본 클래스입니다.
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


class BaseDetector(ABC):
    """모든 검출기의 기본 클래스.
    
    이 클래스는 얼굴 검출, 객체 검출 등 모든 검출 기능의 기본 인터페이스를 제공합니다.
    크로스 플랫폼 호환성을 보장하며, 하드웨어 환경에 따른 최적화를 지원합니다.
    """
    
    def __init__(
        self, 
        model_path: Optional[Union[str, Path]] = None,
        device: str = 'auto',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        기본 검출기 초기화.
        
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
        
        # 성능 모니터링
        self.performance_stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'avg_inference_time': 0.0
        }
        
        logger.info(f"Base Detector 초기화 완료: {self.__class__.__name__}")
        logger.info(f"디바이스: {self.device}")
        logger.info(f"모델 경로: {self.model_path}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """기본 설정을 로드합니다."""
        try:
            config = load_config('detection')
            return config.get('base_detector', {})
        except Exception as e:
            logger.warning(f"기본 설정 로드 실패, 기본값 사용: {e}")
            return {
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4,
                'max_detections': 100,
                'input_size': (640, 640)
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
        return "base_detector.onnx"
    
    @abstractmethod
    def _load_model(self):
        """모델을 로드합니다. 하위 클래스에서 구현해야 합니다."""
        pass
    
    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리를 수행합니다. 하위 클래스에서 구현해야 합니다."""
        pass
    
    @abstractmethod
    def _postprocess(self, raw_output: Any, original_image: np.ndarray) -> List[Dict[str, Any]]:
        """모델 출력을 후처리합니다. 하위 클래스에서 구현해야 합니다."""
        pass
    
    def detect(
        self, 
        image: np.ndarray, 
        confidence_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        이미지에서 객체를 검출합니다.
        
        Args:
            image: 입력 이미지 (numpy 배열)
            confidence_threshold: 신뢰도 임계값 (None이면 설정값 사용)
            
        Returns:
            검출 결과 리스트. 각 결과는 다음 키를 포함:
            - bbox: [x1, y1, x2, y2] 바운딩 박스
            - confidence: 신뢰도 점수
            - class_id: 클래스 ID
            - class_name: 클래스 이름
            
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
            
            # 모델 추론
            raw_output = self._inference(processed_image)
            
            # 후처리
            detections = self._postprocess(raw_output, image)
            
            # 신뢰도 필터링
            threshold = confidence_threshold or self.config.get('confidence_threshold', 0.5)
            filtered_detections = [
                det for det in detections 
                if det.get('confidence', 0) >= threshold
            ]
            
            # 성능 통계 업데이트
            inference_time = time.time() - start_time
            self._update_performance_stats(inference_time)
            
            logger.debug(f"검출 완료: {len(filtered_detections)}개 객체, {inference_time:.3f}초")
            
            return filtered_detections
            
        except Exception as e:
            logger.error(f"검출 실패: {str(e)}")
            raise RuntimeError(f"검출 처리 중 오류 발생: {str(e)}")
    
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
    
    def _inference(self, processed_image: np.ndarray) -> Any:
        """모델 추론을 수행합니다."""
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다")
        
        try:
            # 하위 클래스에서 구현
            return self._run_inference(processed_image)
        except Exception as e:
            logger.error(f"모델 추론 실패: {str(e)}")
            raise RuntimeError(f"추론 중 오류 발생: {str(e)}")
    
    @abstractmethod
    def _run_inference(self, processed_image: np.ndarray) -> Any:
        """실제 모델 추론을 수행합니다. 하위 클래스에서 구현해야 합니다."""
        pass
    
    def _update_performance_stats(self, inference_time: float):
        """성능 통계를 업데이트합니다."""
        self.performance_stats['total_inferences'] += 1
        self.performance_stats['total_time'] += inference_time
        self.performance_stats['avg_inference_time'] = (
            self.performance_stats['total_time'] / self.performance_stats['total_inferences']
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계를 반환합니다."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """성능 통계를 초기화합니다."""
        self.performance_stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'avg_inference_time': 0.0
        }
        logger.info("성능 통계 초기화 완료")
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보를 반환합니다."""
        return {
            'model_path': str(self.model_path),
            'device': self.device,
            'config': self.config,
            'performance_stats': self.performance_stats
        }
    
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
            
            logger.info(f"{self.__class__.__name__} 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"리소스 정리 중 오류: {str(e)}")
    
    def _cleanup_model(self):
        """모델 리소스를 정리합니다. 하위 클래스에서 오버라이드 가능."""
        pass 