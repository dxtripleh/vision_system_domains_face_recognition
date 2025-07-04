#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Image Processor Module.

이미지 전처리 공통 모듈입니다.
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
import cv2

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import get_logger
from common.config import load_config
from common.utils import ValidationUtils

logger = get_logger(__name__)


class ImageProcessor:
    """이미지 전처리 클래스.
    
    이 클래스는 이미지 리사이징, 정규화, 증강 등 다양한 전처리 기능을 제공합니다.
    크로스 플랫폼 호환성을 보장하며, 다양한 이미지 형식을 지원합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        이미지 프로세서 초기화.
        
        Args:
            config: 설정 딕셔너리
        """
        # 설정 로드
        self.config = config or self._load_default_config()
        
        # 지원하는 이미지 형식
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # 성능 모니터링
        self.performance_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_processing_time': 0.0
        }
        
        logger.info(f"Image Processor 초기화 완료")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """기본 설정을 로드합니다."""
        try:
            config = load_config('preprocessing')
            return config.get('image_processor', {})
        except Exception as e:
            logger.warning(f"기본 설정 로드 실패, 기본값 사용: {e}")
            return {
                'default_size': (640, 640),
                'interpolation': cv2.INTER_LINEAR,
                'normalize_mean': [0.485, 0.456, 0.406],
                'normalize_std': [0.229, 0.224, 0.225],
                'enable_augmentation': False,
                'quality_threshold': 0.5
            }
    
    def load_image(
        self, 
        image_path: Union[str, Path]
    ) -> np.ndarray:
        """
        이미지를 로드합니다.
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            로드된 이미지 (numpy 배열)
            
        Raises:
            FileNotFoundError: 이미지 파일을 찾을 수 없는 경우
            ValueError: 지원하지 않는 이미지 형식인 경우
        """
        start_time = time.time()
        
        try:
            image_path = Path(image_path)
            
            # 파일 존재 확인
            if not image_path.exists():
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            
            # 파일 형식 확인
            if image_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"지원하지 않는 이미지 형식: {image_path.suffix}")
            
            # 이미지 로드
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            # BGR을 RGB로 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            logger.debug(f"이미지 로드 완료: {image_path}, {image.shape}, {processing_time:.3f}초")
            
            return image
            
        except Exception as e:
            logger.error(f"이미지 로드 실패: {str(e)}")
            raise
    
    def save_image(
        self, 
        image: np.ndarray, 
        output_path: Union[str, Path],
        quality: int = 95
    ) -> bool:
        """
        이미지를 저장합니다.
        
        Args:
            image: 저장할 이미지 (numpy 배열)
            output_path: 출력 파일 경로
            quality: 이미지 품질 (1-100)
            
        Returns:
            저장 성공 여부
        """
        start_time = time.time()
        
        try:
            output_path = Path(output_path)
            
            # 출력 디렉토리 생성
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # RGB를 BGR로 변환 (OpenCV 저장용)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            # 이미지 저장
            success = cv2.imwrite(str(output_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            if not success:
                raise RuntimeError(f"이미지 저장 실패: {output_path}")
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            logger.debug(f"이미지 저장 완료: {output_path}, {processing_time:.3f}초")
            
            return True
            
        except Exception as e:
            logger.error(f"이미지 저장 실패: {str(e)}")
            return False
    
    def resize_image(
        self, 
        image: np.ndarray, 
        target_size: Optional[Tuple[int, int]] = None,
        interpolation: Optional[int] = None
    ) -> np.ndarray:
        """
        이미지 크기를 조정합니다.
        
        Args:
            image: 입력 이미지
            target_size: 목표 크기 (width, height), None이면 기본값 사용
            interpolation: 보간법, None이면 기본값 사용
            
        Returns:
            리사이즈된 이미지
        """
        start_time = time.time()
        
        try:
            # 기본값 설정
            if target_size is None:
                target_size = self.config.get('default_size', (640, 640))
            
            if interpolation is None:
                interpolation = self.config.get('interpolation', cv2.INTER_LINEAR)
            
            # 리사이즈 수행
            resized_image = cv2.resize(image, target_size, interpolation=interpolation)
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            logger.debug(f"이미지 리사이즈 완료: {image.shape} -> {resized_image.shape}, {processing_time:.3f}초")
            
            return resized_image
            
        except Exception as e:
            logger.error(f"이미지 리사이즈 실패: {str(e)}")
            raise
    
    def normalize_image(
        self, 
        image: np.ndarray,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        이미지를 정규화합니다.
        
        Args:
            image: 입력 이미지 (0-255 범위)
            mean: 평균값 리스트, None이면 기본값 사용
            std: 표준편차 리스트, None이면 기본값 사용
            
        Returns:
            정규화된 이미지 (0-1 범위)
        """
        start_time = time.time()
        
        try:
            # 기본값 설정
            if mean is None:
                mean = self.config.get('normalize_mean', [0.485, 0.456, 0.406])
            
            if std is None:
                std = self.config.get('normalize_std', [0.229, 0.224, 0.225])
            
            # float32로 변환
            image_float = image.astype(np.float32) / 255.0
            
            # 정규화
            if len(image_float.shape) == 3 and image_float.shape[2] == 3:
                # RGB 이미지인 경우
                for i in range(3):
                    image_float[:, :, i] = (image_float[:, :, i] - mean[i]) / std[i]
            else:
                # 그레이스케일 이미지인 경우
                image_float = (image_float - mean[0]) / std[0]
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            logger.debug(f"이미지 정규화 완료: {processing_time:.3f}초")
            
            return image_float
            
        except Exception as e:
            logger.error(f"이미지 정규화 실패: {str(e)}")
            raise
    
    def denormalize_image(
        self, 
        image: np.ndarray,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        정규화된 이미지를 원래 범위로 되돌립니다.
        
        Args:
            image: 정규화된 이미지
            mean: 평균값 리스트, None이면 기본값 사용
            std: 표준편차 리스트, None이면 기본값 사용
            
        Returns:
            역정규화된 이미지 (0-255 범위)
        """
        start_time = time.time()
        
        try:
            # 기본값 설정
            if mean is None:
                mean = self.config.get('normalize_mean', [0.485, 0.456, 0.406])
            
            if std is None:
                std = self.config.get('normalize_std', [0.229, 0.224, 0.225])
            
            # 역정규화
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB 이미지인 경우
                for i in range(3):
                    image[:, :, i] = image[:, :, i] * std[i] + mean[i]
            else:
                # 그레이스케일 이미지인 경우
                image = image * std[0] + mean[0]
            
            # 0-255 범위로 변환
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            logger.debug(f"이미지 역정규화 완료: {processing_time:.3f}초")
            
            return image
            
        except Exception as e:
            logger.error(f"이미지 역정규화 실패: {str(e)}")
            raise
    
    def crop_image(
        self, 
        image: np.ndarray, 
        bbox: List[int]
    ) -> np.ndarray:
        """
        이미지를 크롭합니다.
        
        Args:
            image: 입력 이미지
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            
        Returns:
            크롭된 이미지
        """
        start_time = time.time()
        
        try:
            x1, y1, x2, y2 = bbox
            
            # 경계 확인
            height, width = image.shape[:2]
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # 크롭 수행
            cropped_image = image[y1:y2, x1:x2]
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            logger.debug(f"이미지 크롭 완료: {bbox} -> {cropped_image.shape}, {processing_time:.3f}초")
            
            return cropped_image
            
        except Exception as e:
            logger.error(f"이미지 크롭 실패: {str(e)}")
            raise
    
    def flip_image(
        self, 
        image: np.ndarray, 
        direction: str = 'horizontal'
    ) -> np.ndarray:
        """
        이미지를 뒤집습니다.
        
        Args:
            image: 입력 이미지
            direction: 뒤집기 방향 ('horizontal', 'vertical', 'both')
            
        Returns:
            뒤집힌 이미지
        """
        start_time = time.time()
        
        try:
            if direction == 'horizontal':
                flipped_image = cv2.flip(image, 1)
            elif direction == 'vertical':
                flipped_image = cv2.flip(image, 0)
            elif direction == 'both':
                flipped_image = cv2.flip(image, -1)
            else:
                raise ValueError(f"지원하지 않는 뒤집기 방향: {direction}")
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            logger.debug(f"이미지 뒤집기 완료: {direction}, {processing_time:.3f}초")
            
            return flipped_image
            
        except Exception as e:
            logger.error(f"이미지 뒤집기 실패: {str(e)}")
            raise
    
    def rotate_image(
        self, 
        image: np.ndarray, 
        angle: float,
        center: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        이미지를 회전합니다.
        
        Args:
            image: 입력 이미지
            angle: 회전 각도 (도)
            center: 회전 중심점, None이면 이미지 중심
            
        Returns:
            회전된 이미지
        """
        start_time = time.time()
        
        try:
            height, width = image.shape[:2]
            
            # 회전 중심점 설정
            if center is None:
                center = (width // 2, height // 2)
            
            # 회전 행렬 계산
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 회전 수행
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            logger.debug(f"이미지 회전 완료: {angle}도, {processing_time:.3f}초")
            
            return rotated_image
            
        except Exception as e:
            logger.error(f"이미지 회전 실패: {str(e)}")
            raise
    
    def adjust_brightness(
        self, 
        image: np.ndarray, 
        factor: float
    ) -> np.ndarray:
        """
        이미지 밝기를 조정합니다.
        
        Args:
            image: 입력 이미지
            factor: 밝기 조정 계수 (0.0-2.0)
            
        Returns:
            밝기가 조정된 이미지
        """
        start_time = time.time()
        
        try:
            # 밝기 조정
            adjusted_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            logger.debug(f"밝기 조정 완료: factor={factor}, {processing_time:.3f}초")
            
            return adjusted_image
            
        except Exception as e:
            logger.error(f"밝기 조정 실패: {str(e)}")
            raise
    
    def adjust_contrast(
        self, 
        image: np.ndarray, 
        factor: float
    ) -> np.ndarray:
        """
        이미지 대비를 조정합니다.
        
        Args:
            image: 입력 이미지
            factor: 대비 조정 계수 (0.0-3.0)
            
        Returns:
            대비가 조정된 이미지
        """
        start_time = time.time()
        
        try:
            # 대비 조정
            adjusted_image = cv2.convertScaleAbs(image, alpha=factor, beta=128*(1-factor))
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            logger.debug(f"대비 조정 완료: factor={factor}, {processing_time:.3f}초")
            
            return adjusted_image
            
        except Exception as e:
            logger.error(f"대비 조정 실패: {str(e)}")
            raise
    
    def apply_gaussian_blur(
        self, 
        image: np.ndarray, 
        kernel_size: int = 5,
        sigma: float = 1.0
    ) -> np.ndarray:
        """
        가우시안 블러를 적용합니다.
        
        Args:
            image: 입력 이미지
            kernel_size: 커널 크기 (홀수)
            sigma: 표준편차
            
        Returns:
            블러가 적용된 이미지
        """
        start_time = time.time()
        
        try:
            # 커널 크기가 홀수인지 확인
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # 가우시안 블러 적용
            blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            logger.debug(f"가우시안 블러 완료: kernel={kernel_size}, sigma={sigma}, {processing_time:.3f}초")
            
            return blurred_image
            
        except Exception as e:
            logger.error(f"가우시안 블러 실패: {str(e)}")
            raise
    
    def validate_image_quality(
        self, 
        image: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        이미지 품질을 검증합니다.
        
        Args:
            image: 입력 이미지
            threshold: 품질 임계값, None이면 기본값 사용
            
        Returns:
            품질 검증 결과
        """
        start_time = time.time()
        
        try:
            if threshold is None:
                threshold = self.config.get('quality_threshold', 0.5)
            
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # 라플라시안 분산 계산 (선명도 측정)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 품질 점수 계산 (0-1 범위)
            quality_score = min(laplacian_var / 1000.0, 1.0)
            
            # 품질 판정
            is_good_quality = quality_score >= threshold
            
            result = {
                'quality_score': quality_score,
                'is_good_quality': is_good_quality,
                'laplacian_variance': laplacian_var,
                'threshold': threshold
            }
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            logger.debug(f"품질 검증 완료: score={quality_score:.3f}, {processing_time:.3f}초")
            
            return result
            
        except Exception as e:
            logger.error(f"품질 검증 실패: {str(e)}")
            raise
    
    def _update_performance_stats(self, processing_time: float):
        """성능 통계를 업데이트합니다."""
        self.performance_stats['total_processed'] += 1
        self.performance_stats['total_time'] += processing_time
        self.performance_stats['avg_processing_time'] = (
            self.performance_stats['total_time'] / self.performance_stats['total_processed']
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계를 반환합니다."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """성능 통계를 초기화합니다."""
        self.performance_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_processing_time': 0.0
        }
        logger.info("성능 통계 초기화 완료")
    
    def get_supported_formats(self) -> List[str]:
        """지원하는 이미지 형식을 반환합니다."""
        return self.supported_formats.copy()
    
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """파일 형식이 지원되는지 확인합니다."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_formats 