#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common Utilities.

범용 유틸리티 함수들입니다.
크로스 플랫폼 호환성을 고려하여 설계되었습니다.
"""

import os
import sys
import platform
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

class HardwareDetector:
    """하드웨어 감지 클래스."""
    
    def __init__(self):
        """하드웨어 감지기 초기화."""
        self.system_info = self._get_system_info()
        self.hardware_info = self._get_hardware_info()
    
    def _get_system_info(self) -> Dict[str, str]:
        """시스템 정보를 가져옵니다."""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
    
    def _get_hardware_info(self) -> Dict[str, Union[str, int, float]]:
        """하드웨어 정보를 가져옵니다."""
        # CPU 정보
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        
        # 디스크 정보
        disk = psutil.disk_usage('/')
        
        # GPU 정보 (기본값)
        gpu_info = {
            'available': False,
            'name': 'Unknown',
            'memory_total': 0,
            'memory_used': 0
        }
        
        # GPU 감지 시도
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 첫 번째 GPU
                gpu_info = {
                    'available': True,
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed
                }
        except ImportError:
            logger.debug("GPUtil이 설치되지 않아 GPU 정보를 가져올 수 없습니다.")
        except Exception as e:
            logger.debug(f"GPU 정보 가져오기 실패: {e}")
        
        return {
            'cpu_count': cpu_count,
            'cpu_freq_mhz': cpu_freq.current if cpu_freq else 0,
            'cpu_percent': cpu_percent,
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'disk_total_gb': disk.total / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'disk_percent': disk.percent,
            'gpu': gpu_info
        }
    
    def is_jetson_platform(self) -> bool:
        """Jetson 플랫폼인지 확인합니다."""
        try:
            with open("/proc/device-tree/model", "r") as f:
                return "jetson" in f.read().lower()
        except:
            return False
    
    def is_gpu_available(self) -> bool:
        """GPU 사용 가능 여부를 확인합니다."""
        return self.hardware_info['gpu']['available']
    
    def get_optimal_device(self) -> str:
        """최적의 실행 디바이스를 결정합니다."""
        if self.is_jetson_platform():
            return 'jetson'
        elif self.is_gpu_available():
            return 'gpu'
        else:
            return 'cpu'
    
    def get_system_info(self) -> Dict[str, str]:
        """시스템 정보를 반환합니다."""
        return self.system_info.copy()
    
    def get_hardware_info(self) -> Dict[str, Union[str, int, float]]:
        """하드웨어 정보를 반환합니다."""
        return self.hardware_info.copy()
    
    def get_performance_summary(self) -> Dict[str, str]:
        """성능 요약을 반환합니다."""
        device = self.get_optimal_device()
        cpu_count = self.hardware_info['cpu_count']
        memory_gb = self.hardware_info['memory_total_gb']
        
        return {
            'optimal_device': device,
            'cpu_cores': str(cpu_count),
            'memory_gb': f"{memory_gb:.1f}",
            'platform': self.system_info['platform'],
            'python_version': self.system_info['python_version']
        }

class PathUtils:
    """경로 유틸리티 클래스."""
    
    @staticmethod
    def get_project_root() -> Path:
        """프로젝트 루트 경로를 반환합니다."""
        current_dir = Path(__file__).parent
        return current_dir.parent.parent
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """디렉토리가 존재하지 않으면 생성합니다."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_platform_path(path: str) -> str:
        """플랫폼별 경로를 반환합니다."""
        if platform.system().lower() == "windows":
            return path.replace('/', '\\')
        else:
            return path.replace('\\', '/')
    
    @staticmethod
    def find_files(directory: Union[str, Path], pattern: str) -> List[Path]:
        """패턴에 맞는 파일들을 찾습니다."""
        directory = Path(directory)
        return list(directory.glob(pattern))
    
    @staticmethod
    def get_file_size_mb(file_path: Union[str, Path]) -> float:
        """파일 크기를 MB 단위로 반환합니다."""
        file_path = Path(file_path)
        if file_path.exists():
            return file_path.stat().st_size / (1024 * 1024)
        return 0.0

class TimeUtils:
    """시간 유틸리티 클래스."""
    
    @staticmethod
    def get_timestamp() -> str:
        """현재 타임스탬프를 반환합니다."""
        from datetime import datetime
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """초를 사람이 읽기 쉬운 형태로 변환합니다."""
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}분"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}시간"
    
    @staticmethod
    def get_elapsed_time(start_time: float) -> float:
        """경과 시간을 계산합니다."""
        import time
        return time.time() - start_time

class ValidationUtils:
    """검증 유틸리티 클래스."""
    
    @staticmethod
    def validate_file_exists(file_path: Union[str, Path]) -> bool:
        """파일 존재 여부를 검증합니다."""
        return Path(file_path).exists()
    
    @staticmethod
    def validate_directory_exists(dir_path: Union[str, Path]) -> bool:
        """디렉토리 존재 여부를 검증합니다."""
        return Path(dir_path).exists() and Path(dir_path).is_dir()
    
    @staticmethod
    def validate_image_file(file_path: Union[str, Path]) -> bool:
        """이미지 파일 여부를 검증합니다."""
        file_path = Path(file_path)
        if not file_path.exists():
            return False
        
        # 이미지 확장자 확인
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        return file_path.suffix.lower() in image_extensions
    
    @staticmethod
    def validate_model_file(file_path: Union[str, Path]) -> bool:
        """모델 파일 여부를 검증합니다."""
        file_path = Path(file_path)
        if not file_path.exists():
            return False
        
        # 모델 확장자 확인
        model_extensions = {'.onnx', '.pt', '.pth', '.pb', '.h5'}
        return file_path.suffix.lower() in model_extensions

# 전역 인스턴스
hardware_detector = HardwareDetector()
path_utils = PathUtils()
time_utils = TimeUtils()
validation_utils = ValidationUtils()

# 편의 함수들
def get_hardware_info() -> Dict[str, Union[str, int, float]]:
    """하드웨어 정보를 반환합니다."""
    return hardware_detector.get_hardware_info()

def get_optimal_device() -> str:
    """최적의 실행 디바이스를 반환합니다."""
    return hardware_detector.get_optimal_device()

def get_project_root() -> Path:
    """프로젝트 루트 경로를 반환합니다."""
    return path_utils.get_project_root()

def ensure_dir(path: Union[str, Path]) -> Path:
    """디렉토리가 존재하지 않으면 생성합니다."""
    return path_utils.ensure_dir(path)

def get_timestamp() -> str:
    """현재 타임스탬프를 반환합니다."""
    return time_utils.get_timestamp()

def validate_file_exists(file_path: Union[str, Path]) -> bool:
    """파일 존재 여부를 검증합니다."""
    return validation_utils.validate_file_exists(file_path)
