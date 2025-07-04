#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Humanoid Face Recognition Model.

ONNX 기반 얼굴인식 모델 클래스입니다.
크로스 플랫폼 및 Jetson/CPU/GPU 자동 최적화 지원.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

try:
    import onnxruntime as ort
except ImportError:
    logging.error("ONNX Runtime이 설치되지 않았습니다. pip install onnxruntime을 실행하세요.")
    raise

logger = logging.getLogger(__name__)

class FaceRecognitionModel:
    """얼굴인식 모델 클래스."""
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.model_path = model_path or self._get_default_model_path()
        self.device_config = self._detect_hardware_environment()
        logger.info(f"하드웨어 환경: {self.device_config}")
        self.session = self._initialize_onnx_session()
        self.input_shape = self._get_input_shape()
        self.output_names = [output.name for output in self.session.get_outputs()] if self.session else []
        logger.info(f"모델 로딩 완료: {self.model_path}")
        logger.info(f"입력 형태: {self.input_shape}")
        logger.info(f"출력 이름: {self.output_names}")
        logger.info(f"더미 모델 모드: {self.session is None}")

    def _get_default_config(self) -> Dict:
        return {
            'input_size': (112, 112),
            'normalize': True
        }

    def _get_default_model_path(self) -> str:
        model_path = project_root / "models" / "weights" / "face_recognition_arcface_glint360k_20250628.onnx"
        return str(model_path)

    def _detect_hardware_environment(self) -> Dict:
        import platform
        import psutil
        system = platform.system().lower()
        is_jetson = self._is_jetson_platform()
        gpu_available = False
        try:
            providers = ort.get_available_providers()
            gpu_available = 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers
        except Exception as e:
            logger.warning(f"GPU 정보 확인 실패: {e}")
        if is_jetson:
            return {
                'device': 'jetson',
                'providers': ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            }
        elif gpu_available:
            return {
                'device': 'gpu',
                'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider']
            }
        else:
            return {
                'device': 'cpu',
                'providers': ['CPUExecutionProvider']
            }

    def _is_jetson_platform(self) -> bool:
        try:
            with open("/proc/device-tree/model", "r") as f:
                return "jetson" in f.read().lower()
        except:
            return False

    def _initialize_onnx_session(self) -> Optional[ort.InferenceSession]:
        if not os.path.exists(self.model_path):
            logger.warning(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
            return None
        
        try:
            session = ort.InferenceSession(
                self.model_path,
                providers=self.device_config['providers']
            )
            return session
        except Exception as e:
            logger.warning(f"ONNX 모델 로딩 실패 (더미 모델일 가능성): {e}")
            return None

    def _get_input_shape(self) -> Tuple[int, int, int]:
        if self.session is None:
            return (1, 3, 112, 112)  # 더미 모델용 기본값
        
        input_info = self.session.get_inputs()[0]
        shape = input_info.shape
        if shape[0] == -1:
            shape = (1,) + tuple(shape[1:])
        return tuple(shape)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        input_size = self.config['input_size']
        resized = cv2.resize(image, input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0 if self.config['normalize'] else rgb.astype(np.float32)
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor

    def infer(self, image: np.ndarray) -> np.ndarray:
        input_tensor = self.preprocess(image)
        
        if self.session is None:
            # 더미 모델인 경우 가짜 임베딩 반환
            return np.random.normal(0, 1, (1, 512)).astype(np.float32)
        
        outputs = self.session.run(self.output_names, {self.session.get_inputs()[0].name: input_tensor})
        return outputs[0]

    def get_model_info(self) -> Dict:
        return {
            'model_path': self.model_path,
            'input_shape': self.input_shape,
            'output_names': self.output_names,
            'device_config': self.device_config,
            'config': self.config,
            'is_dummy': self.session is None
        } 