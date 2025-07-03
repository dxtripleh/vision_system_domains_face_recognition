#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
설정 파일 로더 모듈.

이 모듈은 YAML 기반 설정 파일을 로드하고 관리하는 기능을 제공합니다.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    YAML 설정 파일을 로드합니다.
    
    Args:
        config_path: 설정 파일 경로. None이면 기본 설정 파일 사용
        
    Returns:
        설정 딕셔너리
        
    Raises:
        FileNotFoundError: 설정 파일을 찾을 수 없는 경우
        yaml.YAMLError: YAML 파싱 오류
    """
    if config_path is None:
        # 기본 설정 파일 경로
        config_path = 'config/face_recognition_config.yaml'
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}, using default config")
        return get_default_config()
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Config loaded from: {config_path}")
        return config or {}
        
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        return get_default_config()
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    기본 설정을 반환합니다.
    
    Returns:
        기본 설정 딕셔너리
    """
    return {
        'camera': {
            'device_id': 0,
            'width': 640,
            'height': 480,
            'fps': 30
        },
        'detection': {
            'model_name': 'opencv',
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'min_face_size': (80, 80)
        },
        'recognition': {
            'model_name': 'arcface',
            'similarity_threshold': 0.6,
            'device': 'cpu'
        },
        'paths': {
            'models_dir': 'models/weights',
            'data_dir': 'data',
            'output_dir': 'data/output',
            'logs_dir': 'data/logs'
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    설정을 YAML 파일로 저장합니다.
    
    Args:
        config: 저장할 설정 딕셔너리
        config_path: 저장할 파일 경로
        
    Returns:
        저장 성공 여부
    """
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Config saved to: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")
        return False

def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    설정을 업데이트합니다.
    
    Args:
        config: 기존 설정
        updates: 업데이트할 설정
        
    Returns:
        업데이트된 설정
    """
    updated_config = config.copy()
    
    def deep_update(base: Dict, update: Dict):
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_update(base[key], value)
            else:
                base[key] = value
    
    deep_update(updated_config, updates)
    return updated_config

def validate_config(config: Dict[str, Any]) -> bool:
    """
    설정 유효성을 검증합니다.
    
    Args:
        config: 검증할 설정
        
    Returns:
        유효성 검증 결과
    """
    required_keys = ['camera', 'detection', 'recognition', 'paths']
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False
    
    # 카메라 설정 검증
    camera_config = config.get('camera', {})
    if 'device_id' not in camera_config:
        logger.error("Missing camera.device_id in config")
        return False
    
    # 경로 설정 검증
    paths_config = config.get('paths', {})
    required_paths = ['models_dir', 'data_dir', 'output_dir', 'logs_dir']
    for path_key in required_paths:
        if path_key not in paths_config:
            logger.warning(f"Missing path config: {path_key}")
    
    logger.info("Config validation passed")
    return True 