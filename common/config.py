#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common Configuration Management.

범용 설정 관리 모듈입니다.
크로스 플랫폼 호환성을 고려하여 설계되었습니다.
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ConfigManager:
    """설정 관리자 클래스."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        설정 관리자 초기화.
        
        Args:
            config_dir: 설정 파일 디렉토리 경로
        """
        # 프로젝트 루트 경로 계산
        current_dir = Path(__file__).parent
        self.project_root = current_dir.parent.parent
        
        # 설정 디렉토리 설정
        self.config_dir = Path(config_dir) if config_dir else self.project_root / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        # 설정 캐시
        self._config_cache = {}
        
        # 환경별 설정
        self.environment = os.getenv('ENVIRONMENT', 'development')
        
        logger.info(f"설정 관리자 초기화: {self.config_dir}")
    
    def load_config(self, config_name: str, config_type: str = 'yaml') -> Dict[str, Any]:
        """
        설정 파일을 로드합니다.
        
        Args:
            config_name: 설정 파일명 (확장자 제외)
            config_type: 설정 파일 타입 ('yaml', 'json')
            
        Returns:
            설정 딕셔너리
        """
        cache_key = f"{config_name}_{config_type}_{self.environment}"
        
        # 캐시 확인
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        # 기본 설정 파일 경로
        base_config_path = self.config_dir / f"{config_name}.{config_type}"
        
        # 환경별 설정 파일 경로
        env_config_path = self.config_dir / f"{config_name}_{self.environment}.{config_type}"
        
        # 설정 로드
        config = {}
        
        # 기본 설정 로드
        if base_config_path.exists():
            config.update(self._load_file(base_config_path, config_type))
        
        # 환경별 설정 로드 (기본 설정을 오버라이드)
        if env_config_path.exists():
            env_config = self._load_file(env_config_path, config_type)
            config = self._deep_merge(config, env_config)
        
        # 환경 변수 치환
        config = self._resolve_env_variables(config)
        
        # 캐시에 저장
        self._config_cache[cache_key] = config
        
        logger.debug(f"설정 로드 완료: {config_name}")
        return config
    
    def _load_file(self, file_path: Path, config_type: str) -> Dict[str, Any]:
        """파일을 로드합니다."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if config_type == 'yaml':
                    return yaml.safe_load(f) or {}
                elif config_type == 'json':
                    return json.load(f)
                else:
                    raise ValueError(f"지원하지 않는 설정 파일 타입: {config_type}")
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {file_path}, {e}")
            return {}
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """딕셔너리를 깊게 병합합니다."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _resolve_env_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """환경 변수를 치환합니다."""
        import re
        
        def resolve_value(value):
            if isinstance(value, str):
                # ${VAR} 패턴 치환
                pattern = r'\$\{([^}]+)\}'
                
                def replace_var(match):
                    var_name = match.group(1)
                    return os.getenv(var_name, match.group(0))
                
                return re.sub(pattern, replace_var, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            else:
                return value
        
        return resolve_value(config)
    
    def save_config(self, config_name: str, config: Dict[str, Any], config_type: str = 'yaml'):
        """
        설정을 파일로 저장합니다.
        
        Args:
            config_name: 설정 파일명
            config: 저장할 설정 딕셔너리
            config_type: 설정 파일 타입
        """
        file_path = self.config_dir / f"{config_name}.{config_type}"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if config_type == 'yaml':
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                elif config_type == 'json':
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"지원하지 않는 설정 파일 타입: {config_type}")
            
            logger.info(f"설정 저장 완료: {file_path}")
            
        except Exception as e:
            logger.error(f"설정 저장 실패: {file_path}, {e}")
            raise
    
    def get_project_root(self) -> Path:
        """프로젝트 루트 경로를 반환합니다."""
        return self.project_root
    
    def get_data_dir(self) -> Path:
        """데이터 디렉토리 경로를 반환합니다."""
        return self.project_root / "data"
    
    def get_models_dir(self) -> Path:
        """모델 디렉토리 경로를 반환합니다."""
        return self.project_root / "models"
    
    def get_logs_dir(self) -> Path:
        """로그 디렉토리 경로를 반환합니다."""
        return self.project_root / "data" / "logs"
    
    def get_output_dir(self) -> Path:
        """출력 디렉토리 경로를 반환합니다."""
        return self.project_root / "data" / "output"
    
    def clear_cache(self):
        """설정 캐시를 클리어합니다."""
        self._config_cache.clear()
        logger.debug("설정 캐시 클리어 완료")

# 전역 설정 관리자 인스턴스
config_manager = ConfigManager()

def load_config(config_name: str, config_type: str = 'yaml') -> Dict[str, Any]:
    """
    설정을 로드하는 편의 함수.
    
    Args:
        config_name: 설정 파일명
        config_type: 설정 파일 타입
        
    Returns:
        설정 딕셔너리
    """
    return config_manager.load_config(config_name, config_type)

def save_config(config_name: str, config: Dict[str, Any], config_type: str = 'yaml'):
    """
    설정을 저장하는 편의 함수.
    
    Args:
        config_name: 설정 파일명
        config: 저장할 설정 딕셔너리
        config_type: 설정 파일 타입
    """
    config_manager.save_config(config_name, config, config_type)

@contextmanager
def config_context(config_name: str, config_type: str = 'yaml'):
    """
    설정 컨텍스트 매니저.
    
    Args:
        config_name: 설정 파일명
        config_type: 설정 파일 타입
        
    Yields:
        설정 딕셔너리
    """
    config = load_config(config_name, config_type)
    try:
        yield config
    finally:
        pass 