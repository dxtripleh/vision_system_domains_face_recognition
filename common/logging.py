#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common Logging System.

범용 로깅 시스템입니다.
크로스 플랫폼 호환성을 고려하여 설계되었습니다.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    로깅 시스템을 설정합니다.
    
    Args:
        log_level: 로그 레벨 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: 로그 파일 경로 (None이면 자동 생성)
        log_format: 로그 포맷 문자열
        max_bytes: 로그 파일 최대 크기 (바이트)
        backup_count: 백업 파일 개수
        
    Returns:
        루트 로거
    """
    # 프로젝트 루트 경로 계산
    current_file = Path(__file__)
    project_root = current_file.parent.parent
    
    # 로그 레벨 설정
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 로그 포맷 설정
    if log_format is None:
        log_format = (
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
    
    # 로그 파일 경로 설정
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = f"vision_system_{timestamp}.log"
    
    # 로그 파일이 절대 경로가 아니면 data/logs/ 폴더에 저장
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = project_root / "data" / "logs" / log_path.name
    
    # 로그 디렉토리 생성
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 로거 설정
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (로테이션)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 파일 위치 검증 로거 추가
    setup_file_location_logger(logger, project_root)
    
    # 초기화 로그
    logger = logging.getLogger(__name__)
    logger.info(f"로깅 시스템 초기화 완료: {log_path}")
    logger.info(f"로그 레벨: {log_level}")
    
    return logger

def setup_file_location_logger(main_logger: logging.Logger, project_root: Path):
    """파일 위치 검증 로거 설정"""
    # 파일 위치 검증 전용 로거
    file_logger = logging.getLogger('file_location')
    file_logger.setLevel(logging.WARNING)
    
    # 파일 위치 검증 로그 파일
    file_log_path = project_root / "data" / "logs" / "file_location_violations.log"
    file_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(file_log_path, encoding='utf-8')
    file_handler.setLevel(logging.WARNING)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    file_logger.addHandler(file_handler)
    
    # 메인 로거에 추가
    main_logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """
    로거를 가져옵니다.
    
    Args:
        name: 로거 이름
        
    Returns:
        로거 인스턴스
    """
    return logging.getLogger(name)

# 기본 로깅 설정
def setup_default_logging():
    """기본 로깅 설정을 수행합니다."""
    return setup_logging()

# 로깅 설정이 자동으로 수행되도록 함
if not logging.getLogger().handlers:
    setup_default_logging()
