#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
공통 로깅 시스템.

이 모듈은 비전 시스템 전체에서 사용할 수 있는 통합 로깅 기능을 제공합니다.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "data/logs",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> None:
    """로깅 시스템 설정
    
    Args:
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 로그 파일 저장 디렉토리
        log_file: 로그 파일명 (None이면 자동 생성)
        console_output: 콘솔 출력 여부
    """
    # 로그 디렉토리 생성
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 로그 파일명 설정
    if log_file is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        log_file = f"vision_system_{timestamp}.log"
    
    log_file_path = log_path / log_file
    
    # 로그 레벨 설정
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # 로거 설정
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # 파일 핸들러 (회전 로그)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        # 콘솔용 간단한 포맷터
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # 에러 전용 핸들러
    error_file_path = log_path / f"error_{timestamp}.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file_path,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # 초기 로그 메시지
    logger.info(f"Logging system initialized - Level: {log_level}")
    logger.info(f"Log file: {log_file_path}")
    logger.info(f"Error log file: {error_file_path}")


def get_logger(name: str) -> logging.Logger:
    """명명된 로거 반환
    
    Args:
        name: 로거 이름 (보통 __name__ 사용)
        
    Returns:
        설정된 로거 인스턴스
    """
    return logging.getLogger(name)


class LoggerMixin:
    """로깅 기능을 제공하는 믹스인 클래스"""
    
    @property
    def logger(self) -> logging.Logger:
        """클래스별 로거 반환"""
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")


# 성능 로깅을 위한 데코레이터
def log_execution_time(logger: Optional[logging.Logger] = None):
    """함수 실행 시간을 로깅하는 데코레이터
    
    Args:
        logger: 사용할 로거 (None이면 기본 로거 사용)
    """
    import functools
    import time
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            log = logger or logging.getLogger(func.__module__)
            log.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
            
            return result
        return wrapper
    return decorator


# 예외 로깅을 위한 데코레이터
def log_exceptions(logger: Optional[logging.Logger] = None, reraise: bool = True):
    """함수 예외를 로깅하는 데코레이터
    
    Args:
        logger: 사용할 로거 (None이면 기본 로거 사용)
        reraise: 예외를 다시 발생시킬지 여부
    """
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log = logger or logging.getLogger(func.__module__)
                log.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
                
                if reraise:
                    raise
                return None
        return wrapper
    return decorator 