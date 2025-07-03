#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Storage Module.

데이터 저장소 구현체들을 관리하는 모듈입니다.
"""

from typing import Dict, Type, Any
from .file_storage import FileStorage
from .database_storage import DatabaseStorage

# 사용 가능한 저장소들
AVAILABLE_STORAGES = {
    'file': FileStorage,
    'database': DatabaseStorage,
}

def create_storage(storage_type: str, config: Dict[str, Any]):
    """
    저장소 팩토리 함수
    
    Args:
        storage_type: 저장소 타입 ('file', 'database')
        config: 저장소 설정
        
    Returns:
        저장소 인스턴스
    """
    if storage_type not in AVAILABLE_STORAGES:
        raise ValueError(f"Unknown storage type: {storage_type}")
    
    storage_class = AVAILABLE_STORAGES[storage_type]
    return storage_class(config)

__all__ = [
    'FileStorage',
    'DatabaseStorage',
    'create_storage',
    'AVAILABLE_STORAGES'
] 