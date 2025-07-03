#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
데이터 암호화 모듈 (Basic)

기본적인 데이터 암호화 및 보안 기능을 제공합니다.
"""

import base64
import hashlib
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataEncryption:
    """기본 데이터 암호화"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or self._generate_default_key()
    
    def _generate_default_key(self) -> str:
        """기본 암호화 키 생성"""
        # 실제 환경에서는 환경변수나 설정 파일에서 로드
        return os.environ.get('ENCRYPTION_KEY', 'default_vision_system_key_2024')
    
    def encrypt_text(self, text: str) -> str:
        """텍스트 암호화 (기본 해싱)"""
        if not text:
            return ""
        
        # 실제 암호화 대신 해싱 사용 (기본 단계)
        encrypted = hashlib.sha256(
            (text + self.encryption_key).encode()
        ).hexdigest()
        
        return encrypted
    
    def encrypt_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """메타데이터 암호화"""
        encrypted_metadata = {}
        
        for key, value in metadata.items():
            if isinstance(value, str) and self._is_sensitive_field(key):
                encrypted_metadata[key] = self.encrypt_text(value)
            else:
                encrypted_metadata[key] = value
        
        return encrypted_metadata
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """민감한 필드인지 확인"""
        sensitive_fields = [
            'person_id', 'person_name', 'email', 'phone',
            'address', 'ssn', 'credit_card'
        ]
        return field_name.lower() in sensitive_fields
    
    def hash_file(self, file_path: str) -> str:
        """파일 해시 생성"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def verify_file_integrity(self, file_path: str, expected_hash: str) -> bool:
        """파일 무결성 검증"""
        try:
            actual_hash = self.hash_file(file_path)
            return actual_hash == expected_hash
        except Exception as e:
            logger.error(f"File integrity check failed: {e}")
            return False


class ModelSecurity:
    """모델 보안 관리 (기본)"""
    
    def __init__(self):
        self.encryption = DataEncryption()
        self.model_signatures = {}
    
    def sign_model(self, model_path: str) -> str:
        """모델 파일 서명 (해시 기반)"""
        model_hash = self.encryption.hash_file(model_path)
        self.model_signatures[model_path] = model_hash
        logger.info(f"Model signed: {model_path}")
        return model_hash
    
    def verify_model(self, model_path: str, expected_signature: str) -> bool:
        """모델 파일 서명 검증"""
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        actual_hash = self.encryption.hash_file(model_path)
        is_valid = actual_hash == expected_signature
        
        if is_valid:
            logger.info(f"Model verification passed: {model_path}")
        else:
            logger.warning(f"Model verification failed: {model_path}")
        
        return is_valid
    
    def get_model_signature(self, model_path: str) -> Optional[str]:
        """모델 서명 조회"""
        return self.model_signatures.get(model_path) 