#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
개인정보 보호 모듈 (Basic)

얼굴 인식 데이터의 기본적인 개인정보 보호 기능을 제공합니다.
"""

import hashlib
import time
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FaceDataProtection:
    """얼굴 인식 데이터 보호 (기본 GDPR 준수)"""
    
    def __init__(self):
        self.anonymization_enabled = True
        self.retention_policy = {
            'face_embeddings': 30,  # 30일
            'raw_images': 7,        # 7일
            'detection_logs': 90    # 90일
        }
        self.consent_manager = ConsentManager()
    
    def anonymize_face_data(self, face_data: Dict[str, Any]) -> Dict[str, Any]:
        """얼굴 데이터 익명화"""
        if not self.anonymization_enabled:
            return face_data
        
        anonymized_data = face_data.copy()
        
        # 개인 식별 정보 제거
        anonymized_data.pop('person_id', None)
        anonymized_data.pop('person_name', None)
        anonymized_data.pop('email', None)
        anonymized_data.pop('phone', None)
        
        # 얼굴 임베딩 해싱
        if 'embedding' in anonymized_data:
            embedding_hash = hashlib.sha256(
                str(anonymized_data['embedding']).encode()
            ).hexdigest()
            anonymized_data['embedding_hash'] = embedding_hash
            del anonymized_data['embedding']
        
        # 타임스탬프 일반화 (시간 단위로)
        if 'timestamp' in anonymized_data:
            timestamp = anonymized_data['timestamp']
            anonymized_data['timestamp'] = int(timestamp // 3600) * 3600
        
        return anonymized_data
    
    def apply_retention_policy(self):
        """데이터 보존 정책 적용"""
        current_time = time.time()
        
        for data_type, retention_days in self.retention_policy.items():
            cutoff_time = current_time - (retention_days * 24 * 3600)
            self._delete_old_data(data_type, cutoff_time)
    
    def _delete_old_data(self, data_type: str, cutoff_time: float):
        """오래된 데이터 자동 삭제"""
        data_dir = f"data/domains/face_recognition/{data_type}"
        if not os.path.exists(data_dir):
            return
        
        deleted_count = 0
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.getctime(file_path) < cutoff_time:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")
        
        if deleted_count > 0:
            logger.info(f"{data_type}: {deleted_count}개 파일 삭제됨 (보존 정책)")


class ConsentManager:
    """사용자 동의 관리 (기본)"""
    
    def __init__(self):
        self.consent_db = {}  # 실제로는 데이터베이스 사용
        
    def check_consent(self, person_id: str, purpose: str) -> bool:
        """사용자 동의 확인"""
        consent_key = f"{person_id}_{purpose}"
        return self.consent_db.get(consent_key, False)
    
    def record_consent(self, person_id: str, purpose: str, granted: bool):
        """사용자 동의 기록"""
        consent_key = f"{person_id}_{purpose}"
        self.consent_db[consent_key] = {
            'granted': granted,
            'timestamp': time.time(),
            'purpose': purpose
        }
        logger.info(f"Consent recorded: {person_id} - {purpose} = {granted}")
    
    def revoke_consent(self, person_id: str, purpose: str):
        """사용자 동의 철회"""
        consent_key = f"{person_id}_{purpose}"
        if consent_key in self.consent_db:
            del self.consent_db[consent_key]
            logger.info(f"Consent revoked: {person_id} - {purpose}") 