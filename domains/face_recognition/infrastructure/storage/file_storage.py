#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File Storage Implementation.

파일 시스템 기반 데이터 저장소 구현체입니다.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid
import shutil

from common.logging import get_logger

logger = get_logger(__name__)


class FileStorage:
    """파일 시스템 기반 저장소"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        저장소 초기화
        
        Args:
            config: 저장소 설정
        """
        self.config = config
        self.base_path = Path(config.get('base_path', 'data/storage'))
        self.backup_enabled = config.get('backup_enabled', True)
        self.backup_path = Path(config.get('backup_path', 'data/backup'))
        self.auto_cleanup = config.get('auto_cleanup', True)
        self.retention_days = config.get('retention_days', 30)
        
        # 디렉토리 생성
        self.base_path.mkdir(parents=True, exist_ok=True)
        if self.backup_enabled:
            self.backup_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"File storage initialized at: {self.base_path}")
    
    def save(self, data: Dict[str, Any], collection: str, doc_id: Optional[str] = None) -> str:
        """
        데이터 저장
        
        Args:
            data: 저장할 데이터
            collection: 컬렉션명 (폴더명)
            doc_id: 문서 ID (None이면 자동 생성)
            
        Returns:
            str: 저장된 문서 ID
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        try:
            # 컬렉션 디렉토리 생성
            collection_path = self.base_path / collection
            collection_path.mkdir(exist_ok=True)
            
            # 데이터에 메타데이터 추가
            data_with_metadata = {
                'id': doc_id,
                'created_at': time.time(),
                'updated_at': time.time(),
                'data': data
            }
            
            # 파일 저장
            file_path = collection_path / f"{doc_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_with_metadata, f, ensure_ascii=False, indent=2, default=str)
            
            # 백업 생성
            if self.backup_enabled:
                self._create_backup(file_path, collection, doc_id)
            
            logger.debug(f"Data saved: {collection}/{doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error saving data to {collection}/{doc_id}: {str(e)}")
            raise
    
    def load(self, doc_id: str, collection: str) -> Optional[Dict[str, Any]]:
        """
        데이터 로드
        
        Args:
            doc_id: 문서 ID
            collection: 컬렉션명
            
        Returns:
            Optional[Dict[str, Any]]: 로드된 데이터 (없으면 None)
        """
        try:
            file_path = self.base_path / collection / f"{doc_id}.json"
            
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data_with_metadata = json.load(f)
            
            return data_with_metadata.get('data')
            
        except Exception as e:
            logger.error(f"Error loading data from {collection}/{doc_id}: {str(e)}")
            return None
    
    def delete(self, doc_id: str, collection: str) -> bool:
        """
        데이터 삭제
        
        Args:
            doc_id: 문서 ID
            collection: 컬렉션명
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            file_path = self.base_path / collection / f"{doc_id}.json"
            
            if not file_path.exists():
                return False
            
            # 백업 생성 (삭제 전)
            if self.backup_enabled:
                self._create_backup(file_path, collection, doc_id, suffix='_deleted')
            
            # 파일 삭제
            file_path.unlink()
            
            logger.debug(f"Data deleted: {collection}/{doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting data from {collection}/{doc_id}: {str(e)}")
            return False
    
    def list_all(self, collection: str) -> List[str]:
        """
        컬렉션의 모든 문서 ID 조회
        
        Args:
            collection: 컬렉션명
            
        Returns:
            List[str]: 문서 ID 리스트
        """
        try:
            collection_path = self.base_path / collection
            
            if not collection_path.exists():
                return []
            
            doc_ids = []
            for file_path in collection_path.glob("*.json"):
                doc_id = file_path.stem
                doc_ids.append(doc_id)
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error listing documents in {collection}: {str(e)}")
            return []
    
    def count(self, collection: str) -> int:
        """
        컬렉션의 문서 수 조회
        
        Args:
            collection: 컬렉션명
            
        Returns:
            int: 문서 수
        """
        return len(self.list_all(collection))
    
    def _create_backup(self, file_path: Path, collection: str, doc_id: str, suffix: str = '') -> None:
        """백업 파일 생성"""
        if not self.backup_enabled:
            return
        
        try:
            backup_collection_path = self.backup_path / collection
            backup_collection_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            backup_filename = f"{doc_id}_{timestamp}{suffix}.json"
            backup_file_path = backup_collection_path / backup_filename
            
            shutil.copy2(file_path, backup_file_path)
            
        except Exception as e:
            logger.warning(f"Failed to create backup for {collection}/{doc_id}: {str(e)}")
    
    def cleanup_old_data(self) -> None:
        """오래된 데이터 정리"""
        if not self.auto_cleanup:
            return
        
        try:
            cutoff_time = time.time() - (self.retention_days * 24 * 3600)
            deleted_count = 0
            
            for collection_path in self.base_path.iterdir():
                if not collection_path.is_dir():
                    continue
                
                for file_path in collection_path.glob("*.json"):
                    try:
                        # 파일 생성 시간 확인
                        if file_path.stat().st_ctime < cutoff_time:
                            file_path.unlink()
                            deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Error deleting old file {file_path}: {str(e)}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old files")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """저장소 정보 반환"""
        try:
            total_size = 0
            total_files = 0
            
            for file_path in self.base_path.rglob("*.json"):
                total_size += file_path.stat().st_size
                total_files += 1
            
            return {
                'storage_type': 'file',
                'base_path': str(self.base_path),
                'total_files': total_files,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'backup_enabled': self.backup_enabled,
                'auto_cleanup': self.auto_cleanup,
                'retention_days': self.retention_days
            }
            
        except Exception as e:
            logger.error(f"Error getting storage info: {str(e)}")
            return {'storage_type': 'file', 'error': str(e)} 