#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
에러 복구 시스템 스크립트 (error_recovery.py)

파이프라인 실행 중 발생하는 오류를 자동으로 복구합니다.
"""

import os
import sys
import argparse
import json
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

class ErrorRecoverySystem:
    """에러 복구 시스템"""
    
    def __init__(self, domain_path: Path):
        self.domain_path = domain_path
        self.backup_dir = domain_path / "backups"
        self.error_log_file = domain_path / "error_recovery.log"
        self.recovery_config_file = domain_path / "recovery_config.json"
        
        # 백업 디렉토리 생성
        self.backup_dir.mkdir(exist_ok=True)
        
        # 로깅 설정
        self.setup_logging()
        
        # 복구 설정 로드
        self.recovery_config = self.load_recovery_config()
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.error_log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_recovery_config(self) -> Dict:
        """복구 설정 로드"""
        if self.recovery_config_file.exists():
            try:
                return json.loads(self.recovery_config_file.read_text(encoding="utf-8"))
            except:
                pass
        
        # 기본 복구 설정
        default_config = {
            "max_retries": 3,
            "retry_delay_seconds": 5,
            "backup_before_retry": True,
            "auto_rollback": True,
            "error_thresholds": {
                "file_corruption": 1,
                "processing_timeout": 2,
                "memory_overflow": 1
            },
            "recovery_strategies": {
                "file_corruption": "restore_from_backup",
                "processing_timeout": "increase_timeout",
                "memory_overflow": "reduce_batch_size",
                "unknown_error": "retry_with_backoff"
            }
        }
        
        self.save_recovery_config(default_config)
        return default_config
    
    def save_recovery_config(self, config: Dict):
        """복구 설정 저장"""
        self.recovery_config_file.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), 
            encoding="utf-8"
        )
    
    def create_backup(self, stage: str, description: str = "") -> str:
        """단계별 백업 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{stage}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        try:
            # 원본 단계 폴더 복사
            stage_path = self.domain_path / stage
            if stage_path.exists():
                shutil.copytree(stage_path, backup_path)
                
                # 백업 메타데이터 생성
                metadata = {
                    "backup_name": backup_name,
                    "stage": stage,
                    "created_at": datetime.now().isoformat(),
                    "description": description,
                    "original_path": str(stage_path)
                }
                
                metadata_file = backup_path / "backup_metadata.json"
                metadata_file.write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False), 
                    encoding="utf-8"
                )
                
                self.logger.info(f"백업 생성 완료: {backup_name}")
                return backup_name
            else:
                self.logger.warning(f"백업할 단계 폴더가 존재하지 않음: {stage}")
                return ""
                
        except Exception as e:
            self.logger.error(f"백업 생성 실패: {e}")
            return ""
    
    def restore_from_backup(self, stage: str, backup_name: str) -> bool:
        """백업에서 복원"""
        try:
            backup_path = self.backup_dir / backup_name
            if not backup_path.exists():
                self.logger.error(f"백업이 존재하지 않음: {backup_name}")
                return False
            
            stage_path = self.domain_path / stage
            
            # 기존 폴더 삭제
            if stage_path.exists():
                shutil.rmtree(stage_path)
            
            # 백업에서 복원
            shutil.copytree(backup_path, stage_path)
            
            # 백업 메타데이터 파일 제거
            metadata_file = stage_path / "backup_metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            self.logger.info(f"백업에서 복원 완료: {backup_name} → {stage}")
            return True
            
        except Exception as e:
            self.logger.error(f"백업 복원 실패: {e}")
            return False
    
    def detect_error_type(self, error_message: str) -> str:
        """오류 유형 감지"""
        error_message_lower = error_message.lower()
        
        if any(keyword in error_message_lower for keyword in ["corrupt", "damaged", "invalid"]):
            return "file_corruption"
        elif any(keyword in error_message_lower for keyword in ["timeout", "timed out"]):
            return "processing_timeout"
        elif any(keyword in error_message_lower for keyword in ["memory", "out of memory"]):
            return "memory_overflow"
        else:
            return "unknown_error"
    
    def get_recovery_strategy(self, error_type: str) -> str:
        """복구 전략 결정"""
        return self.recovery_config["recovery_strategies"].get(error_type, "retry_with_backoff")
    
    def execute_recovery_strategy(self, stage: str, error_type: str, error_message: str, retry_count: int = 0) -> bool:
        """복구 전략 실행"""
        strategy = self.get_recovery_strategy(error_type)
        
        self.logger.info(f"복구 전략 실행: {strategy} (오류: {error_type})")
        
        if strategy == "restore_from_backup":
            return self.restore_from_backup_strategy(stage, error_type)
        elif strategy == "increase_timeout":
            return self.increase_timeout_strategy(stage, error_type)
        elif strategy == "reduce_batch_size":
            return self.reduce_batch_size_strategy(stage, error_type)
        elif strategy == "retry_with_backoff":
            return self.retry_with_backoff_strategy(stage, error_type, retry_count)
        else:
            self.logger.warning(f"알 수 없는 복구 전략: {strategy}")
            return False
    
    def restore_from_backup_strategy(self, stage: str, error_type: str) -> bool:
        """백업에서 복원 전략"""
        # 가장 최근 백업 찾기
        backups = list(self.backup_dir.glob(f"{stage}_*"))
        if not backups:
            self.logger.error(f"{stage} 단계의 백업이 없습니다")
            return False
        
        # 가장 최근 백업 선택
        latest_backup = max(backups, key=lambda x: x.stat().st_mtime)
        backup_name = latest_backup.name
        
        return self.restore_from_backup(stage, backup_name)
    
    def increase_timeout_strategy(self, stage: str, error_type: str) -> bool:
        """타임아웃 증가 전략"""
        # 설정 파일에서 타임아웃 값 증가
        config_file = self.domain_path / "pipeline_config.json"
        if config_file.exists():
            try:
                config = json.loads(config_file.read_text(encoding="utf-8"))
                current_timeout = config.get("timeout", 30)
                new_timeout = current_timeout * 2
                config["timeout"] = new_timeout
                
                config_file.write_text(
                    json.dumps(config, indent=2, ensure_ascii=False), 
                    encoding="utf-8"
                )
                
                self.logger.info(f"타임아웃 증가: {current_timeout}s → {new_timeout}s")
                return True
                
            except Exception as e:
                self.logger.error(f"타임아웃 설정 수정 실패: {e}")
                return False
        else:
            self.logger.warning("설정 파일이 없어 타임아웃 증가 불가")
            return False
    
    def reduce_batch_size_strategy(self, stage: str, error_type: str) -> bool:
        """배치 크기 감소 전략"""
        # 설정 파일에서 배치 크기 감소
        config_file = self.domain_path / "pipeline_config.json"
        if config_file.exists():
            try:
                config = json.loads(config_file.read_text(encoding="utf-8"))
                current_batch_size = config.get("batch_size", 16)
                new_batch_size = max(1, current_batch_size // 2)
                config["batch_size"] = new_batch_size
                
                config_file.write_text(
                    json.dumps(config, indent=2, ensure_ascii=False), 
                    encoding="utf-8"
                )
                
                self.logger.info(f"배치 크기 감소: {current_batch_size} → {new_batch_size}")
                return True
                
            except Exception as e:
                self.logger.error(f"배치 크기 설정 수정 실패: {e}")
                return False
        else:
            self.logger.warning("설정 파일이 없어 배치 크기 감소 불가")
            return False
    
    def retry_with_backoff_strategy(self, stage: str, error_type: str, retry_count: int) -> bool:
        """지수 백오프 재시도 전략"""
        if retry_count >= self.recovery_config["max_retries"]:
            self.logger.error(f"최대 재시도 횟수 초과: {retry_count}")
            return False
        
        # 지수 백오프 계산
        delay = self.recovery_config["retry_delay_seconds"] * (2 ** retry_count)
        self.logger.info(f"재시도 대기: {delay}초 (재시도 {retry_count + 1}/{self.recovery_config['max_retries']})")
        
        time.sleep(delay)
        return True
    
    def handle_pipeline_error(self, stage: str, error: Exception, retry_count: int = 0) -> bool:
        """파이프라인 오류 처리"""
        error_message = str(error)
        error_type = self.detect_error_type(error_message)
        
        self.logger.error(f"파이프라인 오류 발생: {stage} - {error_type} - {error_message}")
        
        # 백업 생성 (설정에 따라)
        if self.recovery_config["backup_before_retry"]:
            backup_name = self.create_backup(stage, f"오류 복구용 백업: {error_type}")
        
        # 복구 전략 실행
        recovery_success = self.execute_recovery_strategy(stage, error_type, error_message, retry_count)
        
        if recovery_success:
            self.logger.info(f"복구 성공: {stage}")
        else:
            self.logger.error(f"복구 실패: {stage}")
        
        return recovery_success
    
    def cleanup_old_backups(self, keep_days: int = 7):
        """오래된 백업 정리"""
        cutoff_time = time.time() - (keep_days * 24 * 3600)
        removed_count = 0
        
        for backup_path in self.backup_dir.iterdir():
            if backup_path.is_dir():
                if backup_path.stat().st_mtime < cutoff_time:
                    try:
                        shutil.rmtree(backup_path)
                        removed_count += 1
                        self.logger.info(f"오래된 백업 삭제: {backup_path.name}")
                    except Exception as e:
                        self.logger.error(f"백업 삭제 실패: {backup_path.name} - {e}")
        
        self.logger.info(f"백업 정리 완료: {removed_count}개 삭제")
    
    def list_backups(self) -> List[Dict]:
        """백업 목록 조회"""
        backups = []
        
        for backup_path in self.backup_dir.iterdir():
            if backup_path.is_dir():
                metadata_file = backup_path / "backup_metadata.json"
                if metadata_file.exists():
                    try:
                        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                        backups.append(metadata)
                    except:
                        pass
        
        # 생성 시간순 정렬
        backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return backups
    
    def print_backup_list(self):
        """백업 목록 출력"""
        backups = self.list_backups()
        
        if not backups:
            print(" 백업이 없습니다.")
            return
        
        print(f" 백업 목록 ({len(backups)}개):")
        for i, backup in enumerate(backups, 1):
            print(f"  {i}. {backup['backup_name']}")
            print(f"     단계: {backup['stage']}")
            print(f"     생성: {backup['created_at']}")
            print(f"     설명: {backup['description']}")
            print()

def main():
    parser = argparse.ArgumentParser(description="에러 복구 시스템")
    parser.add_argument("domain", help="도메인명")
    parser.add_argument("feature", help="기능명")
    parser.add_argument("--backup", help="백업 생성 (단계명)")
    parser.add_argument("--restore", help="백업에서 복원 (백업명)")
    parser.add_argument("--list", action="store_true", help="백업 목록 조회")
    parser.add_argument("--cleanup", type=int, default=7, help="오래된 백업 정리 (보관일수)")
    parser.add_argument("--base-path", default="data/domains", help="기본 경로")
    
    args = parser.parse_args()
    
    domain_path = Path(args.base_path) / args.domain / args.feature
    
    if not domain_path.exists():
        print(f" 도메인 경로가 존재하지 않습니다: {domain_path}")
        sys.exit(1)
    
    recovery_system = ErrorRecoverySystem(domain_path)
    
    if args.backup:
        backup_name = recovery_system.create_backup(args.backup, "수동 백업")
        if backup_name:
            print(f"백업 생성 완료: {backup_name}")
        else:
            print("백업 생성 실패")
    
    elif args.restore:
        success = recovery_system.restore_from_backup("", args.restore)
        if success:
            print("백업 복원 완료")
        else:
            print("백업 복원 실패")
    
    elif args.list:
        recovery_system.print_backup_list()
    
    elif args.cleanup:
        recovery_system.cleanup_old_backups(args.cleanup)
    
    else:
        print(f"에러 복구 시스템이 초기화되었습니다.")
        print(f"도메인: {args.domain}/{args.feature}")
        print(f"백업 디렉토리: {recovery_system.backup_dir}")
        print(f"오류 로그: {recovery_system.error_log_file}")

if __name__ == "__main__":
    main()
