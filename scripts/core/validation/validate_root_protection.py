#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
루트 디렉토리 보호 검증 시스템.

이 모듈은 프로젝트 루트에 생성되면 안 되는 파일들을 감시하고 방지합니다.
.gitignore와는 별도로 실제 개발 규칙을 강제합니다.
"""

import os
import sys
import time
import shutil
from pathlib import Path
from typing import List, Dict, Set, Optional
import logging
from datetime import datetime
import json

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging

class RootProtectionValidator:
    """루트 디렉토리 보호 검증기"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.setup_logging()
        
        # 📋 루트에 생성 금지된 파일 패턴들
        self.forbidden_patterns = {
            # 🚫 로그 파일들
            'logs': ['*.log', 'error_*.log', 'debug_*.log', 'system_*.log'],
            
            # 🚫 임시 파일들
            'temp': ['*.tmp', '*.temp', 'temp_*', 'cache_*', '_temp_*'],
            
            # 🚫 결과 파일들
            'output': ['output_*', 'result_*', 'processed_*'],
            
            # 🚫 모델 파일들
            'models': ['*.onnx', '*.pt', '*.pth', '*.h5', '*.pb', '*.tflite', '*.bin'],
            
            # 🚫 이미지/비디오 파일들
            'media': ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.mp4', '*.avi', '*.mov'],
            
            # 🚫 데이터 파일들
            'data': ['*.csv', '*.json', '*.xml', '*.parquet'],
            
            # 🚫 스크립트 파일들 (scripts 폴더에 있어야 함)
            'scripts': ['run_*.py', 'test_*.py', 'demo_*.py', 'example_*.py', 'benchmark_*.py'],
            
            # 🚫 설정 파일들 (config 폴더에 있어야 함)
            'configs': ['*.yaml', '*.yml', '*.toml', '*.ini'],
            
            # 🚫 백업 파일들
            'backups': ['*_backup.*', '*_old.*', '*.bak'],
            
            # 🚫 개발 중 생성되는 기타 파일들
            'misc': ['*.prof', '*.dump', '*.pickle', '*.pkl']
        }
        
        # ✅ 예외적으로 허용되는 파일들
        self.allowed_root_files = {
            'README.md', '.gitignore', 'requirements.txt', 'LICENSE', 
            'pyproject.toml', 'setup.py', 'setup.cfg', 'Makefile',
            'Dockerfile', 'docker-compose.yml', '.env.example'
        }
        
        # 📁 올바른 경로 매핑
        self.correct_locations = {
            'logs': 'data/logs/',
            'temp': 'data/temp/', 
            'output': 'data/output/',
            'models': 'models/weights/',
            'media': 'data/output/',
            'data': 'data/output/',
            'scripts': 'scripts/',
            'configs': 'config/',
            'backups': 'data/temp/',
            'misc': 'data/temp/'
        }
        
        # 위반 기록 파일
        self.violation_log = self.project_root / 'data' / 'logs' / 'root_protection_violations.json'
        self.ensure_log_directory()
    
    def setup_logging(self):
        """로깅 설정"""
        setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def ensure_log_directory(self):
        """로그 디렉토리 존재 보장"""
        self.violation_log.parent.mkdir(parents=True, exist_ok=True)
    
    def scan_root_violations(self) -> Dict[str, List[str]]:
        """루트 디렉토리 위반 파일 스캔"""
        violations = {}
        
        for file_path in self.project_root.iterdir():
            if file_path.is_file():
                filename = file_path.name
                
                # 허용된 파일인지 확인
                if filename in self.allowed_root_files:
                    continue
                
                # 금지된 패턴 확인
                violation_type = self._check_forbidden_pattern(filename)
                if violation_type:
                    if violation_type not in violations:
                        violations[violation_type] = []
                    violations[violation_type].append(filename)
        
        return violations
    
    def _check_forbidden_pattern(self, filename: str) -> Optional[str]:
        """파일명이 금지된 패턴인지 확인"""
        import fnmatch
        
        for category, patterns in self.forbidden_patterns.items():
            for pattern in patterns:
                if fnmatch.fnmatch(filename, pattern):
                    return category
        return None
    
    def auto_cleanup_violations(self, violations: Dict[str, List[str]], 
                              dry_run: bool = False) -> Dict[str, int]:
        """위반 파일들 자동 정리"""
        cleanup_results = {}
        
        for category, files in violations.items():
            moved_count = 0
            target_dir = self.project_root / self.correct_locations[category]
            
            # 대상 디렉토리 생성
            if not dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)
            
            for filename in files:
                source_path = self.project_root / filename
                target_path = target_dir / filename
                
                try:
                    if not dry_run:
                        # 파일명 충돌 방지
                        if target_path.exists():
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            name_parts = filename.rsplit('.', 1)
                            if len(name_parts) == 2:
                                new_name = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
                            else:
                                new_name = f"{filename}_{timestamp}"
                            target_path = target_dir / new_name
                        
                        shutil.move(str(source_path), str(target_path))
                        self.logger.info(f"파일 이동: {filename} → {target_path}")
                    
                    moved_count += 1
                    
                except Exception as e:
                    self.logger.error(f"파일 이동 실패 {filename}: {str(e)}")
            
            cleanup_results[category] = moved_count
        
        return cleanup_results
    
    def log_violation(self, violations: Dict[str, List[str]]):
        """위반 내역 로깅"""
        violation_record = {
            'timestamp': datetime.now().isoformat(),
            'violations': violations,
            'total_files': sum(len(files) for files in violations.values())
        }
        
        # 기존 로그 읽기
        existing_logs = []
        if self.violation_log.exists():
            try:
                with open(self.violation_log, 'r', encoding='utf-8') as f:
                    existing_logs = json.load(f)
            except:
                existing_logs = []
        
        # 새 로그 추가
        existing_logs.append(violation_record)
        
        # 최대 100개 기록만 유지
        if len(existing_logs) > 100:
            existing_logs = existing_logs[-100:]
        
        # 로그 저장
        with open(self.violation_log, 'w', encoding='utf-8') as f:
            json.dump(existing_logs, f, indent=2, ensure_ascii=False)
    
    def generate_violation_report(self, violations: Dict[str, List[str]]) -> str:
        """위반 리포트 생성"""
        if not violations:
            return "✅ 루트 디렉토리 보호 규칙 준수: 위반 파일 없음"
        
        report_lines = ["❌ 루트 디렉토리 보호 규칙 위반 발견!"]
        report_lines.append("=" * 60)
        
        total_violations = sum(len(files) for files in violations.values())
        report_lines.append(f"총 위반 파일: {total_violations}개")
        report_lines.append("")
        
        for category, files in violations.items():
            report_lines.append(f"📁 {category.upper()} 위반 ({len(files)}개):")
            report_lines.append(f"   올바른 위치: {self.correct_locations[category]}")
            for file in files:
                report_lines.append(f"   - {file}")
            report_lines.append("")
        
        report_lines.append("🔧 해결 방법:")
        report_lines.append("   1. 자동 정리: python scripts/core/validation/validate_root_protection.py --auto-fix")
        report_lines.append("   2. 수동 이동: 각 파일을 올바른 폴더로 이동")
        report_lines.append("   3. 개발 규칙 준수: 처음부터 올바른 경로에 파일 생성")
        
        return "\n".join(report_lines)
    
    def monitor_realtime(self, duration_seconds: int = 60):
        """실시간 모니터링"""
        self.logger.info(f"루트 디렉토리 실시간 모니터링 시작 ({duration_seconds}초)")
        
        start_time = time.time()
        check_interval = 5  # 5초마다 체크
        
        while time.time() - start_time < duration_seconds:
            violations = self.scan_root_violations()
            
            if violations:
                self.logger.warning("실시간 위반 감지!")
                print(self.generate_violation_report(violations))
                
                # 자동 정리 옵션
                response = input("자동 정리를 실행하시겠습니까? (y/n): ")
                if response.lower() == 'y':
                    self.auto_cleanup_violations(violations)
            
            time.sleep(check_interval)
        
        self.logger.info("실시간 모니터링 종료")
    
    def validate_development_rules(self) -> bool:
        """개발 규칙 검증"""
        violations = self.scan_root_violations()
        
        if violations:
            # 위반 로깅
            self.log_violation(violations)
            
            # 리포트 출력
            print(self.generate_violation_report(violations))
            
            return False
        else:
            print("✅ 루트 디렉토리 보호 규칙 준수: 모든 파일이 올바른 위치에 있습니다.")
            return True

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="루트 디렉토리 보호 검증")
    parser.add_argument("--auto-fix", action="store_true", help="위반 파일 자동 정리")
    parser.add_argument("--dry-run", action="store_true", help="실제 이동 없이 시뮬레이션")
    parser.add_argument("--monitor", type=int, help="실시간 모니터링 (초 단위)")
    
    args = parser.parse_args()
    
    validator = RootProtectionValidator()
    
    if args.monitor:
        validator.monitor_realtime(args.monitor)
    else:
        violations = validator.scan_root_violations()
        
        if violations:
            print(validator.generate_violation_report(violations))
            validator.log_violation(violations)
            
            if args.auto_fix:
                print("\n🔧 자동 정리 실행 중...")
                results = validator.auto_cleanup_violations(violations, dry_run=args.dry_run)
                
                for category, count in results.items():
                    print(f"   {category}: {count}개 파일 정리됨")
                
                if args.dry_run:
                    print("   (시뮬레이션 모드 - 실제 파일은 이동되지 않음)")
        else:
            print("✅ 루트 디렉토리 보호 규칙 준수: 모든 파일이 올바른 위치에 있습니다.")

if __name__ == "__main__":
    main() 