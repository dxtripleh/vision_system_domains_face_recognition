#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
파일 위치 검증 스크립트.

개발 중 최상위 루트에 임시 파일이나 디버그 파일이 생성되는 것을 방지합니다.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Set
import time
import threading

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from common.logging import setup_logging

logger = logging.getLogger(__name__)

class FileLocationValidator:
    """파일 위치 검증기"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.forbidden_patterns = self._get_forbidden_patterns()
        self.allowed_locations = self._get_allowed_locations()
        self.monitoring = False
        self.monitor_thread = None
        
    def _get_forbidden_patterns(self) -> List[str]:
        """금지된 파일 패턴"""
        return [
            # 디버그 및 임시 파일
            "*.log",
            "*.tmp", 
            "*.temp",
            "debug_*",
            "temp_*",
            "test_*",
            "output_*",
            "result_*",
            "capture_*",
            "frame_*",
            "image_*",
            "video_*",
            "*.jpg",
            "*.png", 
            "*.mp4",
            "*.avi",
            "*.mov",
            "*.json",
            "*.csv",
            "*.txt",
            "*.dat",
            "*.bin",
            "*.pkl",
            "*.npy",
            "*.npz",
            # IDE 및 편집기 파일
            "*.swp",
            "*.swo",
            "*~",
            ".DS_Store",
            "Thumbs.db",
            # Python 관련
            "*.pyc",
            "__pycache__",
            "*.pyo",
            "*.pyd"
        ]
    
    def _get_allowed_locations(self) -> Dict[str, List[str]]:
        """허용된 파일 위치"""
        return {
            'data/temp/': ['*.log', '*.tmp', '*.temp', '*.jpg', '*.png', '*.mp4', '*.json', '*.csv'],
            'data/logs/': ['*.log', '*.txt'],
            'data/output/': ['*.jpg', '*.png', '*.mp4', '*.json', '*.csv', '*.txt'],
            'data/domains/': ['*.jpg', '*.png', '*.mp4', '*.json', '*.csv'],
            'tests/data/': ['*.jpg', '*.png', '*.mp4', '*.json', '*.csv'],
            'experiments/': ['*.log', '*.tmp', '*.jpg', '*.png', '*.json', '*.csv'],
            'notebooks/': ['*.ipynb', '*.log', '*.tmp']
        }
    
    def validate_current_state(self) -> Dict[str, List[str]]:
        """현재 상태 검증"""
        violations = {}
        
        for pattern in self.forbidden_patterns:
            found_files = list(self.project_root.glob(pattern))
            if found_files:
                violations[pattern] = [str(f) for f in found_files]
        
        return violations
    
    def check_file_location(self, file_path: Path) -> bool:
        """파일 위치가 올바른지 확인"""
        if not file_path.exists():
            return True
        
        # 프로젝트 루트에 직접 있는 파일인지 확인
        if file_path.parent == self.project_root:
            filename = file_path.name
            
            # 허용된 파일들 (프로젝트 설정 파일들)
            allowed_files = {
                'README.md', 'requirements.txt', 'pyproject.toml', 
                'setup.py', 'LICENSE', '.gitignore', '.env.example',
                'pytest.ini', 'tox.ini', 'Makefile', 'Dockerfile',
                'docker-compose.yml', '.dockerignore'
            }
            
            if filename in allowed_files:
                return True
            
            # 금지된 패턴과 매치되는지 확인
            for pattern in self.forbidden_patterns:
                if self._pattern_matches(filename, pattern):
                    return False
        
        return True
    
    def _pattern_matches(self, filename: str, pattern: str) -> bool:
        """파일명이 패턴과 매치되는지 확인"""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)
    
    def suggest_correct_location(self, filename: str) -> str:
        """올바른 파일 위치 제안"""
        if filename.endswith('.log'):
            return "data/logs/"
        elif filename.endswith(('.jpg', '.png', '.mp4', '.avi')):
            return "data/output/"
        elif filename.endswith(('.json', '.csv', '.txt')):
            return "data/temp/"
        elif filename.endswith('.tmp') or filename.endswith('.temp'):
            return "data/temp/"
        else:
            return "data/temp/"
    
    def start_monitoring(self):
        """실시간 파일 생성 모니터링 시작"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("파일 위치 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("파일 위치 모니터링 중지")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring:
            try:
                violations = self.validate_current_state()
                if violations:
                    self._handle_violations(violations)
                time.sleep(5)  # 5초마다 체크
            except Exception as e:
                logger.error(f"모니터링 오류: {str(e)}")
    
    def _handle_violations(self, violations: Dict[str, List[str]]):
        """위반 사항 처리"""
        for pattern, files in violations.items():
            for file_path in files:
                logger.warning(f"금지된 위치에 파일 발견: {file_path}")
                logger.warning(f"패턴: {pattern}")
                
                # 올바른 위치 제안
                filename = Path(file_path).name
                suggested_location = self.suggest_correct_location(filename)
                logger.info(f"제안 위치: {suggested_location}{filename}")
                
                # 자동 이동 옵션 (선택적)
                if self._should_auto_move(file_path):
                    self._move_file_to_correct_location(file_path, suggested_location)
    
    def _should_auto_move(self, file_path: str) -> bool:
        """파일을 자동으로 이동할지 결정"""
        # 개발 환경에서만 자동 이동
        return os.environ.get('AUTO_MOVE_FILES', 'false').lower() == 'true'
    
    def _move_file_to_correct_location(self, file_path: str, suggested_location: str):
        """파일을 올바른 위치로 이동"""
        try:
            source_path = Path(file_path)
            target_dir = self.project_root / suggested_location
            target_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = target_dir / source_path.name
            source_path.rename(target_path)
            
            logger.info(f"파일 이동 완료: {file_path} -> {target_path}")
        except Exception as e:
            logger.error(f"파일 이동 실패: {str(e)}")

def create_file_location_hook():
    """Git pre-commit 훅 생성"""
    hook_content = '''#!/bin/bash
# Git pre-commit 훅: 파일 위치 검증

echo "파일 위치 검증 중..."

# Python 스크립트 실행
python scripts/validate_file_locations.py --pre-commit

if [ $? -ne 0 ]; then
    echo "❌ 파일 위치 검증 실패"
    echo "최상위 루트에 임시 파일이 생성되었습니다."
    echo "올바른 위치로 파일을 이동하세요."
    exit 1
fi

echo "✅ 파일 위치 검증 통과"
'''
    
    hook_path = Path('.git/hooks/pre-commit')
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(hook_path, 'w') as f:
        f.write(hook_content)
    
    # 실행 권한 부여
    os.chmod(hook_path, 0o755)
    logger.info("Git pre-commit 훅 생성 완료")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="파일 위치 검증")
    parser.add_argument("--pre-commit", action="store_true", help="Git pre-commit 훅 모드")
    parser.add_argument("--monitor", action="store_true", help="실시간 모니터링 모드")
    parser.add_argument("--create-hook", action="store_true", help="Git 훅 생성")
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    
    project_root = Path(__file__).parent.parent
    validator = FileLocationValidator(project_root)
    
    if args.create_hook:
        create_file_location_hook()
        return
    
    if args.monitor:
        try:
            validator.start_monitoring()
            print("파일 위치 모니터링 중... (Ctrl+C로 종료)")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            validator.stop_monitoring()
            print("\n모니터링 종료")
        return
    
    # 현재 상태 검증
    violations = validator.validate_current_state()
    
    if violations:
        print("❌ 파일 위치 위반 발견:")
        for pattern, files in violations.items():
            print(f"\n패턴: {pattern}")
            for file_path in files:
                print(f"  - {file_path}")
                filename = Path(file_path).name
                suggested_location = validator.suggest_correct_location(filename)
                print(f"    → 제안 위치: {suggested_location}{filename}")
        
        if args.pre_commit:
            sys.exit(1)
    else:
        print("✅ 모든 파일이 올바른 위치에 있습니다.")
        if args.pre_commit:
            sys.exit(0)

if __name__ == "__main__":
    main() 