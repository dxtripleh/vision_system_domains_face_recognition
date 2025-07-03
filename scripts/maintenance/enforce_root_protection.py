#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
최상위 루트 보호 강제 스크립트

최상위에 허용되지 않은 파일들을 자동으로 정리하고 
향후 생성을 방지하는 시스템입니다.
"""

import os
import sys
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set

# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent.parent

class RootProtectionEnforcer:
    """최상위 루트 보호 강제 시스템"""
    
    def __init__(self):
        self.project_root = project_root
        
        # 허용된 최상위 파일들 (엄격한 제한)
        self.allowed_files = {
            'README.md',
            'requirements.txt', 
            'pytest.ini',
            '.gitignore',
            'launcher.py'  # 통합 런처만 허용
        }
        
        # 허용된 최상위 폴더들
        self.allowed_dirs = {
            'common',
            'config', 
            'data',
            'datasets',
            'docs',
            'domains',
            'models',
            'requirements',
            'scripts',
            'shared',
            'tools',
            '.cursor',  # IDE 폴더
            '.git'      # Git 폴더
        }
        
        # 자동 이동 규칙
        self.auto_move_rules = {
            # run_ 파일들
            'run_*.py': 'tools/legacy/',
            
            # 다운로드 스크립트
            'download_*.py': 'tools/setup/',
            
            # 상태 문서들
            '*_STATUS.md': 'docs/status/',
            'CURRENT_*.md': 'docs/status/',
            
            # 설정 파일들
            '*.yaml': 'config/',
            '*.json': 'config/',
            
            # 임시 파일들
            '*.tmp': 'data/runtime/temp/',
            '*.log': 'data/runtime/logs/',
            
            # 백업 파일들
            '*.bak': 'data/backups/',
            '*~': 'data/runtime/temp/'
        }
        
        # 삭제할 파일 패턴
        self.delete_patterns = {
            '*.pyc',
            '__pycache__',
            '.DS_Store',
            'Thumbs.db',
            '*.swp',
            '*.swo'
        }
    
    def enforce_protection(self, auto_fix: bool = True) -> Dict:
        """루트 보호 강제 실행"""
        print("🛡️  최상위 루트 보호 강제 실행")
        print("="*50)
        
        violations = self._scan_violations()
        
        if not violations['files'] and not violations['dirs']:
            print("✅ 최상위 루트가 깨끗합니다!")
            return {'status': 'clean', 'violations': violations}
        
        print(f"⚠️  {len(violations['files'])}개 파일, {len(violations['dirs'])}개 폴더 위반 발견")
        
        if auto_fix:
            self._auto_fix_violations(violations)
            return {'status': 'fixed', 'violations': violations}
        else:
            self._report_violations(violations)
            return {'status': 'reported', 'violations': violations}
    
    def _scan_violations(self) -> Dict[str, List[str]]:
        """위반 사항 스캔"""
        violations = {'files': [], 'dirs': []}
        
        for item in self.project_root.iterdir():
            if item.is_file():
                if item.name not in self.allowed_files:
                    violations['files'].append(item.name)
            elif item.is_dir():
                if item.name not in self.allowed_dirs:
                    violations['dirs'].append(item.name)
        
        return violations
    
    def _auto_fix_violations(self, violations: Dict[str, List[str]]):
        """위반 사항 자동 수정"""
        print("\n🔧 자동 수정 실행 중...")
        
        fixed_count = 0
        
        # 파일 처리
        for filename in violations['files']:
            file_path = self.project_root / filename
            if self._move_file_by_rules(file_path):
                fixed_count += 1
            elif self._should_delete_file(filename):
                self._delete_file(file_path)
                fixed_count += 1
        
        # 폴더 처리
        for dirname in violations['dirs']:
            dir_path = self.project_root / dirname
            if self._move_directory(dir_path):
                fixed_count += 1
        
        print(f"✅ {fixed_count}개 항목 수정 완료")
    
    def _move_file_by_rules(self, file_path: Path) -> bool:
        """규칙에 따라 파일 이동"""
        filename = file_path.name
        
        for pattern, dest_dir in self.auto_move_rules.items():
            if self._match_pattern(filename, pattern):
                dest_path = self.project_root / dest_dir
                dest_path.mkdir(parents=True, exist_ok=True)
                
                dest_file = dest_path / filename
                
                # 중복 파일 처리
                if dest_file.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    name_parts = filename.rsplit('.', 1)
                    if len(name_parts) == 2:
                        new_name = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
                    else:
                        new_name = f"{filename}_{timestamp}"
                    dest_file = dest_path / new_name
                
                try:
                    shutil.move(str(file_path), str(dest_file))
                    print(f"   📦 이동: {filename} → {dest_dir}")
                    return True
                except Exception as e:
                    print(f"   ❌ 이동 실패: {filename} - {str(e)}")
                    return False
        
        # 규칙에 맞지 않으면 legacy로 이동
        legacy_dir = self.project_root / 'tools' / 'legacy'
        legacy_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            dest_file = legacy_dir / filename
            if dest_file.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name_parts = filename.rsplit('.', 1)
                if len(name_parts) == 2:
                    new_name = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
                else:
                    new_name = f"{filename}_{timestamp}"
                dest_file = legacy_dir / new_name
            
            shutil.move(str(file_path), str(dest_file))
            print(f"   📦 레거시 이동: {filename} → tools/legacy/")
            return True
        except Exception as e:
            print(f"   ❌ 레거시 이동 실패: {filename} - {str(e)}")
            return False
    
    def _should_delete_file(self, filename: str) -> bool:
        """파일 삭제 여부 판단"""
        for pattern in self.delete_patterns:
            if self._match_pattern(filename, pattern):
                return True
        return False
    
    def _delete_file(self, file_path: Path):
        """파일 삭제"""
        try:
            file_path.unlink()
            print(f"   🗑️  삭제: {file_path.name}")
        except Exception as e:
            print(f"   ❌ 삭제 실패: {file_path.name} - {str(e)}")
    
    def _move_directory(self, dir_path: Path) -> bool:
        """디렉토리 이동"""
        dirname = dir_path.name
        
        # 임시 폴더들은 data/runtime/temp로
        if 'temp' in dirname.lower() or 'tmp' in dirname.lower():
            dest_dir = self.project_root / 'data' / 'runtime' / 'temp'
        # 백업 폴더들은 data/backups로
        elif 'backup' in dirname.lower() or 'bak' in dirname.lower():
            dest_dir = self.project_root / 'data' / 'backups'
        # 기타는 tools/legacy로
        else:
            dest_dir = self.project_root / 'tools' / 'legacy'
        
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / dirname
        
        # 중복 처리
        if dest_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_path = dest_dir / f"{dirname}_{timestamp}"
        
        try:
            shutil.move(str(dir_path), str(dest_path))
            print(f"   📁 이동: {dirname}/ → {dest_dir.relative_to(self.project_root)}/")
            return True
        except Exception as e:
            print(f"   ❌ 폴더 이동 실패: {dirname} - {str(e)}")
            return False
    
    def _match_pattern(self, filename: str, pattern: str) -> bool:
        """패턴 매칭"""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)
    
    def _report_violations(self, violations: Dict[str, List[str]]):
        """위반 사항 보고"""
        print("\n📋 위반 사항 보고:")
        
        if violations['files']:
            print(f"\n❌ 허용되지 않은 파일들 ({len(violations['files'])}개):")
            for filename in violations['files']:
                print(f"   - {filename}")
        
        if violations['dirs']:
            print(f"\n❌ 허용되지 않은 폴더들 ({len(violations['dirs'])}개):")
            for dirname in violations['dirs']:
                print(f"   - {dirname}/")
        
        print("\n💡 자동 수정하려면: --auto-fix 옵션을 사용하세요")
    
    def setup_monitoring(self):
        """지속적인 모니터링 설정"""
        print("👁️  지속적인 루트 보호 모니터링 설정")
        
        # Git pre-commit 훅 생성
        self._create_git_hook()
        
        # 주기적 검사 스크립트 생성
        self._create_periodic_checker()
    
    def _create_git_hook(self):
        """Git pre-commit 훅 생성"""
        git_hooks_dir = self.project_root / '.git' / 'hooks'
        if not git_hooks_dir.exists():
            print("   ⚠️  .git/hooks 폴더가 없습니다. Git 저장소인지 확인하세요.")
            return
        
        hook_content = """#!/bin/sh
# 최상위 루트 보호 pre-commit 훅

echo "🛡️  최상위 루트 보호 검사 중..."
python scripts/maintenance/enforce_root_protection.py --check-only

if [ $? -ne 0 ]; then
    echo "❌ 최상위 루트에 허용되지 않은 파일이 있습니다."
    echo "💡 자동 수정: python scripts/maintenance/enforce_root_protection.py --auto-fix"
    exit 1
fi

echo "✅ 최상위 루트 보호 검사 통과"
"""
        
        hook_file = git_hooks_dir / 'pre-commit'
        try:
            with open(hook_file, 'w', encoding='utf-8') as f:
                f.write(hook_content)
            
            # 실행 권한 부여 (Windows에서는 무시됨)
            if os.name != 'nt':
                os.chmod(hook_file, 0o755)
            
            print(f"   ✅ Git pre-commit 훅 생성: {hook_file}")
        except Exception as e:
            print(f"   ❌ Git 훅 생성 실패: {str(e)}")
    
    def _create_periodic_checker(self):
        """주기적 검사 스크립트 생성"""
        checker_content = f"""#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
주기적 루트 보호 검사기

이 스크립트를 작업 스케줄러나 cron에 등록하여
주기적으로 최상위 루트를 보호할 수 있습니다.
'''

import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.maintenance.enforce_root_protection import RootProtectionEnforcer

if __name__ == "__main__":
    enforcer = RootProtectionEnforcer()
    result = enforcer.enforce_protection(auto_fix=True)
    
    if result['status'] != 'clean':
        print("⚠️  루트 보호 위반이 발견되어 자동 수정했습니다.")
        # 필요시 알림 시스템 연동 가능
"""
        
        checker_file = self.project_root / 'scripts' / 'maintenance' / 'periodic_root_checker.py'
        try:
            with open(checker_file, 'w', encoding='utf-8') as f:
                f.write(checker_content)
            print(f"   ✅ 주기적 검사기 생성: {checker_file}")
        except Exception as e:
            print(f"   ❌ 주기적 검사기 생성 실패: {str(e)}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="최상위 루트 보호 강제")
    parser.add_argument("--auto-fix", action="store_true", help="위반 사항 자동 수정")
    parser.add_argument("--check-only", action="store_true", help="검사만 수행 (수정 안함)")
    parser.add_argument("--setup-monitoring", action="store_true", help="지속적 모니터링 설정")
    args = parser.parse_args()
    
    enforcer = RootProtectionEnforcer()
    
    if args.setup_monitoring:
        enforcer.setup_monitoring()
        return
    
    if args.check_only:
        result = enforcer.enforce_protection(auto_fix=False)
        sys.exit(0 if result['status'] == 'clean' else 1)
    
    result = enforcer.enforce_protection(auto_fix=args.auto_fix)
    
    if result['status'] == 'clean':
        print("\n🎉 최상위 루트가 완벽하게 보호되고 있습니다!")
    elif result['status'] == 'fixed':
        print("\n🎉 모든 위반 사항이 자동으로 수정되었습니다!")
    else:
        print("\n⚠️  위반 사항이 발견되었습니다. --auto-fix로 자동 수정하세요.")

if __name__ == "__main__":
    main() 