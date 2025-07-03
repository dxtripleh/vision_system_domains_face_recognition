#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
프로젝트 구조 자동 검증 및 문서 생성 시스템.

최상위 루트 관리 규칙을 자동으로 검증하고,
누락된 README/STRUCTURE 파일을 자동 생성합니다.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 허용되는 최상위 파일들
ALLOWED_ROOT_FILES = {
    "README.md": "프로젝트 개요",
    "launcher.py": "통합 런처", 
    "requirements.txt": "의존성 정의",
    ".gitignore": "Git 제외 파일",
    "pytest.ini": "테스트 설정",
    "PROJECT_ROOT_RULES.md": "루트 관리 규칙"  # 임시 허용
}

# 허용되는 최상위 폴더들
ALLOWED_ROOT_DIRS = {
    "domains": "도메인별 비즈니스 로직",
    "shared": "공유 모듈 (비전 코어, 보안)",
    "common": "범용 유틸리티", 
    "config": "전역 설정",
    "models": "AI 모델 저장소",
    "datasets": "학습 데이터셋",
    "data": "런타임 데이터",
    "scripts": "개발 스크립트",
    "requirements": "환경별 의존성",
    "docs": "문서 저장소",
    "tools": "시스템 도구",
    ".cursor": "Cursor IDE 설정 (허용)"
}

class ProjectStructureValidator:
    """프로젝트 구조 검증기"""
    
    def __init__(self):
        self.violations = []
        self.missing_docs = []
        self.project_root = PROJECT_ROOT
        
    def validate_root_structure(self) -> Dict:
        """최상위 구조 검증"""
        print("🔍 최상위 루트 구조 검증 중...")
        
        violations = []
        
        # 최상위 파일들 검증
        for item in self.project_root.iterdir():
            if item.is_file():
                if item.name not in ALLOWED_ROOT_FILES:
                    violations.append({
                        'type': 'forbidden_file',
                        'path': str(item),
                        'name': item.name,
                        'suggestion': self._suggest_file_location(item.name)
                    })
            elif item.is_dir():
                if item.name not in ALLOWED_ROOT_DIRS:
                    violations.append({
                        'type': 'forbidden_dir',
                        'path': str(item),
                        'name': item.name,
                        'suggestion': self._suggest_dir_location(item.name)
                    })
        
        self.violations.extend(violations)
        
        return {
            'violations': violations,
            'allowed_files': len([f for f in self.project_root.iterdir() if f.is_file() and f.name in ALLOWED_ROOT_FILES]),
            'forbidden_files': len([v for v in violations if v['type'] == 'forbidden_file']),
            'forbidden_dirs': len([v for v in violations if v['type'] == 'forbidden_dir'])
        }
    
    def _suggest_file_location(self, filename: str) -> str:
        """파일에 대한 권장 위치 제안"""
        if filename.startswith('run_'):
            return 'domains/{domain}/runners/'
        elif filename.startswith('test_'):
            return 'tests/'
        elif filename.endswith('_STATUS.md'):
            return 'docs/status/'
        elif filename.endswith('_GUIDE.md'):
            return 'docs/guides/'
        elif filename.startswith('download_'):
            return 'tools/setup/'
        elif filename.startswith('setup_'):
            return 'tools/setup/'
        elif filename.startswith('validate_'):
            return 'tools/validation/'
        elif filename == 'main.py':
            return 'tools/legacy/'
        else:
            return 'tools/misc/'
    
    def _suggest_dir_location(self, dirname: str) -> str:
        """폴더에 대한 권장 위치 제안"""
        if dirname in ['temp', 'tmp']:
            return 'data/temp/'
        elif dirname in ['logs', 'log']:
            return 'data/logs/'
        elif dirname in ['backup', 'backups']:
            return 'data/backups/'
        else:
            return 'tools/misc/'
    
    def check_missing_documentation(self) -> List[Dict]:
        """누락된 문서 검사"""
        print("📝 누락된 문서 검사 중...")
        
        missing = []
        
        # 최상위 폴더들의 문서 검사
        for dir_name in ALLOWED_ROOT_DIRS.keys():
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                
                # README.md 검사
                readme_path = dir_path / "README.md"
                if not readme_path.exists():
                    missing.append({
                        'type': 'missing_readme',
                        'path': str(dir_path),
                        'required_file': 'README.md',
                        'description': ALLOWED_ROOT_DIRS[dir_name]
                    })
                
                # STRUCTURE.md 검사 (하위 폴더가 3개 이상인 경우)
                subdirs = [item for item in dir_path.iterdir() if item.is_dir()]
                if len(subdirs) >= 3:
                    structure_path = dir_path / "STRUCTURE.md"
                    if not structure_path.exists():
                        missing.append({
                            'type': 'missing_structure',
                            'path': str(dir_path),
                            'required_file': 'STRUCTURE.md',
                            'subdirs_count': len(subdirs)
                        })
        
        self.missing_docs = missing
        return missing
    
    def auto_fix_violations(self) -> Dict:
        """규칙 위반 자동 수정"""
        print("🔧 규칙 위반 자동 수정 중...")
        
        fixed = []
        failed = []
        
        for violation in self.violations:
            try:
                if violation['type'] == 'forbidden_file':
                    # 파일 이동
                    source = Path(violation['path'])
                    dest_dir = Path(violation['suggestion'])
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    dest_path = dest_dir / source.name
                    source.rename(dest_path)
                    
                    fixed.append({
                        'type': 'moved_file',
                        'from': str(source),
                        'to': str(dest_path)
                    })
                    
                elif violation['type'] == 'forbidden_dir':
                    # 폴더 이동 (위험하므로 로그만)
                    failed.append({
                        'type': 'dir_move_skipped',
                        'path': violation['path'],
                        'reason': '폴더 이동은 수동으로 처리 필요'
                    })
                    
            except Exception as e:
                failed.append({
                    'type': 'fix_failed',
                    'path': violation['path'],
                    'error': str(e)
                })
        
        return {'fixed': fixed, 'failed': failed}
    
    def generate_missing_documentation(self) -> Dict:
        """누락된 문서 자동 생성"""
        print("📚 누락된 문서 자동 생성 중...")
        
        generated = []
        failed = []
        
        for missing in self.missing_docs:
            try:
                if missing['type'] == 'missing_readme':
                    self._generate_readme(missing)
                    generated.append(missing['path'] + '/README.md')
                    
                elif missing['type'] == 'missing_structure':
                    self._generate_structure(missing)
                    generated.append(missing['path'] + '/STRUCTURE.md')
                    
            except Exception as e:
                failed.append({
                    'path': missing['path'],
                    'error': str(e)
                })
        
        return {'generated': generated, 'failed': failed}
    
    def _generate_readme(self, missing_info: Dict):
        """README.md 자동 생성"""
        dir_path = Path(missing_info['path'])
        dir_name = dir_path.name
        
        # 폴더 내용 분석
        files = [f.name for f in dir_path.iterdir() if f.is_file()]
        dirs = [d.name for d in dir_path.iterdir() if d.is_dir()]
        
        readme_content = f"""# 📁 {dir_name.upper()} 폴더

## 🎯 **목적**
{missing_info.get('description', '이 폴더의 목적을 설명합니다.')}

## 📂 **구조**
```
{dir_name}/
"""
        
        # 하위 항목들 추가
        for d in sorted(dirs):
            readme_content += f"├── {d}/\n"
        for f in sorted(files):
            readme_content += f"├── {f}\n"
            
        readme_content += """```

## 🚀 **사용법**
이 폴더의 주요 기능과 사용 방법을 설명합니다.

### 주요 기능
- 기능 1: 설명
- 기능 2: 설명
- 기능 3: 설명

### 사용 예시
```bash
# 사용 예시를 여기에 작성
```

## 📝 **주의사항**
- 주의사항 1
- 주의사항 2
- 주의사항 3

## 🔗 **관련 문서**
- [프로젝트 개요](../README.md)
- [구조 문서](STRUCTURE.md) (존재하는 경우)

---
*이 문서는 자동 생성되었습니다. 필요에 따라 내용을 수정하세요.*
"""
        
        readme_path = dir_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def _generate_structure(self, missing_info: Dict):
        """STRUCTURE.md 자동 생성"""
        dir_path = Path(missing_info['path'])
        dir_name = dir_path.name
        
        structure_content = f"""# 🏗️ {dir_name.upper()} 폴더 구조

## 📊 **전체 구조**
```
{dir_name}/
"""
        
        # 재귀적으로 구조 생성
        def add_structure(path: Path, prefix: str = ""):
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                
                if item.is_dir():
                    structure_content_ref.append(f"{prefix}{current_prefix}{item.name}/")
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    # 깊이 제한 (2레벨까지만)
                    if len(prefix) < 8:
                        add_structure(item, next_prefix)
                else:
                    structure_content_ref.append(f"{prefix}{current_prefix}{item.name}")
        
        structure_content_ref = [structure_content]
        add_structure(dir_path)
        structure_content = "\n".join(structure_content_ref)
        
        structure_content += """
```

## 📝 **폴더별 설명**

### 주요 하위 폴더들
"""
        
        # 하위 폴더들 설명 추가
        subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
        for subdir in sorted(subdirs):
            structure_content += f"""
#### `{subdir.name}/`
- **목적**: [폴더 목적 설명]
- **주요 파일**: [주요 파일들 나열]
- **사용법**: [사용 방법 간단 설명]
"""
        
        structure_content += """
## 🔄 **파일 흐름**
1. 입력 → 처리 → 출력 흐름 설명
2. 주요 처리 단계별 설명
3. 데이터 변환 과정 설명

## 📋 **개발 가이드라인**
- 새 파일 추가 시 규칙
- 네이밍 컨벤션
- 폴더 구조 유지 방법

---
*이 문서는 자동 생성되었습니다. 필요에 따라 내용을 수정하세요.*
"""
        
        structure_path = dir_path / "STRUCTURE.md"
        with open(structure_path, 'w', encoding='utf-8') as f:
            f.write(structure_content)
    
    def generate_report(self) -> str:
        """검증 결과 보고서 생성"""
        report = f"""# 🔍 프로젝트 구조 검증 보고서

## 📅 **검증 일시**
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 **검증 결과 요약**
- 총 위반 사항: {len(self.violations)}개
- 누락된 문서: {len(self.missing_docs)}개
- 검증 상태: {'✅ 통과' if len(self.violations) == 0 else '❌ 위반 발견'}

## 🚨 **규칙 위반 사항**
"""
        
        if self.violations:
            for violation in self.violations:
                report += f"""
### {violation['type']}
- **파일/폴더**: `{violation['name']}`
- **위치**: `{violation['path']}`
- **권장 이동**: `{violation['suggestion']}`
"""
        else:
            report += "✅ 규칙 위반 사항이 없습니다.\n"
        
        report += f"""
## 📝 **누락된 문서**
"""
        
        if self.missing_docs:
            for missing in self.missing_docs:
                report += f"""
### {missing['type']}
- **폴더**: `{missing['path']}`
- **필요 파일**: `{missing['required_file']}`
"""
        else:
            report += "✅ 모든 필수 문서가 존재합니다.\n"
        
        return report

def main():
    """메인 실행 함수"""
    print("🔧 프로젝트 구조 검증 및 문서 생성 시작")
    print("=" * 60)
    
    validator = ProjectStructureValidator()
    
    # 1. 구조 검증
    validation_result = validator.validate_root_structure()
    print(f"  📊 허용된 파일: {validation_result['allowed_files']}개")
    print(f"  🚨 금지된 파일: {validation_result['forbidden_files']}개")
    print(f"  🚨 금지된 폴더: {validation_result['forbidden_dirs']}개")
    
    # 2. 문서 검사
    missing_docs = validator.check_missing_documentation()
    print(f"  📝 누락된 문서: {len(missing_docs)}개")
    
    # 3. 자동 수정 (선택적)
    if validation_result['forbidden_files'] > 0:
        print("\n🔧 자동 수정 실행 중...")
        fix_result = validator.auto_fix_violations()
        print(f"  ✅ 수정 완료: {len(fix_result['fixed'])}개")
        print(f"  ❌ 수정 실패: {len(fix_result['failed'])}개")
    
    # 4. 문서 자동 생성
    if len(missing_docs) > 0:
        print("\n📚 문서 자동 생성 중...")
        doc_result = validator.generate_missing_documentation()
        print(f"  ✅ 생성 완료: {len(doc_result['generated'])}개")
        print(f"  ❌ 생성 실패: {len(doc_result['failed'])}개")
    
    # 5. 보고서 생성
    report = validator.generate_report()
    report_path = PROJECT_ROOT / "docs" / "status" / "STRUCTURE_VALIDATION_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📋 검증 보고서 생성: {report_path}")
    print("\n" + "=" * 60)
    print("✅ 프로젝트 구조 검증 및 문서 생성 완료!")

if __name__ == "__main__":
    main() 