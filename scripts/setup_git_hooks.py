#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Git 훅 자동 설정 스크립트 (setup_git_hooks.py)

개발 규칙 준수를 위한 Git 훅을 자동으로 설정합니다.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Git 훅 설정
GIT_HOOKS = {
    'pre-commit': {
        'checks': ['validate_rules', 'code_style_check', 'docstring_check'],
        'path': 'scripts/git_hooks/pre-commit',
        'description': '커밋 전 규칙 검증'
    },
    'pre-push': {
        'checks': ['validate_rules', 'security_check'],
        'path': 'scripts/git_hooks/pre-push',
        'description': '푸시 전 보안 검증'
    },
    'commit-msg': {
        'checks': ['check_commit_message'],
        'path': 'scripts/git_hooks/commit-msg',
        'description': '커밋 메시지 검증'
    }
}

def check_git_repository() -> bool:
    """Git 저장소인지 확인"""
    try:
        result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                              capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_git_hooks_dir() -> Path:
    """Git 훅 디렉토리 경로 반환"""
    try:
        result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                              capture_output=True, text=True, check=True)
        git_dir = Path(result.stdout.strip())
        return git_dir / 'hooks'
    except subprocess.CalledProcessError:
        return Path('.git/hooks')

def create_pre_commit_hook(hooks_dir: Path):
    """pre-commit 훅 생성"""
    hook_content = '''#!/bin/bash
# pre-commit 훅 - 커밋 전 규칙 검증

echo "🔍 커밋 전 규칙 검증 중..."

# 1. 파이프라인 구조 검증
echo "  - 파이프라인 구조 검증..."
python scripts/validate_pipeline.py humanoid face_recognition
if [ $? -ne 0 ]; then
    echo "❌ 파이프라인 구조 검증 실패"
    exit 1
fi

# 2. 파일명 패턴 검증
echo "  - 파일명 패턴 검증..."
python scripts/validate_filenames.py humanoid face_recognition
if [ $? -ne 0 ]; then
    echo "❌ 파일명 패턴 검증 실패"
    exit 1
fi

# 3. 추적성 검증
echo "  - 추적성 검증..."
python scripts/manage_traceability.py humanoid face_recognition verify
if [ $? -ne 0 ]; then
    echo "❌ 추적성 검증 실패"
    exit 1
fi

# 4. 코드 스타일 검사 (선택적)
if command -v black &> /dev/null; then
    echo "  - 코드 스타일 검사..."
    black --check --diff .
    if [ $? -ne 0 ]; then
        echo "❌ 코드 스타일 검사 실패 (black --check . 로 수정 가능)"
        exit 1
    fi
fi

echo "✅ 모든 검증 통과!"
exit 0
'''
    
    hook_file = hooks_dir / 'pre-commit'
    hook_file.write_text(hook_content, encoding='utf-8')
    hook_file.chmod(0o755)  # 실행 권한 부여
    
    print(f" pre-commit 훅 생성 완료: {hook_file}")

def create_pre_push_hook(hooks_dir: Path):
    """pre-push 훅 생성"""
    hook_content = '''#!/bin/bash
# pre-push 훅 - 푸시 전 보안 검증

echo "🔒 푸시 전 보안 검증 중..."

# 1. 민감 정보 검사
echo "  - 민감 정보 검사..."
if grep -r "password\|secret\|key\|token" --include="*.py" --include="*.yaml" --include="*.json" . | grep -v "#.*password\|#.*secret"; then
    echo "❌ 민감 정보가 포함된 파일이 있습니다"
    exit 1
fi

# 2. 하드코딩된 경로 검사
echo "  - 하드코딩된 경로 검사..."
if grep -r "C:\\\\\|/home/\|/Users/" --include="*.py" . | grep -v "#.*C:\\\\\|#.*/home/\|#.*/Users/"; then
    echo "❌ 하드코딩된 경로가 있습니다"
    exit 1
fi

# 3. 시뮬레이션 모드 검사
echo "  - 시뮬레이션 모드 검사..."
if grep -r "USE_SIMULATION.*True\|dummy\|mock" --include="*.py" . | grep -v "#.*USE_SIMULATION\|#.*dummy\|#.*mock"; then
    echo "❌ 시뮬레이션 모드가 활성화되어 있습니다"
    exit 1
fi

echo "✅ 보안 검증 통과!"
exit 0
'''
    
    hook_file = hooks_dir / 'pre-push'
    hook_file.write_text(hook_content, encoding='utf-8')
    hook_file.chmod(0o755)
    
    print(f" pre-push 훅 생성 완료: {hook_file}")

def create_commit_msg_hook(hooks_dir: Path):
    """commit-msg 훅 생성"""
    hook_content = '''#!/bin/bash
# commit-msg 훅 - 커밋 메시지 검증

echo "📝 커밋 메시지 검증 중..."

# 커밋 메시지 파일 경로
commit_msg_file="$1"

# 커밋 메시지 읽기
commit_msg=$(cat "$commit_msg_file")

# 1. 최소 길이 검사
if [ ${#commit_msg} -lt 10 ]; then
    echo "❌ 커밋 메시지가 너무 짧습니다 (최소 10자)"
    exit 1
fi

# 2. 한국어 포함 검사
if ! echo "$commit_msg" | grep -q "[가-힣]"; then
    echo "❌ 한국어 설명이 포함되어야 합니다"
    exit 1
fi

# 3. 금지된 단어 검사
forbidden_words=("test" "temp" "debug" "fix" "update")
for word in "${forbidden_words[@]}"; do
    if echo "$commit_msg" | grep -qi "^$word"; then
        echo "❌ 금지된 단어로 시작합니다: $word"
        echo "   예시: 'test' → '테스트 코드 추가'"
        exit 1
    fi
done

echo "✅ 커밋 메시지 검증 통과!"
exit 0
'''
    
    hook_file = hooks_dir / 'commit-msg'
    hook_file.write_text(hook_content, encoding='utf-8')
    hook_file.chmod(0o755)
    
    print(f" commit-msg 훅 생성 완료: {hook_file}")

def install_git_hook(hook_name: str, hook_config: Dict):
    """Git 훅 설치"""
    hooks_dir = get_git_hooks_dir()
    
    if hook_name == 'pre-commit':
        create_pre_commit_hook(hooks_dir)
    elif hook_name == 'pre-push':
        create_pre_push_hook(hooks_dir)
    elif hook_name == 'commit-msg':
        create_commit_msg_hook(hooks_dir)
    else:
        print(f" 알 수 없는 훅: {hook_name}")

def setup_git_hooks():
    """Git 훅 자동 설치"""
    print("🔧 Git 훅 자동 설정 시작...")
    
    # Git 저장소 확인
    if not check_git_repository():
        print("❌ Git 저장소가 아닙니다. git init을 먼저 실행하세요.")
        return False
    
    # 훅 디렉토리 생성
    hooks_dir = get_git_hooks_dir()
    hooks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f" 훅 디렉토리: {hooks_dir}")
    
    # 각 훅 설치
    for hook_name, hook_config in GIT_HOOKS.items():
        print(f" {hook_name} 훅 설치 중...")
        install_git_hook(hook_name, hook_config)
    
    print("✅ Git 훅 설정 완료!")
    print()
    print("📋 설정된 훅:")
    for hook_name, hook_config in GIT_HOOKS.items():
        print(f"  - {hook_name}: {hook_config['description']}")
    
    return True

def remove_git_hooks():
    """Git 훅 제거"""
    print("🗑️ Git 훅 제거 중...")
    
    hooks_dir = get_git_hooks_dir()
    
    for hook_name in GIT_HOOKS.keys():
        hook_file = hooks_dir / hook_name
        if hook_file.exists():
            hook_file.unlink()
            print(f" {hook_name} 훅 제거 완료")
    
    print("✅ Git 훅 제거 완료!")

def test_git_hooks():
    """Git 훅 테스트"""
    print("🧪 Git 훅 테스트 중...")
    
    hooks_dir = get_git_hooks_dir()
    
    for hook_name in GIT_HOOKS.keys():
        hook_file = hooks_dir / hook_name
        if hook_file.exists():
            print(f" ✅ {hook_name}: 존재함")
        else:
            print(f" ❌ {hook_name}: 없음")

def main():
    parser = argparse.ArgumentParser(description="Git 훅 자동 설정 스크립트")
    parser.add_argument("--remove", action="store_true", help="Git 훅 제거")
    parser.add_argument("--test", action="store_true", help="Git 훅 테스트")
    
    args = parser.parse_args()
    
    if args.remove:
        remove_git_hooks()
    elif args.test:
        test_git_hooks()
    else:
        setup_git_hooks()

if __name__ == "__main__":
    main()
