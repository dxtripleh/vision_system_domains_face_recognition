#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
파일 보호 시스템 설정 스크립트.

개발 중 최상위 루트에 임시 파일이 생성되는 것을 방지하는 시스템을 설정합니다.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from common.logging import setup_logging

logger = logging.getLogger(__name__)

def setup_file_protection():
    """파일 보호 시스템 설정"""
    print("🔒 파일 보호 시스템 설정 중...")
    
    # 1. Git pre-commit 훅 설정
    setup_git_hooks()
    
    # 2. IDE 설정 파일 생성
    setup_ide_configs()
    
    # 3. 개발 도구 설정
    setup_development_tools()
    
    # 4. 파일 위치 검증 스크립트 실행 권한 설정
    setup_script_permissions()
    
    print("✅ 파일 보호 시스템 설정 완료!")

def setup_git_hooks():
    """Git 훅 설정"""
    print("📝 Git pre-commit 훅 설정...")
    
    try:
        # 파일 위치 검증 훅 생성
        result = subprocess.run([
            sys.executable, 
            "scripts/validate_file_locations.py", 
            "--create-hook"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Git pre-commit 훅 생성 완료")
        else:
            print(f"⚠️ Git 훅 생성 실패: {result.stderr}")
            
    except Exception as e:
        print(f"⚠️ Git 훅 설정 오류: {str(e)}")

def setup_ide_configs():
    """IDE 설정 파일 생성"""
    print("⚙️ IDE 설정 파일 생성...")
    
    # VS Code 설정
    vscode_dir = project_root / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    
    # VS Code 설정 파일
    vscode_settings = {
        "python.defaultInterpreterPath": "./venv/bin/python",
        "python.linting.enabled": True,
        "python.linting.pylintEnabled": True,
        "python.formatting.provider": "black",
        "python.sortImports.args": ["--profile", "black"],
        "files.exclude": {
            "**/__pycache__": True,
            "**/*.pyc": True,
            "**/*.log": True,
            "**/*.tmp": True,
            "**/temp_backup": True
        },
        "search.exclude": {
            "**/data/temp": True,
            "**/data/logs": True,
            "**/models/weights": True,
            "**/datasets": True
        }
    }
    
    import json
    with open(vscode_dir / "settings.json", 'w', encoding='utf-8') as f:
        json.dump(vscode_settings, f, indent=2, ensure_ascii=False)
    
    # VS Code launch.json (디버깅 설정)
    launch_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Current File",
                "type": "python",
                "request": "launch",
                "program": "${file}",
                "console": "integratedTerminal",
                "cwd": "${workspaceFolder}",
                "env": {
                    "PYTHONPATH": "${workspaceFolder}"
                }
            },
            {
                "name": "Face Recognition",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/domains/humanoid/face_recognition/run_face_recognition.py",
                "console": "integratedTerminal",
                "cwd": "${workspaceFolder}",
                "env": {
                    "PYTHONPATH": "${workspaceFolder}"
                }
            },
            {
                "name": "Defect Detection",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/domains/factory/defect_detection/run_defect_detection.py",
                "console": "integratedTerminal",
                "cwd": "${workspaceFolder}",
                "env": {
                    "PYTHONPATH": "${workspaceFolder}"
                }
            }
        ]
    }
    
    with open(vscode_dir / "launch.json", 'w', encoding='utf-8') as f:
        json.dump(launch_config, f, indent=2, ensure_ascii=False)
    
    print("✅ VS Code 설정 완료")

def setup_development_tools():
    """개발 도구 설정"""
    print("🛠️ 개발 도구 설정...")
    
    # pre-commit 설정
    pre_commit_config = {
        "repos": [
            {
                "repo": "https://github.com/psf/black",
                "rev": "23.3.0",
                "hooks": [
                    {
                        "id": "black",
                        "language_version": "python3"
                    }
                ]
            },
            {
                "repo": "https://github.com/pycqa/isort",
                "rev": "5.12.0",
                "hooks": [
                    {
                        "id": "isort"
                    }
                ]
            },
            {
                "repo": "https://github.com/pycqa/flake8",
                "rev": "6.0.0",
                "hooks": [
                    {
                        "id": "flake8"
                    }
                ]
            },
            {
                "repo": "local",
                "hooks": [
                    {
                        "id": "file-location-validator",
                        "name": "File Location Validator",
                        "entry": "python scripts/validate_file_locations.py --pre-commit",
                        "language": "system",
                        "pass_filenames": False
                    }
                ]
            }
        ]
    }
    
    import yaml
    with open(project_root / ".pre-commit-config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(pre_commit_config, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ pre-commit 설정 완료")

def setup_script_permissions():
    """스크립트 실행 권한 설정"""
    print("🔧 스크립트 권한 설정...")
    
    scripts_to_make_executable = [
        "scripts/validate_file_locations.py",
        "scripts/setup_file_protection.py"
    ]
    
    for script_path in scripts_to_make_executable:
        script_file = project_root / script_path
        if script_file.exists():
            try:
                os.chmod(script_file, 0o755)
                print(f"✅ {script_path} 실행 권한 설정 완료")
            except Exception as e:
                print(f"⚠️ {script_path} 권한 설정 실패: {str(e)}")

def create_file_watcher():
    """파일 감시자 생성 (선택적)"""
    print("👁️ 파일 감시자 설정...")
    
    watcher_script = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
파일 감시자 스크립트.

실시간으로 파일 생성을 모니터링하고 잘못된 위치의 파일을 감지합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from scripts.validate_file_locations import FileLocationValidator

def main():
    """메인 함수"""
    validator = FileLocationValidator(project_root)
    validator.start_monitoring()
    
    try:
        print("파일 감시자 시작... (Ctrl+C로 종료)")
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\n파일 감시자 종료")
        validator.stop_monitoring()

if __name__ == "__main__":
    main()
'''
    
    watcher_path = project_root / "scripts" / "watch_files.py"
    with open(watcher_path, 'w', encoding='utf-8') as f:
        f.write(watcher_script)
    
    # 실행 권한 부여
    os.chmod(watcher_path, 0o755)
    print("✅ 파일 감시자 생성 완료")

def create_development_guide():
    """개발 가이드 생성"""
    print("📚 개발 가이드 생성...")
    
    guide_content = """# 파일 위치 규칙 개발 가이드

## 🚫 금지 사항

### 절대 금지: 최상위 루트에 임시 파일 생성
```python
# ❌ 잘못된 예시
cv2.imwrite("captured_frame.jpg", frame)  # 루트에 저장
with open("debug.log", "w") as f:         # 루트에 저장
    f.write("debug info")

# ✅ 올바른 예시
from common.file_utils import save_image, save_json
save_image(frame, "captured_frame.jpg")   # data/output/에 저장
save_json({"debug": "info"}, "debug.log") # data/temp/에 저장
```

## 📁 올바른 파일 위치

### 이미지/비디오 파일
- `data/output/` - 최종 결과물
- `data/temp/` - 임시 파일

### 로그 파일
- `data/logs/` - 시스템 로그
- `data/temp/` - 임시 로그

### 데이터 파일
- `data/temp/` - 임시 데이터
- `data/domains/{domain}/` - 도메인별 데이터

### 모델 파일
- `models/weights/` - 모델 가중치 (.onnx)

## 🛠️ 사용법

### 1. 파일 저장 유틸리티 사용
```python
from common.file_utils import save_image, save_json, save_csv

# 이미지 저장
save_image(frame, "result.jpg")

# JSON 저장
save_json({"result": "success"}, "output.json")

# CSV 저장
save_csv(data_list, "results.csv")
```

### 2. 임시 파일 생성
```python
from common.file_utils import create_temp_file

temp_file = create_temp_file("debug", ".log")
```

### 3. 파일 위치 검증
```python
# 수동 검증
python scripts/validate_file_locations.py

# 실시간 모니터링
python scripts/validate_file_locations.py --monitor

# Git 훅 설정
python scripts/validate_file_locations.py --create-hook
```

## 🔧 개발 환경 설정

### VS Code 설정
- `.vscode/settings.json` - 파일 제외 설정
- `.vscode/launch.json` - 디버깅 설정

### pre-commit 훅
- 자동 코드 포맷팅 (black, isort)
- 파일 위치 검증
- 코드 품질 검사 (flake8)

### 파일 감시자
```bash
# 실시간 파일 생성 모니터링
python scripts/watch_files.py
```

## ⚠️ 주의사항

1. **절대 금지**: 하드코딩된 경로 사용
2. **절대 금지**: print() 대신 logger 사용
3. **절대 금지**: 예외 처리 없는 파일 저장
4. **권장**: common.file_utils 모듈 사용

## 🆘 문제 해결

### 파일 위치 위반 시
1. `python scripts/validate_file_locations.py` 실행
2. 제안된 올바른 위치로 파일 이동
3. 코드에서 common.file_utils 사용하도록 수정

### 자동 파일 이동
```bash
export AUTO_MOVE_FILES=true
python scripts/validate_file_locations.py --monitor
```
"""
    
    guide_path = project_root / "docs" / "file_location_guide.md"
    guide_path.parent.mkdir(exist_ok=True)
    
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print("✅ 개발 가이드 생성 완료")

def main():
    """메인 함수"""
    print("🔒 Vision System 파일 보호 시스템 설정")
    print("=" * 50)
    
    try:
        # 기본 설정
        setup_file_protection()
        
        # 추가 기능 (선택적)
        create_file_watcher()
        create_development_guide()
        
        print("\n🎉 설정 완료!")
        print("\n📋 다음 단계:")
        print("1. pre-commit 설치: pip install pre-commit")
        print("2. pre-commit 설정: pre-commit install")
        print("3. 파일 감시자 실행: python scripts/watch_files.py")
        print("4. 개발 가이드 확인: docs/file_location_guide.md")
        
    except Exception as e:
        print(f"❌ 설정 실패: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 