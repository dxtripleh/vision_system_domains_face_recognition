# 🛠️ 프로젝트 구조 수동 정리 가이드

## 📋 현재 문제점
- 최상위 루트에 실행 파일들이 너무 많음
- `data/storage`에 도메인별 데이터가 혼재
- 올바른 도메인 주도 설계(DDD) 구조 필요

## 🎯 정리 단계

### 1단계: 디렉토리 생성
```bash
# Windows PowerShell에서 실행
mkdir domains\face_recognition\data\storage\faces
mkdir domains\face_recognition\data\storage\persons
mkdir domains\face_recognition\runners\demos
mkdir tools\setup
```

### 2단계: 파일 이동

#### 데모 파일들을 도메인 내부로 이동
```bash
move run_simple_demo.py domains\face_recognition\runners\demos\
move run_face_recognition_demo.py domains\face_recognition\runners\demos\
move run_face_registration.py domains\face_recognition\runners\data_collection\
```

#### 도구들을 tools 폴더로 이동
```bash
move download_models.py tools\setup\
```

### 3단계: 데이터 저장소 이동
```bash
# data/storage의 내용을 domains/face_recognition/data/storage로 이동
xcopy data\storage\faces domains\face_recognition\data\storage\faces /E /I
xcopy data\storage\persons domains\face_recognition\data\storage\persons /E /I

# 원본 삭제 (확인 후)
rmdir data\storage\faces /S /Q
rmdir data\storage\persons /S /Q
```

### 4단계: Import 경로 수정

이동된 파일들의 import 경로를 수정해야 합니다:

#### domains/face_recognition/runners/demos/run_simple_demo.py
```python
# 변경 전
project_root = Path(__file__).parent

# 변경 후  
project_root = Path(__file__).parent.parent.parent.parent.parent
```

#### domains/face_recognition/runners/demos/run_face_recognition_demo.py
```python
# 변경 전
project_root = Path(__file__).parent

# 변경 후
project_root = Path(__file__).parent.parent.parent.parent.parent
```

#### domains/face_recognition/runners/data_collection/run_face_registration.py
```python
# 변경 전
project_root = Path(__file__).parent

# 변경 후
project_root = Path(__file__).parent.parent.parent.parent.parent
```

### 5단계: 도메인 저장소 설정 생성

`domains/face_recognition/config/storage_config.py` 파일 생성:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition Domain Storage Configuration.

얼굴인식 도메인의 데이터 저장소 설정입니다.
"""

from pathlib import Path

# 도메인 루트 경로
DOMAIN_ROOT = Path(__file__).parent.parent

# 데이터 저장 경로들
STORAGE_PATHS = {
    "faces": DOMAIN_ROOT / "data" / "storage" / "faces",
    "persons": DOMAIN_ROOT / "data" / "storage" / "persons",
    "temp": DOMAIN_ROOT / "data" / "temp",
    "logs": DOMAIN_ROOT / "data" / "logs",
    "models": DOMAIN_ROOT / "models",
    "configs": DOMAIN_ROOT / "config"
}

def get_storage_path(storage_type: str) -> Path:
    """저장소 타입별 경로 반환"""
    return STORAGE_PATHS.get(storage_type, STORAGE_PATHS["temp"])

def ensure_directories():
    """필요한 디렉토리들 생성"""
    for path in STORAGE_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("✅ Face Recognition 도메인 저장소 디렉토리 생성 완료")
```

### 6단계: Repository 클래스 수정

`domains/face_recognition/core/repositories/face_repository.py`와 `person_repository.py`에서 저장소 경로를 새로운 설정으로 변경:

```python
# 추가
from domains.face_recognition.config.storage_config import get_storage_path

class FaceRepository:
    def __init__(self):
        # 변경 전
        # self.storage_path = Path("data/storage/faces")
        
        # 변경 후
        self.storage_path = get_storage_path("faces")
```

### 7단계: 통합 런처 생성

최상위에 `launcher.py` 하나만 생성:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vision System Project Launcher.
"""

def main():
    print("🎯 Vision System Project Launcher")
    print("=" * 50)
    print("사용 가능한 명령:")
    print()
    print("📊 얼굴인식 (Face Recognition)")
    print("  python domains/face_recognition/runners/demos/run_simple_demo.py")
    print("  python domains/face_recognition/runners/demos/run_face_recognition_demo.py")
    print("  python domains/face_recognition/runners/data_collection/run_face_registration.py")
    print()
    print("🛠️ 시스템 도구")
    print("  python tools/setup/download_models.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
```

## 🎯 정리 후 최종 구조

```
vision_system/
├── domains/face_recognition/
│   ├── data/storage/          # ✅ 도메인별 데이터 저장소
│   ├── runners/demos/         # ✅ 데모 실행 파일들
│   └── runners/data_collection/  # ✅ 데이터 수집 도구들
├── tools/setup/               # ✅ 시스템 도구들
├── data/                      # ✅ 시스템 공통 데이터만
└── launcher.py                # ✅ 통합 런처 (하나만)
```

## 🚀 정리 완료 후 실행

```bash
# 새로운 구조로 실행
python domains/face_recognition/runners/demos/run_simple_demo.py
python domains/face_recognition/runners/demos/run_face_recognition_demo.py
python domains/face_recognition/runners/data_collection/run_face_registration.py
```

## ⚠️ 주의사항

1. **백업**: 중요한 데이터는 이동 전에 백업하세요
2. **테스트**: 각 단계마다 파일이 제대로 이동되었는지 확인하세요
3. **Import 경로**: 파일 이동 후 반드시 import 경로를 수정하세요
4. **런처 종료**: 현재 실행 중인 런처를 먼저 종료하세요

이렇게 정리하면 도메인 주도 설계(DDD) 원칙에 맞는 깔끔한 프로젝트 구조가 됩니다! 