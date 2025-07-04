# Domains - 도메인별 독립 개발

## 📋 개요

이 폴더는 비전 시스템의 도메인별 독립 개발을 위한 구조입니다. 각 도메인은 특정 비즈니스 영역에 특화된 기능들을 포함하며, 서로 독립적으로 개발되고 관리됩니다.

## 🏗️ 전체 구조

```
domains/
├── __init__.py                    # 도메인 패키지 초기화
├── README.md                      # 이 파일
├── humanoid/                      # 인간형 도메인
│   ├── __init__.py
│   ├── README.md                  # 인간형 도메인 설명
│   └── face_recognition/          # 얼굴인식 기능
│       ├── __init__.py
│       ├── README.md              # 얼굴인식 기능 설명
│       ├── run_face_recognition.py # 실행 스크립트
│       ├── models/                # 모델 클래스들
│       ├── services/              # 서비스 클래스들
│       ├── utils/                 # 유틸리티 함수들
│       ├── tests/                 # 테스트 파일들
│       ├── configs/               # 설정 파일들
│       └── pipeline/              # 9단계 파이프라인
└── factory/                       # 공장 도메인
    ├── __init__.py
    ├── README.md                  # 공장 도메인 설명
    └── defect_detection/          # 불량 검출 기능
        ├── __init__.py
        ├── README.md              # 불량 검출 기능 설명
        ├── run_defect_detection.py # 실행 스크립트
        ├── models/                # 모델 클래스들
        ├── services/              # 서비스 클래스들
        ├── utils/                 # 유틸리티 함수들
        ├── tests/                 # 테스트 파일들
        ├── configs/               # 설정 파일들
        └── pipeline/              # 9단계 파이프라인
```

## 🎯 도메인별 설명

### 1. Humanoid Domain (인간형 도메인)
- **목적**: 인간과 관련된 비전 인식 기능
- **주요 기능**: 얼굴인식, 감정 검출, 자세 추정
- **적용 분야**: 보안, 인사 관리, 사용자 인터페이스
- **폴더**: `humanoid/`

### 2. Factory Domain (공장 도메인)
- **목적**: 제조업 관련 비전 검사 기능
- **주요 기능**: 불량 검출, 품질 평가, 재고 추적
- **적용 분야**: 제조업, 품질 관리, 자동화
- **폴더**: `factory/`

### 3. Powerline Inspection Domain (활선 검사 도메인) - 향후
- **목적**: 전력선 검사 관련 비전 기능
- **주요 기능**: 활선 검사, 결함 검출, 열화상 분석
- **적용 분야**: 전력 산업, 인프라 관리
- **폴더**: `powerline_inspection/` (향후 생성)

## 📁 표준 폴더 구조 (모든 도메인 기능에 적용)

각 도메인 기능(`{domain}/{feature}/`)은 다음 표준 구조를 따릅니다:

```
{feature}/
├── __init__.py                    # 패키지 초기화
├── README.md                      # 기능 설명
├── run_{feature}.py               # 실행 스크립트
├── models/                        # 모델 클래스들
│   ├── __init__.py
│   └── {feature}_model.py         # ONNX 모델 클래스
├── services/                      # 서비스 클래스들
│   ├── __init__.py
│   └── {feature}_service.py       # 비즈니스 로직 서비스
├── utils/                         # 유틸리티 함수들
│   ├── __init__.py
│   └── demo.py                    # 데모 스크립트
├── tests/                         # 테스트 파일들
│   ├── __init__.py
│   └── test_{feature}.py          # 단위 테스트
├── configs/                       # 설정 파일들
│   └── __init__.py
└── pipeline/                      # 9단계 파이프라인
    └── __init__.py
```

## 🔗 의존성 규칙

### 도메인 간 의존성
```python
# ❌ 절대 금지: 도메인 간 직접 import
from domains.humanoid.face_recognition import something
# 다른 도메인에서 사용 시

# ✅ 올바른 방법: 공유 모듈을 통한 통신
from shared.vision_core.detection import BaseDetector
```

### 계층별 의존성 (위에서 아래로만 허용)
```
Level 4: domains/              # 도메인 계층
    ↓
Level 3: models/               # 모델 계층
    ↓
Level 2: shared/               # 공유 모듈 계층
    ↓
Level 1: common/, config/      # 기반 계층
```

## 🚀 새로운 도메인 생성

### 1. 도메인 카테고리 생성
```bash
# 새로운 도메인 카테고리 생성
mkdir domains/new_domain
touch domains/new_domain/__init__.py
touch domains/new_domain/README.md
```

### 2. 도메인 기능 생성
```bash
# 새로운 기능 생성
mkdir domains/new_domain/new_feature
touch domains/new_domain/new_feature/__init__.py
touch domains/new_domain/new_feature/README.md
```

### 3. 표준 구조 생성
```bash
# 표준 폴더 구조 생성
mkdir -p domains/new_domain/new_feature/{models,services,utils,tests,configs,pipeline}
touch domains/new_domain/new_feature/{models,services,utils,tests,configs,pipeline}/__init__.py
```

## 📝 파일 네이밍 규칙

### 실행 파일
- **실행 스크립트**: `run_{feature}.py`
- **예시**: `run_face_recognition.py`, `run_defect_detection.py`

### 모델 파일
- **모델 클래스**: `{feature}_model.py`
- **예시**: `face_recognition_model.py`, `defect_detection_model.py`

### 서비스 파일
- **서비스 클래스**: `{feature}_service.py`
- **예시**: `face_recognition_service.py`, `defect_detection_service.py`

### 테스트 파일
- **테스트 클래스**: `test_{feature}.py`
- **예시**: `test_face_recognition.py`, `test_defect_detection.py`

## 🧪 테스트 규칙

### 테스트 파일 위치
1. **도메인 개발 테스트**: `domains/{domain}/{feature}/tests/`
2. **통합 테스트**: `tests/` (프로젝트 루트)
3. **파이프라인 테스트**: `domains/{domain}/{feature}/pipeline/tests/`

### 테스트 실행
```bash
# 도메인 테스트
python -m pytest domains/humanoid/face_recognition/tests/ -v

# 통합 테스트
python -m pytest tests/ -v

# 특정 테스트
python -m pytest domains/humanoid/face_recognition/tests/test_service.py -v
```

## 🔧 개발 가이드

### 1. 새로운 기능 개발 시
1. 표준 폴더 구조 생성
2. 모델 클래스 구현 (`models/`)
3. 서비스 클래스 구현 (`services/`)
4. 실행 스크립트 구현 (`run_{feature}.py`)
5. 테스트 작성 (`tests/`)
6. README.md 작성

### 2. 기존 기능 확장 시
1. 기존 구조 유지
2. 새로운 모델/서비스 추가
3. 기존 테스트 수정/확장
4. 문서 업데이트

### 3. 코드 품질 관리
1. Type Hints 사용 필수
2. Docstring 작성 필수
3. 예외 처리 구현
4. 로깅 사용
5. 테스트 커버리지 유지

## 📊 성능 모니터링

### 각 도메인별 성능 지표
- **처리 속도**: FPS (Frames Per Second)
- **정확도**: 검출/인식 정확도
- **메모리 사용량**: RAM/GPU 메모리
- **지연시간**: 입력부터 출력까지의 시간

### 성능 측정 방법
```python
# 성능 측정 예시
import time
start_time = time.time()
result = service.process_frame(frame)
processing_time = time.time() - start_time
fps = 1.0 / processing_time
```

## 🔗 관련 문서

- [프로젝트 전체 README](../../README.md)
- [공통 모듈 문서](../../common/README.md)
- [공유 모듈 문서](../../shared/README.md)
- [설정 파일 문서](../../config/README.md)
- [데이터 파이프라인 가이드](../../PIPELINE_IMPLEMENTATION_SUMMARY.md)

## ⚠️ 중요 규칙

### 모든 개발자가 준수해야 할 사항
1. **도메인 간 직접 의존성 금지**
2. **표준 폴더 구조 준수**
3. **파일 네이밍 규칙 준수**
4. **테스트 파일 위치 규칙 준수**
5. **코드 품질 표준 준수**

### 코드 리뷰 시 확인 사항
1. 도메인 간 의존성 위반 여부
2. 표준 폴더 구조 준수 여부
3. 파일 네이밍 규칙 준수 여부
4. 테스트 코드 포함 여부
5. 문서 작성 여부

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 해당 도메인의 README.md 확인
2. 상위 폴더의 README.md 확인
3. 프로젝트 루트의 README.md 확인
4. 관련 문서 참조 