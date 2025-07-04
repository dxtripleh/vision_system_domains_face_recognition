# Domains Structure - 도메인 구조 상세 가이드

## 📋 개요

이 문서는 `domains/` 폴더의 상세한 구조와 각 구성 요소의 역할을 설명합니다. 초보자도 쉽게 이해할 수 있도록 단계별로 설명합니다.

## 🏗️ 전체 구조 개요

```
domains/
├── __init__.py                    # 도메인 패키지 초기화
├── README.md                      # 도메인 개요 및 사용법
├── STRUCTURE.md                   # 이 파일 - 상세 구조 가이드
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

## 🎯 도메인 개념 설명

### 도메인이란?
도메인(Domain)은 특정 비즈니스 영역이나 문제 영역을 의미합니다. 예를 들어:
- **Humanoid Domain**: 인간과 관련된 모든 비전 기능
- **Factory Domain**: 공장과 관련된 모든 비전 기능
- **Infrastructure Domain**: 인프라와 관련된 모든 비전 기능 (향후)

### 도메인 분리의 이유
1. **독립성**: 각 도메인은 독립적으로 개발/배포 가능
2. **유지보수성**: 특정 도메인의 변경이 다른 도메인에 영향 없음
3. **확장성**: 새로운 도메인을 쉽게 추가 가능
4. **팀 분업**: 도메인별로 다른 팀이 담당 가능

## 📁 표준 폴더 구조 상세 설명

### 1. 최상위 도메인 폴더 (`domains/{domain}/`)

```
{domain}/
├── __init__.py                    # 도메인 패키지 초기화
├── README.md                      # 도메인 설명 문서
└── {feature}/                     # 특정 기능 폴더
```

**역할**:
- `__init__.py`: Python 패키지로 인식되게 하는 필수 파일
- `README.md`: 해당 도메인의 목적, 기능, 사용법 설명
- `{feature}/`: 도메인 내의 특정 기능 (예: face_recognition, defect_detection)

### 2. 기능별 표준 구조 (`domains/{domain}/{feature}/`)

```
{feature}/
├── __init__.py                    # 기능 패키지 초기화
├── README.md                      # 기능 설명 문서
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

## 🔍 각 폴더 상세 설명

### 1. `models/` 폴더
**목적**: AI 모델 관련 클래스들을 포함

**포함 파일**:
- `__init__.py`: 모델 클래스들을 외부에서 import할 수 있게 함
- `{feature}_model.py`: ONNX 기반 모델 클래스

**예시**:
```python
# models/face_detection_model.py
class FaceDetectionModel:
    def __init__(self, model_path: str):
        # 모델 초기화
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        # 얼굴 검출 수행
```

### 2. `services/` 폴더
**목적**: 비즈니스 로직을 담당하는 서비스 클래스들

**포함 파일**:
- `__init__.py`: 서비스 클래스들을 외부에서 import할 수 있게 함
- `{feature}_service.py`: 비즈니스 로직 서비스

**예시**:
```python
# services/face_recognition_service.py
class FaceRecognitionService:
    def __init__(self, detection_model, recognition_model):
        # 서비스 초기화
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        # 프레임 처리 로직
```

### 3. `utils/` 폴더
**목적**: 헬퍼 함수들과 유틸리티 스크립트들

**포함 파일**:
- `__init__.py`: 유틸리티 함수들을 외부에서 import할 수 있게 함
- `demo.py`: 데모 스크립트

**예시**:
```python
# utils/demo.py
def create_test_image() -> np.ndarray:
    # 테스트용 이미지 생성

def visualize_results(image: np.ndarray, results: List[Dict]) -> np.ndarray:
    # 결과 시각화
```

### 4. `tests/` 폴더
**목적**: 단위 테스트와 통합 테스트

**포함 파일**:
- `__init__.py`: 테스트 모듈들을 외부에서 import할 수 있게 함
- `test_{feature}.py`: 단위 테스트

**예시**:
```python
# tests/test_face_recognition.py
class FaceRecognitionTester:
    def test_face_detection_model(self):
        # 얼굴 검출 모델 테스트
    
    def test_face_recognition_model(self):
        # 얼굴 인식 모델 테스트
```

### 5. `configs/` 폴더
**목적**: 설정 파일들을 관리

**포함 파일**:
- `__init__.py`: 설정 관련 상수들을 정의
- `{feature}_config.yaml`: YAML 형식 설정 파일 (선택적)

**예시**:
```python
# configs/__init__.py
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MODEL_PATH = "models/weights/face_detection.onnx"
```

### 6. `pipeline/` 폴더
**목적**: 9단계 데이터 파이프라인 구현

**포함 파일**:
- `__init__.py`: 파이프라인 모듈들을 외부에서 import할 수 있게 함
- `step_1_capture.py`: 1단계 - 데이터 캡처
- `step_2_extract.py`: 2단계 - 특징 추출
- `step_3_cluster.py`: 3단계 - 클러스터링
- `step_4_label.py`: 4단계 - 라벨링
- `step_5_embed.py`: 5단계 - 임베딩
- `step_6_realtime.py`: 6단계 - 실시간 인식
- `step_7_database.py`: 7단계 - 데이터베이스 관리
- `step_8_monitor.py`: 8단계 - 성능 모니터링
- `step_9_learning.py`: 9단계 - 연속 학습

## 📝 파일 네이밍 규칙

### 1. 실행 파일
- **패턴**: `run_{feature}.py`
- **예시**: `run_face_recognition.py`, `run_defect_detection.py`
- **역할**: 해당 기능을 실행하는 메인 스크립트

### 2. 모델 파일
- **패턴**: `{feature}_model.py`
- **예시**: `face_detection_model.py`, `defect_detection_model.py`
- **역할**: ONNX 모델을 로드하고 추론을 수행하는 클래스

### 3. 서비스 파일
- **패턴**: `{feature}_service.py`
- **예시**: `face_recognition_service.py`, `defect_detection_service.py`
- **역할**: 비즈니스 로직을 담당하는 서비스 클래스

### 4. 테스트 파일
- **패턴**: `test_{feature}.py`
- **예시**: `test_face_recognition.py`, `test_defect_detection.py`
- **역할**: 해당 기능의 단위 테스트

## 🔗 의존성 규칙

### 1. 도메인 간 의존성 (절대 금지)
```python
# ❌ 절대 금지
from domains.humanoid.face_recognition import FaceRecognitionModel
# factory 도메인에서 humanoid 도메인을 직접 import

# ✅ 올바른 방법
from shared.vision_core.detection import BaseDetector
# 공유 모듈을 통해서만 공통 기능 사용
```

### 2. 계층별 의존성 (위에서 아래로만 허용)
```
Level 4: domains/              # 도메인 계층
    ↓
Level 3: models/               # 모델 계층
    ↓
Level 2: shared/               # 공유 모듈 계층
    ↓
Level 1: common/, config/      # 기반 계층
```

### 3. 같은 도메인 내 의존성 (자유롭게 허용)
```python
# ✅ 허용
from .models.face_detection_model import FaceDetectionModel
from .services.face_recognition_service import FaceRecognitionService
# 같은 도메인 내에서는 자유롭게 import 가능
```

## 🚀 새로운 도메인 생성 가이드

### 1단계: 도메인 카테고리 결정
```bash
# 새로운 도메인 카테고리 생성
mkdir domains/new_domain
touch domains/new_domain/__init__.py
touch domains/new_domain/README.md
```

### 2단계: 기능 폴더 생성
```bash
# 새로운 기능 생성
mkdir domains/new_domain/new_feature
touch domains/new_domain/new_feature/__init__.py
touch domains/new_domain/new_feature/README.md
```

### 3단계: 표준 구조 생성
```bash
# 표준 폴더 구조 생성
mkdir -p domains/new_domain/new_feature/{models,services,utils,tests,configs,pipeline}
touch domains/new_domain/new_feature/{models,services,utils,tests,configs,pipeline}/__init__.py
```

### 4단계: 필수 파일 생성
```bash
# 실행 스크립트
touch domains/new_domain/new_feature/run_new_feature.py

# 모델 클래스
touch domains/new_domain/new_feature/models/new_feature_model.py

# 서비스 클래스
touch domains/new_domain/new_feature/services/new_feature_service.py

# 테스트 파일
touch domains/new_domain/new_feature/tests/test_new_feature.py
```

## 🧪 테스트 규칙

### 1. 테스트 파일 위치
- **도메인 개발 테스트**: `domains/{domain}/{feature}/tests/`
- **통합 테스트**: `tests/` (프로젝트 루트)
- **파이프라인 테스트**: `domains/{domain}/{feature}/pipeline/tests/`

### 2. 테스트 실행
```bash
# 도메인 테스트
python -m pytest domains/humanoid/face_recognition/tests/ -v

# 통합 테스트
python -m pytest tests/ -v

# 특정 테스트
python -m pytest domains/humanoid/face_recognition/tests/test_service.py -v
```

### 3. 테스트 작성 규칙
- **클래스명**: `{Feature}Tester` (예: `FaceRecognitionTester`)
- **메서드명**: `test_{component}_{action}` (예: `test_face_detection_model`)
- **문서화**: 각 테스트에 대한 설명 필수

## 📊 데이터 관리 규칙

### 1. 학습 데이터
- **위치**: `datasets/{domain}/{feature}/`
- **구조**: raw/, processed/, annotations/, splits/
- **형식**: 이미지 파일, 라벨링 파일 (JSON, YAML, CSV)

### 2. 런타임 데이터
- **위치**: `data/domains/{domain}/{feature}/`
- **구조**: captures/, results/, logs/
- **보존**: 자동 정리 정책 적용

### 3. 모델 가중치
- **위치**: `models/weights/`
- **형식**: .onnx 파일만 허용
- **네이밍**: `{task}_{architecture}_{dataset}_{date}.onnx`

## 🔧 개발 가이드

### 1. 코드 품질 관리
- **Type Hints**: 모든 함수에 타입 힌트 필수
- **Docstring**: Google Style 문서화 필수
- **예외 처리**: 적절한 예외 처리 구현
- **로깅**: 구조화된 로깅 사용

### 2. 성능 최적화
- **ONNX 사용**: 추론 시 ONNX 모델 필수
- **배치 처리**: 가능한 경우 배치 처리 구현
- **메모리 관리**: 적절한 메모리 해제
- **GPU 활용**: GPU 사용 가능 시 자동 감지

### 3. 플랫폼 호환성
- **경로 처리**: `pathlib.Path` 사용
- **하드웨어 감지**: 자동 하드웨어 환경 감지
- **카메라 백엔드**: 플랫폼별 카메라 백엔드 사용

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
5. 개발팀에 문의

---

**마지막 업데이트**: 2025-07-04
**버전**: 1.0.0
**작성자**: Vision System Team 