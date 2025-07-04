# Tests - 테스트 코드

## 📋 개요

`tests/` 폴더는 프로젝트 전체의 공식 테스트 코드들을 포함합니다. 단위 테스트, 통합 테스트, 성능 테스트 등 다양한 수준의 테스트를 통해 코드의 품질과 안정성을 보장합니다.

## 🏗️ 구조

```
tests/
├── __init__.py
├── README.md
├── unit/                       # 단위 테스트
│   ├── test_common/            # common 모듈 테스트
│   ├── test_shared/            # shared 모듈 테스트
│   ├── test_face_recognition/  # 얼굴인식 테스트
│   └── test_defect_detection/  # 불량검출 테스트
├── integration/                # 통합 테스트
│   ├── test_domains/           # 도메인 통합 테스트
│   ├── test_data_flow/         # 데이터 흐름 테스트
│   └── test_system/            # 시스템 통합 테스트
├── performance/                # 성능 테스트
│   ├── test_inference_speed/   # 추론 속도 테스트
│   ├── test_memory_usage/      # 메모리 사용량 테스트
│   └── test_concurrent/        # 동시성 테스트
├── e2e/                        # End-to-End 테스트
│   ├── test_face_recognition_workflow/  # 얼굴인식 워크플로우
│   └── test_defect_detection_workflow/  # 불량검출 워크플로우
├── fixtures/                   # 테스트 데이터
│   ├── images/                 # 테스트 이미지
│   ├── models/                 # 테스트 모델
│   └── configs/                # 테스트 설정
└── utils/                      # 테스트 유틸리티
    ├── test_helpers.py         # 테스트 헬퍼 함수
    ├── mock_data.py            # 목 데이터
    └── test_config.py          # 테스트 설정
```

## 📁 테스트 유형별 설명

### `unit/` - 단위 테스트

개별 함수, 클래스, 모듈의 동작을 검증하는 테스트입니다.

#### `test_common/` - common 모듈 테스트
- **목적**: common 모듈의 각 함수별 동작 검증
- **테스트 대상**:
  - `config.py`: 설정 로딩/저장 테스트
  - `logging.py`: 로깅 시스템 테스트
  - `utils.py`: 유틸리티 함수 테스트

```python
# test_common/test_config.py 예시
import pytest
from common.config import load_config, get_config

def test_load_config():
    """설정 파일 로딩 테스트"""
    config = load_config('tests/fixtures/configs/test_config.yaml')
    assert config['model']['confidence_threshold'] == 0.5

def test_get_config_with_default():
    """기본값과 함께 설정값 가져오기 테스트"""
    value = get_config('nonexistent.key', default='default_value')
    assert value == 'default_value'
```

#### `test_shared/` - shared 모듈 테스트
- **목적**: shared 모듈의 공통 기능 검증
- **테스트 대상**:
  - `vision_core/detection/`: 객체 검출 공통 기능
  - `vision_core/recognition/`: 객체 인식 공통 기능
  - `vision_core/preprocessing/`: 전처리 공통 기능

#### `test_face_recognition/` - 얼굴인식 테스트
- **목적**: 얼굴인식 기능의 각 컴포넌트 검증
- **테스트 대상**:
  - 모델 로딩
  - 이미지 전처리
  - 얼굴 검출
  - 얼굴 인식

#### `test_defect_detection/` - 불량검출 테스트
- **목적**: 불량검출 기능의 각 컴포넌트 검증
- **테스트 대상**:
  - 모델 로딩
  - 이미지 전처리
  - 불량 검출
  - 불량 분류

### `integration/` - 통합 테스트

여러 모듈이 함께 동작할 때의 상호작용을 검증하는 테스트입니다.

#### `test_domains/` - 도메인 통합 테스트
- **목적**: 도메인 내부의 모듈 간 상호작용 검증
- **테스트 내용**:
  - 얼굴인식 도메인의 전체 워크플로우
  - 불량검출 도메인의 전체 워크플로우

```python
# test_domains/test_face_recognition_integration.py 예시
import pytest
from domains.humanoid.face_recognition.face_recognition_model import FaceRecognitionModel
from domains.humanoid.face_recognition.run_face_recognition import FaceRecognitionRunner

def test_face_recognition_integration():
    """얼굴인식 통합 테스트"""
    # 모델 초기화
    model = FaceRecognitionModel()
    
    # 실행기 초기화
    runner = FaceRecognitionRunner(model)
    
    # 테스트 이미지로 실행
    test_image = "tests/fixtures/images/test_face.jpg"
    result = runner.process_image(test_image)
    
    # 결과 검증
    assert result is not None
    assert 'faces' in result
    assert len(result['faces']) > 0
```

#### `test_data_flow/` - 데이터 흐름 테스트
- **목적**: 데이터가 시스템을 통과하는 과정 검증
- **테스트 내용**:
  - 입력 데이터 처리
  - 중간 결과 생성
  - 최종 출력 검증

#### `test_system/` - 시스템 통합 테스트
- **목적**: 전체 시스템의 동작 검증
- **테스트 내용**:
  - 시스템 초기화
  - 모듈 간 통신
  - 오류 처리

### `performance/` - 성능 테스트

시스템의 성능 지표를 측정하는 테스트입니다.

#### `test_inference_speed/` - 추론 속도 테스트
- **목적**: 모델 추론 속도 측정
- **측정 지표**:
  - FPS (Frames Per Second)
  - 평균 추론 시간
  - 지연 시간 분포

```python
# test_performance/test_inference_speed.py 예시
import time
import pytest
from domains.humanoid.face_recognition.face_recognition_model import FaceRecognitionModel

def test_face_recognition_inference_speed():
    """얼굴인식 추론 속도 테스트"""
    model = FaceRecognitionModel()
    test_image = "tests/fixtures/images/test_face.jpg"
    
    # 워밍업
    for _ in range(10):
        model.detect_faces(test_image)
    
    # 성능 측정
    times = []
    for _ in range(100):
        start_time = time.time()
        model.detect_faces(test_image)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time
    
    # 성능 기준 검증
    assert fps >= 30.0, f"FPS가 너무 낮음: {fps:.2f}"
    assert avg_time <= 0.033, f"평균 추론 시간이 너무 김: {avg_time:.3f}초"
```

#### `test_memory_usage/` - 메모리 사용량 테스트
- **목적**: 메모리 사용량 측정
- **측정 지표**:
  - 메모리 사용량
  - 메모리 누수 검사
  - 가비지 컬렉션 효과

#### `test_concurrent/` - 동시성 테스트
- **목적**: 동시 요청 처리 능력 측정
- **측정 지표**:
  - 동시 처리 성능
  - 스레드 안전성
  - 리소스 경합

### `e2e/` - End-to-End 테스트

사용자 관점에서 전체 시스템의 동작을 검증하는 테스트입니다.

#### `test_face_recognition_workflow/` - 얼굴인식 워크플로우
- **목적**: 얼굴인식 전체 과정 검증
- **테스트 시나리오**:
  1. 카메라 연결
  2. 이미지 캡처
  3. 얼굴 검출
  4. 얼굴 인식
  5. 결과 출력

#### `test_defect_detection_workflow/` - 불량검출 워크플로우
- **목적**: 불량검출 전체 과정 검증
- **테스트 시나리오**:
  1. 카메라 연결
  2. 제품 이미지 캡처
  3. 불량 검출
  4. 불량 분류
  5. 결과 저장

## 🔧 테스트 실행

### 기본 테스트 실행

```bash
# 모든 테스트 실행
python -m pytest tests/

# 특정 테스트 유형 실행
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/performance/
python -m pytest tests/e2e/

# 특정 모듈 테스트
python -m pytest tests/unit/test_common/
python -m pytest tests/unit/test_face_recognition/
```

### 테스트 옵션

```bash
# 상세 출력
python -m pytest tests/ -v

# 실패한 테스트만 재실행
python -m pytest tests/ --lf

# 테스트 커버리지 측정
python -m pytest tests/ --cov=domains --cov=shared --cov=common

# 병렬 실행
python -m pytest tests/ -n 4

# 특정 마커가 있는 테스트만 실행
python -m pytest tests/ -m "slow"
python -m pytest tests/ -m "not slow"
```

### CI/CD에서 테스트 실행

```yaml
# .github/workflows/test.yml 예시
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-xdist
      - name: Run tests
        run: |
          python -m pytest tests/ -v --cov=domains --cov=shared --cov=common --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## 📊 테스트 결과 분석

### 테스트 커버리지

```bash
# 커버리지 리포트 생성
python -m pytest tests/ --cov=domains --cov=shared --cov=common --cov-report=html

# HTML 리포트 확인
open htmlcov/index.html
```

### 성능 테스트 결과

```python
# 성능 테스트 결과 분석
import json
from tests.utils.performance_analyzer import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
results = analyzer.analyze_performance_results("tests/results/performance.json")

print(f"평균 FPS: {results['avg_fps']:.2f}")
print(f"최대 메모리 사용량: {results['max_memory_mb']:.2f} MB")
print(f"성능 등급: {results['performance_grade']}")
```

## 🐛 테스트 문제 해결

### 일반적인 문제들

#### 1. 테스트 데이터 누락
```bash
# 테스트 데이터 다운로드
python scripts/data/download_test_data.py

# 테스트 데이터 검증
python -m pytest tests/ --validate-fixtures
```

#### 2. 모델 파일 누락
```bash
# 테스트용 모델 다운로드
python scripts/model/download_test_models.py

# 모델 파일 검증
python -m pytest tests/ --validate-models
```

#### 3. 환경 설정 문제
```bash
# 테스트 환경 설정
python scripts/setup/setup_test_environment.py

# 환경 검증
python -m pytest tests/ --validate-environment
```

## 📋 테스트 작성 가이드

### 단위 테스트 작성

```python
# tests/unit/test_new_feature.py
import pytest
from unittest.mock import Mock, patch
from domains.humanoid.face_recognition.face_recognition_model import FaceRecognitionModel

class TestFaceRecognitionModel:
    """얼굴인식 모델 테스트 클래스"""
    
    def setup_method(self):
        """각 테스트 전 실행"""
        self.model = FaceRecognitionModel()
    
    def teardown_method(self):
        """각 테스트 후 실행"""
        self.model = None
    
    def test_model_initialization(self):
        """모델 초기화 테스트"""
        assert self.model is not None
        assert hasattr(self.model, 'detect_faces')
    
    def test_detect_faces_with_valid_image(self):
        """유효한 이미지로 얼굴 검출 테스트"""
        test_image = "tests/fixtures/images/test_face.jpg"
        result = self.model.detect_faces(test_image)
        
        assert result is not None
        assert isinstance(result, list)
    
    def test_detect_faces_with_invalid_image(self):
        """잘못된 이미지로 얼굴 검출 테스트"""
        with pytest.raises(ValueError):
            self.model.detect_faces("nonexistent_image.jpg")
    
    @pytest.mark.slow
    def test_detect_faces_performance(self):
        """얼굴 검출 성능 테스트"""
        test_image = "tests/fixtures/images/test_face.jpg"
        
        import time
        start_time = time.time()
        self.model.detect_faces(test_image)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 0.1  # 100ms 이내
```

### 통합 테스트 작성

```python
# tests/integration/test_face_recognition_workflow.py
import pytest
from pathlib import Path
from domains.humanoid.face_recognition.run_face_recognition import FaceRecognitionRunner

class TestFaceRecognitionWorkflow:
    """얼굴인식 워크플로우 통합 테스트"""
    
    @pytest.fixture
    def runner(self):
        """테스트용 실행기 생성"""
        return FaceRecognitionRunner()
    
    @pytest.fixture
    def test_images(self):
        """테스트 이미지 목록"""
        image_dir = Path("tests/fixtures/images/")
        return list(image_dir.glob("*.jpg"))
    
    def test_complete_workflow(self, runner, test_images):
        """완전한 워크플로우 테스트"""
        for image_path in test_images:
            # 이미지 처리
            result = runner.process_image(str(image_path))
            
            # 결과 검증
            assert result is not None
            assert 'faces' in result
            assert 'processing_time' in result
            assert result['processing_time'] > 0
    
    def test_error_handling(self, runner):
        """오류 처리 테스트"""
        # 잘못된 입력으로 테스트
        with pytest.raises(ValueError):
            runner.process_image("invalid_path.jpg")
```

### 성능 테스트 작성

```python
# tests/performance/test_face_recognition_performance.py
import pytest
import time
import psutil
import os
from domains.humanoid.face_recognition.face_recognition_model import FaceRecognitionModel

class TestFaceRecognitionPerformance:
    """얼굴인식 성능 테스트"""
    
    @pytest.fixture
    def model(self):
        """테스트용 모델"""
        return FaceRecognitionModel()
    
    @pytest.fixture
    def test_image(self):
        """테스트 이미지"""
        return "tests/fixtures/images/test_face.jpg"
    
    def test_inference_speed(self, model, test_image):
        """추론 속도 테스트"""
        # 워밍업
        for _ in range(5):
            model.detect_faces(test_image)
        
        # 성능 측정
        times = []
        for _ in range(50):
            start_time = time.time()
            model.detect_faces(test_image)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        
        # 성능 기준 검증
        assert fps >= 30.0, f"FPS가 기준 미달: {fps:.2f}"
        assert avg_time <= 0.033, f"평균 시간이 기준 초과: {avg_time:.3f}초"
    
    def test_memory_usage(self, model, test_image):
        """메모리 사용량 테스트"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 모델 사용
        for _ in range(100):
            model.detect_faces(test_image)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 메모리 증가량 검증 (메모리 누수 방지)
        assert memory_increase < 100, f"메모리 증가량이 너무 큼: {memory_increase:.2f} MB"
```

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 이 README 파일을 먼저 확인
2. pytest 문서 확인
3. 테스트 로그 확인
4. 프로젝트 루트의 README.md 확인

## 📄 라이선스

이 모듈의 코드는 프로젝트 전체 라이선스를 따릅니다. 