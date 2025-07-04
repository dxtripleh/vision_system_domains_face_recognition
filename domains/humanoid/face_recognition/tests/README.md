# Tests - 얼굴인식 테스트 모듈

## 📋 개요

이 폴더는 얼굴인식 기능의 모든 테스트 코드를 포함합니다. 단위 테스트, 통합 테스트, 성능 테스트 등 다양한 종류의 테스트를 통해 코드의 품질과 안정성을 보장합니다.

## 🏗️ 폴더 구조

```
tests/
├── __init__.py                    # 테스트 패키지 초기화
├── README.md                      # 이 파일
├── test_models.py                 # 모델 테스트
├── test_services.py               # 서비스 테스트
├── test_utils.py                  # 유틸리티 테스트
├── test_integration.py            # 통합 테스트
├── test_performance.py            # 성능 테스트
└── conftest.py                    # pytest 설정 및 픽스처
```

## 🔍 포함된 테스트들

### 1. Model Tests (모델 테스트)
- **파일**: `test_models.py`
- **목적**: 얼굴 검출 및 인식 모델의 정확성 검증
- **테스트 항목**: 모델 로딩, 추론, 결과 형식, 성능

### 2. Service Tests (서비스 테스트)
- **파일**: `test_services.py`
- **목적**: 얼굴인식 서비스의 비즈니스 로직 검증
- **테스트 항목**: 서비스 초기화, 얼굴 인식, 데이터베이스 연동

### 3. Utils Tests (유틸리티 테스트)
- **파일**: `test_utils.py`
- **목적**: 유틸리티 함수들의 정확성 검증
- **테스트 항목**: 이미지 처리, 시각화, 데이터 변환

### 4. Integration Tests (통합 테스트)
- **파일**: `test_integration.py`
- **목적**: 전체 얼굴인식 파이프라인 검증
- **테스트 항목**: end-to-end 처리, 모듈 간 연동

### 5. Performance Tests (성능 테스트)
- **파일**: `test_performance.py`
- **목적**: 성능 요구사항 충족 여부 검증
- **테스트 항목**: 처리 속도, 메모리 사용량, 정확도

## 🚀 테스트 실행

### 전체 테스트 실행
```bash
# 프로젝트 루트에서 실행
python -m pytest domains/humanoid/face_recognition/tests/ -v

# 특정 테스트 파일 실행
python -m pytest domains/humanoid/face_recognition/tests/test_models.py -v

# 특정 테스트 클래스 실행
python -m pytest domains/humanoid/face_recognition/tests/test_models.py::TestFaceDetectionModel -v

# 특정 테스트 함수 실행
python -m pytest domains/humanoid/face_recognition/tests/test_models.py::TestFaceDetectionModel::test_detect_faces -v
```

### 테스트 카테고리별 실행
```bash
# 단위 테스트만 실행
python -m pytest domains/humanoid/face_recognition/tests/ -m "not integration" -v

# 통합 테스트만 실행
python -m pytest domains/humanoid/face_recognition/tests/ -m "integration" -v

# 성능 테스트만 실행
python -m pytest domains/humanoid/face_recognition/tests/ -m "performance" -v
```

### 테스트 커버리지 확인
```bash
# 커버리지와 함께 테스트 실행
python -m pytest domains/humanoid/face_recognition/tests/ --cov=domains.humanoid.face_recognition --cov-report=html

# 커버리지 리포트 확인
open htmlcov/index.html
```

## 📊 테스트 예시

### 모델 테스트 예시
```python
import pytest
import numpy as np
from domains.humanoid.face_recognition.models import FaceDetectionModel

class TestFaceDetectionModel:
    """얼굴 검출 모델 테스트"""
    
    @pytest.fixture
    def model(self):
        """모델 픽스처"""
        return FaceDetectionModel()
    
    @pytest.fixture
    def test_image(self):
        """테스트 이미지 픽스처"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_model_initialization(self, model):
        """모델 초기화 테스트"""
        assert model is not None
        assert hasattr(model, 'detect')
    
    def test_detect_faces(self, model, test_image):
        """얼굴 검출 테스트"""
        faces = model.detect(test_image)
        
        assert isinstance(faces, list)
        for face in faces:
            assert 'bbox' in face
            assert 'confidence' in face
            assert len(face['bbox']) == 4
    
    def test_detect_faces_empty_image(self, model):
        """빈 이미지 검출 테스트"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = model.detect(empty_image)
        
        assert isinstance(faces, list)
        assert len(faces) == 0
```

### 서비스 테스트 예시
```python
import pytest
from unittest.mock import Mock, patch
from domains.humanoid.face_recognition.services import FaceRecognitionService

class TestFaceRecognitionService:
    """얼굴인식 서비스 테스트"""
    
    @pytest.fixture
    def service(self):
        """서비스 픽스처"""
        return FaceRecognitionService()
    
    @pytest.fixture
    def mock_detection_result(self):
        """가짜 검출 결과 픽스처"""
        return {
            'faces': [
                {
                    'bbox': [100, 100, 200, 200],
                    'confidence': 0.95,
                    'image': np.random.randint(0, 255, (100, 100, 3))
                }
            ]
        }
    
    @patch('domains.humanoid.face_recognition.services.FaceDetectionModel')
    def test_recognize_faces(self, mock_detection_model, service, mock_detection_result):
        """얼굴 인식 테스트"""
        # 가짜 모델 설정
        mock_model = Mock()
        mock_model.detect.return_value = mock_detection_result['faces']
        mock_detection_model.return_value = mock_model
        
        # 테스트 실행
        result = service.recognize_faces(np.random.randint(0, 255, (480, 640, 3)))
        
        # 결과 검증
        assert 'faces' in result
        assert isinstance(result['faces'], list)
```

### 통합 테스트 예시
```python
import pytest
from domains.humanoid.face_recognition.services import FaceRecognitionService
from domains.humanoid.face_recognition.models import FaceDetectionModel, FaceRecognitionModel

@pytest.mark.integration
class TestFaceRecognitionIntegration:
    """얼굴인식 통합 테스트"""
    
    def test_full_pipeline(self):
        """전체 파이프라인 테스트"""
        # 모델 초기화
        detection_model = FaceDetectionModel()
        recognition_model = FaceRecognitionModel()
        
        # 서비스 초기화
        service = FaceRecognitionService()
        
        # 테스트 이미지 생성
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 전체 파이프라인 실행
        result = service.recognize_faces(test_image)
        
        # 결과 검증
        assert isinstance(result, dict)
        assert 'faces' in result
        assert 'processing_time' in result
        assert result['processing_time'] > 0
```

## 🔧 테스트 설정

### pytest 설정 (conftest.py)
```python
import pytest
import numpy as np
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """테스트 데이터 디렉토리"""
    return Path(__file__).parent / "test_data"

@pytest.fixture(scope="session")
def sample_images(test_data_dir):
    """샘플 이미지들"""
    images = []
    for i in range(5):
        # 다양한 크기의 테스트 이미지 생성
        image = np.random.randint(0, 255, (480 + i*100, 640 + i*100, 3), dtype=np.uint8)
        images.append(image)
    return images

@pytest.fixture
def mock_face_database():
    """가짜 얼굴 데이터베이스"""
    return {
        "faces": [
            {
                "id": "test_person_001",
                "name": "테스트 사용자 1",
                "embedding": np.random.rand(512).tolist(),
                "metadata": {"age": 30, "gender": "male"}
            }
        ]
    }
```

### 테스트 마커 설정
```python
# pytest.ini 또는 pyproject.toml
[tool.pytest.ini_options]
markers = [
    "unit: 단위 테스트",
    "integration: 통합 테스트",
    "performance: 성능 테스트",
    "slow: 느린 테스트",
    "gpu: GPU 필요 테스트"
]
```

## 📈 성능 테스트

### 성능 테스트 예시
```python
import pytest
import time
import psutil
from domains.humanoid.face_recognition.models import FaceDetectionModel

@pytest.mark.performance
class TestPerformance:
    """성능 테스트"""
    
    def test_detection_speed(self):
        """검출 속도 테스트"""
        model = FaceDetectionModel()
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 성능 측정
        start_time = time.time()
        for _ in range(100):
            faces = model.detect(test_image)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        fps = 1.0 / avg_time
        
        # 성능 요구사항 검증
        assert fps >= 15.0, f"FPS가 너무 낮음: {fps:.1f}"
        assert avg_time < 0.1, f"평균 처리 시간이 너무 김: {avg_time*1000:.1f}ms"
    
    def test_memory_usage(self):
        """메모리 사용량 테스트"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        model = FaceDetectionModel()
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 메모리 사용량 측정
        for _ in range(10):
            faces = model.detect(test_image)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # 메모리 요구사항 검증
        assert memory_increase < 500, f"메모리 사용량이 너무 높음: {memory_increase:.1f}MB"
```

## 🐛 문제 해결

### 일반적인 테스트 문제들

#### 1. Import 오류
```python
# 해결 방법: sys.path에 프로젝트 루트 추가
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
```

#### 2. 모델 파일 없음
```python
# 해결 방법: 가짜 모델 사용
@pytest.fixture
def mock_model():
    """가짜 모델 픽스처"""
    with patch('onnxruntime.InferenceSession') as mock_session:
        mock_session.return_value = Mock()
        yield mock_session
```

#### 3. 테스트 데이터 없음
```python
# 해결 방법: 동적 테스트 데이터 생성
@pytest.fixture
def test_images():
    """테스트 이미지들"""
    return [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(5)
    ]
```

## 📝 테스트 작성 가이드

### 테스트 작성 규칙
1. **테스트 함수명**: `test_{기능}_{조건}_{예상결과}`
2. **테스트 클래스명**: `Test{클래스명}`
3. **픽스처 사용**: 반복되는 설정은 픽스처로 분리
4. **가짜 객체 사용**: 외부 의존성은 Mock으로 대체
5. **명확한 검증**: assert 문으로 명확한 결과 검증

### 테스트 구조
```python
class TestExample:
    """테스트 예시"""
    
    def setup_method(self):
        """각 테스트 전 실행"""
        pass
    
    def teardown_method(self):
        """각 테스트 후 실행"""
        pass
    
    def test_something(self):
        """테스트 함수"""
        # Given (준비)
        input_data = "test"
        
        # When (실행)
        result = process_data(input_data)
        
        # Then (검증)
        assert result == "expected"
```

## 🔗 관련 문서

- [얼굴인식 기능 문서](../README.md)
- [모델 문서](../models/README.md)
- [서비스 문서](../services/README.md)
- [유틸리티 문서](../utils/README.md)
- [Humanoid 도메인 문서](../../README.md)
- [프로젝트 전체 문서](../../../../README.md)

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 이 README 파일 확인
2. 상위 폴더의 README.md 확인
3. pytest 공식 문서 참조
4. 개발팀에 문의

---

**마지막 업데이트**: 2025-07-04
**버전**: 1.0.0
**작성자**: Vision System Team 