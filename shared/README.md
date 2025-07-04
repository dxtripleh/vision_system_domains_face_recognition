# Shared - 공통 모듈

## 📋 개요

`shared/` 폴더는 여러 도메인에서 공통으로 사용하는 기능들을 포함합니다. 도메인 간 독립성을 유지하면서도 공통 기능을 효율적으로 공유할 수 있도록 설계되었습니다.

## 🏗️ 구조

```
shared/
├── __init__.py
├── README.md
└── vision_core/           # 비전 알고리즘 공통 기능
    ├── __init__.py
    ├── detection/         # 객체 검출 공통 기능
    ├── recognition/       # 객체 인식 공통 기능
    └── preprocessing/     # 이미지 전처리 공통 기능
```

## 📁 모듈별 설명

### `vision_core/` - 비전 알고리즘 공통 기능

비전 시스템의 핵심 알고리즘들을 공통으로 제공하는 모듈입니다.

#### `detection/` - 객체 검출 공통 기능
- **목적**: 다양한 객체 검출 알고리즘의 공통 인터페이스
- **기능**: 
  - 바운딩 박스 검출
  - 다중 객체 검출
  - 신뢰도 기반 필터링
  - NMS (Non-Maximum Suppression)

#### `recognition/` - 객체 인식 공통 기능
- **목적**: 검출된 객체의 분류 및 인식
- **기능**:
  - 객체 분류
  - 특징점 추출
  - 유사도 계산
  - 임베딩 생성

#### `preprocessing/` - 이미지 전처리 공통 기능
- **목적**: 이미지 전처리 공통 기능
- **기능**:
  - 이미지 리사이징
  - 정규화
  - 데이터 증강
  - 노이즈 제거

## 🔧 사용법

### 도메인에서 공통 모듈 사용

```python
# 얼굴인식 도메인에서 사용 예시
from shared.vision_core.detection import BaseDetector
from shared.vision_core.recognition import BaseRecognizer
from shared.vision_core.preprocessing import ImagePreprocessor

class FaceRecognitionModel:
    def __init__(self):
        self.detector = BaseDetector()
        self.recognizer = BaseRecognizer()
        self.preprocessor = ImagePreprocessor()
    
    def process(self, image):
        # 전처리
        processed_image = self.preprocessor.process(image)
        
        # 검출
        detections = self.detector.detect(processed_image)
        
        # 인식
        recognitions = self.recognizer.recognize(processed_image, detections)
        
        return recognitions
```

### 불량검출 도메인에서 사용 예시

```python
# 불량검출 도메인에서 사용 예시
from shared.vision_core.detection import BaseDetector
from shared.vision_core.preprocessing import ImagePreprocessor

class DefectDetectionModel:
    def __init__(self):
        self.detector = BaseDetector()
        self.preprocessor = ImagePreprocessor()
    
    def detect_defects(self, image):
        # 전처리
        processed_image = self.preprocessor.process(image)
        
        # 불량 검출
        defects = self.detector.detect(processed_image)
        
        return defects
```

## 🔗 의존성 규칙

### 허용되는 의존성
- **도메인 → Shared**: 모든 도메인에서 shared 모듈 사용 가능
- **Shared → Common**: shared 모듈에서 common 모듈 사용 가능
- **Shared → Config**: shared 모듈에서 config 모듈 사용 가능

### 금지되는 의존성
- **Shared → Domains**: shared 모듈에서 도메인 모듈 사용 금지
- **Shared → Models**: shared 모듈에서 models 모듈 직접 사용 금지

## 📋 공통 인터페이스

### BaseDetector 클래스
```python
class BaseDetector:
    """객체 검출 기본 클래스"""
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """이미지에서 객체 검출"""
        pass
    
    def set_confidence_threshold(self, threshold: float):
        """신뢰도 임계값 설정"""
        pass
    
    def set_max_detections(self, max_count: int):
        """최대 검출 개수 설정"""
        pass
```

### BaseRecognizer 클래스
```python
class BaseRecognizer:
    """객체 인식 기본 클래스"""
    
    def recognize(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """검출된 객체 인식"""
        pass
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """특징점 추출"""
        pass
    
    def compare_features(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """특징점 비교"""
        pass
```

### ImagePreprocessor 클래스
```python
class ImagePreprocessor:
    """이미지 전처리 기본 클래스"""
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        pass
    
    def resize(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """이미지 리사이징"""
        pass
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """이미지 정규화"""
        pass
```

## 🚀 새로운 공통 기능 추가

### 1단계: 기능 분석
- 여러 도메인에서 공통으로 사용되는 기능인지 확인
- 도메인별 특수성보다는 공통성이 높은지 검토

### 2단계: 인터페이스 설계
- 추상화된 인터페이스 설계
- 확장 가능한 구조로 설계
- 타입 힌트 및 문서화

### 3단계: 구현
```python
# shared/vision_core/new_feature/__init__.py
from .base_new_feature import BaseNewFeature

__all__ = ['BaseNewFeature']
```

```python
# shared/vision_core/new_feature/base_new_feature.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class BaseNewFeature(ABC):
    """새로운 공통 기능 기본 클래스"""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """데이터 처리"""
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]):
        """설정"""
        pass
```

### 4단계: 테스트
- 단위 테스트 작성
- 여러 도메인에서 사용 가능한지 검증
- 성능 테스트

## 📊 성능 최적화

### 공통 최적화 기법
- **캐싱**: 자주 사용되는 결과 캐싱
- **배치 처리**: 여러 입력을 한 번에 처리
- **메모리 관리**: 효율적인 메모리 사용

### 하드웨어 최적화
- **GPU 가속**: CUDA 지원 기능
- **멀티스레딩**: CPU 멀티코어 활용
- **벡터화**: NumPy 벡터 연산 활용

## 🔍 디버깅 및 로깅

### 로깅 설정
```python
import logging

logger = logging.getLogger(__name__)

class BaseDetector:
    def detect(self, image):
        logger.debug(f"Starting detection on image shape: {image.shape}")
        try:
            result = self._detect_impl(image)
            logger.info(f"Detection completed: {len(result)} objects found")
            return result
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            raise
```

### 디버깅 도구
- **프로파일링**: 성능 병목 지점 찾기
- **메모리 모니터링**: 메모리 사용량 추적
- **로깅**: 상세한 실행 로그

## 🐛 문제 해결

### 일반적인 문제들

#### 1. Import 오류
```python
# ❌ 잘못된 방법
from domains.humanoid.face_recognition import FaceModel

# ✅ 올바른 방법
from shared.vision_core.detection import BaseDetector
```

#### 2. 순환 의존성
```python
# ❌ 금지: shared에서 도메인 import
from domains.factory.defect_detection import DefectModel

# ✅ 권장: 추상화된 인터페이스 사용
from shared.vision_core.detection import BaseDetector
```

#### 3. 성능 문제
- 공통 모듈의 성능 병목 확인
- 캐싱 전략 검토
- 하드웨어 최적화 확인

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 이 README 파일을 먼저 확인
2. 각 하위 모듈의 README 파일 확인
3. 프로젝트 루트의 README.md 확인
4. 로그 파일 확인 (`data/logs/`)

## 📄 라이선스

이 모듈의 코드는 프로젝트 전체 라이선스를 따릅니다. 