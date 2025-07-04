# Utils - 얼굴인식 유틸리티 모듈

## 📋 개요

이 폴더는 얼굴인식 기능에 필요한 유틸리티 함수들과 헬퍼 스크립트들을 포함합니다. 이미지 처리, 시각화, 데이터 변환 등 다양한 보조 기능을 제공합니다.

## 🏗️ 폴더 구조

```
utils/
├── __init__.py                    # 유틸리티 패키지 초기화
├── README.md                      # 이 파일
└── demo.py                        # 데모 스크립트
```

## 🔍 포함된 유틸리티들

### 1. Demo Script (데모 스크립트)
- **파일**: `demo.py`
- **목적**: 얼굴인식 기능 데모 및 테스트
- **기능**: 테스트 이미지 생성, 결과 시각화, 성능 측정
- **사용처**: 개발 테스트, 기능 검증, 데모 시연

## 🚀 사용법

### 데모 스크립트 사용법
```python
from domains.humanoid.face_recognition.utils.demo import main, create_test_image

# 데모 실행
main()

# 테스트 이미지 생성
test_image = create_test_image(width=640, height=480)
```

### 유틸리티 함수 사용법
```python
# 이미지 전처리
from domains.humanoid.face_recognition.utils import preprocess_image
processed_image = preprocess_image(image, target_size=(112, 112))

# 결과 시각화
from domains.humanoid.face_recognition.utils import visualize_results
annotated_image = visualize_results(image, detection_results)

# 데이터 변환
from domains.humanoid.face_recognition.utils import convert_bbox_format
normalized_bbox = convert_bbox_format(bbox, from_format='xyxy', to_format='xywh')
```

## 🔧 유틸리티 기능

### 이미지 처리 기능
- **크기 조정**: 다양한 크기로 이미지 리사이징
- **정규화**: 픽셀값 정규화 (0-255 → 0-1)
- **색상 변환**: BGR ↔ RGB 변환
- **회전/반전**: 이미지 기하학적 변환

### 시각화 기능
- **바운딩 박스 그리기**: 얼굴 검출 결과 시각화
- **랜드마크 표시**: 얼굴 특징점 표시
- **신뢰도 표시**: 검출/인식 신뢰도 점수 표시
- **신원 정보 표시**: 인식된 사람 정보 표시

### 데이터 변환 기능
- **좌표 변환**: 다양한 바운딩 박스 형식 변환
- **임베딩 변환**: 특징 벡터 형식 변환
- **메타데이터 변환**: JSON ↔ 딕셔너리 변환

### 성능 측정 기능
- **시간 측정**: 함수 실행 시간 측정
- **메모리 측정**: 메모리 사용량 측정
- **FPS 계산**: 프레임 처리 속도 계산
- **정확도 계산**: 검출/인식 정확도 계산

## 📊 유틸리티 예시

### 이미지 전처리 예시
```python
def preprocess_image(image, target_size=(112, 112)):
    """이미지 전처리 함수"""
    # 크기 조정
    resized = cv2.resize(image, target_size)
    
    # 색상 변환 (BGR → RGB)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # 정규화
    normalized = rgb.astype(np.float32) / 255.0
    
    return normalized
```

### 결과 시각화 예시
```python
def visualize_results(image, results):
    """검출 결과 시각화"""
    annotated_image = image.copy()
    
    for result in results:
        bbox = result['bbox']
        confidence = result['confidence']
        identity = result.get('identity', 'Unknown')
        
        # 바운딩 박스 그리기
        cv2.rectangle(annotated_image, 
                     (bbox[0], bbox[1]), 
                     (bbox[2], bbox[3]), 
                     (0, 255, 0), 2)
        
        # 라벨 그리기
        label = f"{identity}: {confidence:.2f}"
        cv2.putText(annotated_image, label, 
                   (bbox[0], bbox[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated_image
```

### 성능 측정 예시
```python
import time
import psutil

def measure_performance(func):
    """함수 성능 측정 데코레이터"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        print(f"실행 시간: {execution_time*1000:.2f}ms")
        print(f"메모리 사용: {memory_used/1024/1024:.2f}MB")
        
        return result
    return wrapper
```

## 🔗 의존성

### 내부 의존성
- `../models/`: 얼굴인식 모델들
- `../services/`: 얼굴인식 서비스들
- `common/`: 공통 유틸리티
- `shared/vision_core/`: 비전 알고리즘 공통 기능

### 외부 의존성
```python
# requirements.txt
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
pillow>=8.3.0
psutil>=5.8.0
```

## 🧪 테스트

### 유틸리티 테스트 실행
```bash
# 전체 유틸리티 테스트
python -m pytest tests/test_utils.py -v

# 특정 유틸리티 테스트
python -m pytest tests/test_utils.py::TestImageProcessing -v
python -m pytest tests/test_utils.py::TestVisualization -v
```

### 테스트 예시
```python
def test_image_preprocessing():
    """이미지 전처리 테스트"""
    # 테스트 이미지 생성
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 전처리 수행
    processed = preprocess_image(test_image, target_size=(112, 112))
    
    # 결과 검증
    assert processed.shape == (112, 112, 3)
    assert processed.dtype == np.float32
    assert processed.max() <= 1.0
    assert processed.min() >= 0.0
```

## 📝 파일 관리

### 유틸리티 파일 네이밍 규칙
- **패턴**: `{function}_{category}.py`
- **예시**: `image_processing.py`, `visualization.py`, `data_conversion.py`

### 유틸리티 함수 네이밍 규칙
- **패턴**: `{action}_{object}_{detail}`
- **예시**: `preprocess_image`, `visualize_detection_results`, `convert_bbox_format`

## 🔧 개발 가이드

### 새로운 유틸리티 추가
1. **유틸리티 파일 생성**: `new_utility.py` 파일 생성
2. **함수 구현**: 필요한 유틸리티 함수들 구현
3. **타입 힌트**: 모든 함수에 타입 힌트 추가
4. **문서화**: 함수별 docstring 작성
5. **테스트 작성**: 단위 테스트 작성

### 유틸리티 확장
```python
# 새로운 이미지 처리 유틸리티
def enhance_image_quality(image):
    """이미지 품질 향상"""
    # 노이즈 제거
    denoised = cv2.fastNlMeansDenoisingColored(image)
    
    # 대비 향상
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced
```

## 🐛 문제 해결

### 일반적인 문제들

#### 1. 이미지 로딩 실패
```python
# 해결 방법
def safe_load_image(image_path):
    """안전한 이미지 로딩"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        return image
    except Exception as e:
        print(f"이미지 로딩 오류: {e}")
        return None
```

#### 2. 메모리 부족
```python
# 해결 방법
def optimize_memory_usage(image):
    """메모리 사용량 최적화"""
    # 이미지 크기 줄이기
    if image.shape[0] > 1080 or image.shape[1] > 1920:
        scale = min(1080/image.shape[0], 1920/image.shape[1])
        new_size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
        image = cv2.resize(image, new_size)
    
    # 데이터 타입 최적화
    if image.dtype == np.float64:
        image = image.astype(np.float32)
    
    return image
```

#### 3. 시각화 성능 저하
```python
# 해결 방법
def optimize_visualization(image, results, max_display=10):
    """시각화 성능 최적화"""
    # 표시할 결과 수 제한
    if len(results) > max_display:
        results = sorted(results, key=lambda x: x['confidence'], reverse=True)[:max_display]
    
    # 이미지 크기 조정
    if image.shape[0] > 720:
        scale = 720 / image.shape[0]
        new_size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
        image = cv2.resize(image, new_size)
    
    return visualize_results(image, results)
```

## 📈 성능 최적화

### 성능 최적화 기법
- **배치 처리**: 여러 이미지를 한 번에 처리
- **메모리 풀링**: 메모리 재사용으로 할당 오버헤드 감소
- **캐싱**: 자주 사용되는 결과 캐싱
- **병렬 처리**: 멀티스레딩/멀티프로세싱 활용

### 최적화 예시
```python
from functools import lru_cache
import multiprocessing as mp

@lru_cache(maxsize=128)
def cached_image_processing(image_hash):
    """캐시된 이미지 처리"""
    # 이미지 처리 로직
    pass

def parallel_image_processing(images):
    """병렬 이미지 처리"""
    with mp.Pool() as pool:
        results = pool.map(process_single_image, images)
    return results
```

## 🔗 관련 문서

- [얼굴인식 기능 문서](../README.md)
- [모델 문서](../models/README.md)
- [서비스 문서](../services/README.md)
- [Humanoid 도메인 문서](../../README.md)
- [프로젝트 전체 문서](../../../../README.md)

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 이 README 파일 확인
2. 상위 폴더의 README.md 확인
3. 테스트 코드 참조
4. 개발팀에 문의

---

**마지막 업데이트**: 2025-07-04
**버전**: 1.0.0
**작성자**: Vision System Team 