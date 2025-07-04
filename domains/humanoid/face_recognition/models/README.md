# Models - 얼굴인식 모델 모듈

## 📋 개요

이 폴더는 얼굴인식 기능에 필요한 모든 AI 모델 클래스들을 포함합니다. 얼굴 검출과 얼굴 인식을 위한 ONNX 기반 딥러닝 모델들을 관리합니다.

## 🏗️ 폴더 구조

```
models/
├── __init__.py                    # 모델 패키지 초기화
├── README.md                      # 이 파일
├── face_detection_model.py        # 얼굴 검출 모델
└── face_recognition_model.py      # 얼굴 인식 모델
```

## 🔍 포함된 모델들

### 1. FaceDetectionModel (얼굴 검출 모델)
- **파일**: `face_detection_model.py`
- **목적**: 이미지에서 얼굴 영역을 찾아내는 모델
- **기술**: RetinaFace, MTCNN 기반 ONNX 모델
- **입력**: 이미지 (BGR 형식)
- **출력**: 얼굴 바운딩 박스, 랜드마크, 신뢰도 점수

### 2. FaceRecognitionModel (얼굴 인식 모델)
- **파일**: `face_recognition_model.py`
- **목적**: 얼굴 이미지에서 특징 벡터(임베딩)를 추출하는 모델
- **기술**: ArcFace, FaceNet 기반 ONNX 모델
- **입력**: 얼굴 이미지 (정규화된 크기)
- **출력**: 512차원 특징 벡터

## 🚀 사용법

### 기본 사용법
```python
from domains.humanoid.face_recognition.models import FaceDetectionModel, FaceRecognitionModel

# 모델 초기화
detection_model = FaceDetectionModel()
recognition_model = FaceRecognitionModel()

# 얼굴 검출
faces = detection_model.detect(image)

# 얼굴 인식 (특징 추출)
for face in faces:
    embedding = recognition_model.extract_features(face['image'])
```

### 고급 사용법
```python
# 커스텀 설정으로 모델 초기화
detection_model = FaceDetectionModel(
    model_path="models/weights/custom_face_detection.onnx",
    config={
        'confidence_threshold': 0.7,
        'min_face_size': 80
    }
)

# 배치 처리
embeddings = recognition_model.extract_features_batch(face_images)
```

## 🔧 모델 설정

### FaceDetectionModel 설정
```python
detection_config = {
    'confidence_threshold': 0.5,    # 검출 신뢰도 임계값
    'min_face_size': 80,           # 최소 얼굴 크기 (픽셀)
    'max_faces': 10,               # 최대 검출 얼굴 수
    'input_size': (640, 640),      # 모델 입력 크기
    'nms_threshold': 0.4           # NMS 임계값
}
```

### FaceRecognitionModel 설정
```python
recognition_config = {
    'embedding_dim': 512,          # 특징 벡터 차원
    'input_size': (112, 112),      # 모델 입력 크기
    'normalize_embeddings': True,  # 특징 벡터 정규화 여부
    'distance_metric': 'cosine'    # 거리 측정 방법
}
```

## 📊 모델 성능

### FaceDetectionModel 성능
- **검출 정확도**: 95% 이상
- **처리 속도**: 30 FPS (GPU)
- **최소 얼굴 크기**: 20x20 픽셀
- **메모리 사용량**: 1GB 이하

### FaceRecognitionModel 성능
- **인식 정확도**: 90% 이상
- **처리 속도**: 100 FPS (GPU)
- **특징 벡터 차원**: 512
- **메모리 사용량**: 500MB 이하

## 🔗 의존성

### 내부 의존성
- `common/`: 공통 유틸리티
- `shared/vision_core/`: 비전 알고리즘 공통 기능

### 외부 의존성
```python
# requirements.txt
onnxruntime>=1.12.0
opencv-python>=4.5.0
numpy>=1.21.0
```

## 🧪 테스트

### 모델 테스트 실행
```bash
# 전체 모델 테스트
python -m pytest tests/test_models.py -v

# 특정 모델 테스트
python -m pytest tests/test_models.py::TestFaceDetectionModel -v
python -m pytest tests/test_models.py::TestFaceRecognitionModel -v
```

### 테스트 예시
```python
def test_face_detection_model():
    """얼굴 검출 모델 테스트"""
    model = FaceDetectionModel()
    
    # 테스트 이미지 로드
    test_image = load_test_image()
    
    # 얼굴 검출 수행
    faces = model.detect(test_image)
    
    # 결과 검증
    assert len(faces) > 0
    assert all('bbox' in face for face in faces)
    assert all('confidence' in face for face in faces)
```

## 📝 모델 파일 관리

### 모델 파일 위치
```
models/weights/
├── face_detection_retinaface.onnx      # 얼굴 검출 모델
├── face_recognition_arcface.onnx       # 얼굴 인식 모델
└── face_recognition_facenet.onnx       # 대체 얼굴 인식 모델
```

### 모델 파일 네이밍 규칙
- **패턴**: `{task}_{architecture}_{dataset}_{date}.onnx`
- **예시**: `face_detection_retinaface_widerface_20240101.onnx`

### 모델 버전 관리
```python
# 모델 버전 정보
MODEL_VERSIONS = {
    'face_detection': {
        'current': 'retinaface_v1.0',
        'backup': 'mtcnn_v0.9',
        'experimental': 'yolov8_face_v1.1'
    },
    'face_recognition': {
        'current': 'arcface_v1.0',
        'backup': 'facenet_v0.9',
        'experimental': 'cosface_v1.1'
    }
}
```

## 🔧 개발 가이드

### 새로운 모델 추가
1. **모델 클래스 생성**: `new_model.py` 파일 생성
2. **기본 인터페이스 구현**: `detect()` 또는 `extract_features()` 메서드 구현
3. **설정 관리**: 모델별 설정 클래스 구현
4. **테스트 작성**: 단위 테스트 및 통합 테스트 작성
5. **문서화**: 클래스 및 메서드 문서화

### 모델 최적화
```python
# GPU 최적화
model = FaceDetectionModel(
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# 배치 처리 최적화
model.set_batch_size(4)

# 메모리 최적화
model.enable_memory_optimization()
```

## 🐛 문제 해결

### 일반적인 문제들

#### 1. 모델 파일을 찾을 수 없음
```python
# 해결 방법
model_path = "models/weights/face_detection.onnx"
if not os.path.exists(model_path):
    # 기본 모델 다운로드 또는 경로 수정
    model_path = get_default_model_path()
```

#### 2. GPU 메모리 부족
```python
# 해결 방법
model = FaceDetectionModel(
    providers=['CPUExecutionProvider'],  # CPU만 사용
    batch_size=1  # 배치 크기 줄이기
)
```

#### 3. 추론 속도가 느림
```python
# 해결 방법
model = FaceDetectionModel(
    input_size=(320, 320),  # 입력 크기 줄이기
    optimization_level='ORT_ENABLE_ALL'  # 최적화 레벨 높이기
)
```

## 📈 성능 모니터링

### 성능 지표
- **추론 시간**: 평균 처리 시간 (ms)
- **메모리 사용량**: GPU/CPU 메모리 사용량
- **정확도**: 검출/인식 정확도
- **처리량**: FPS (Frames Per Second)

### 성능 측정
```python
import time

# 성능 측정
start_time = time.time()
result = model.detect(image)
inference_time = time.time() - start_time

print(f"추론 시간: {inference_time*1000:.2f}ms")
print(f"FPS: {1.0/inference_time:.1f}")
```

## 🔗 관련 문서

- [얼굴인식 기능 문서](../README.md)
- [Humanoid 도메인 문서](../../README.md)
- [프로젝트 전체 문서](../../../../README.md)
- [공유 모듈 문서](../../../../shared/README.md)

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