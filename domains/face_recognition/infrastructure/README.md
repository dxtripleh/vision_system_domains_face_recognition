# Face Recognition Infrastructure Module 🏗️

얼굴인식 도메인의 기술적 구현을 담당하는 인프라 계층입니다. 실제 AI 모델, 데이터 저장소, 외부 시스템과의 연동을 구현합니다.

## 📋 목차

- [개요](#개요)
- [폴더 구조](#폴더-구조)
- [핵심 구성 요소](#핵심-구성-요소)
- [사용 방법](#사용-방법)
- [모델 관리](#모델-관리)
- [성능 최적화](#성능-최적화)

## 🎯 개요

### 무엇인가요?
Infrastructure 모듈은 Core 모듈의 **추상적인 인터페이스를 구체적으로 구현**합니다. 실제 AI 모델 실행, 데이터베이스 연동, 파일 시스템 접근 등의 기술적 세부사항을 처리합니다.

### Core와의 관계
```
Core (추상화)           Infrastructure (구현)
├── PersonRepository → ├── DatabasePersonStorage
├── FaceDetector     → ├── RetinaFaceDetector
├── FaceRecognizer   → ├── ArcFaceRecognizer
└── EmbeddingStorage → └── VectorDatabaseStorage
```

## 📁 폴더 구조

```
infrastructure/
├── README.md               # 👈 현재 파일
├── models/                 # AI 모델 구현체들
│   ├── detectors/         # 얼굴 검출 모델들
│   │   ├── retinaface_detector.py
│   │   ├── mtcnn_detector.py
│   │   └── yolo_face_detector.py
│   ├── recognizers/       # 얼굴 인식 모델들
│   │   ├── arcface_recognizer.py
│   │   ├── facenet_recognizer.py
│   │   └── insightface_recognizer.py
│   └── model_factory.py   # 모델 팩토리
├── storage/               # 데이터 저장 구현체들
│   ├── database/         # 데이터베이스 저장소
│   │   ├── sqlite_storage.py
│   │   └── postgresql_storage.py
│   ├── vector_db/        # 벡터 데이터베이스
│   │   ├── faiss_storage.py
│   │   ├── pinecone_storage.py
│   │   └── chroma_storage.py
│   └── file_storage/     # 파일 시스템 저장소
│       ├── local_file_storage.py
│       └── s3_storage.py
└── detection_engines/     # 검출 엔진 구현체들
    ├── opencv_engine.py
    ├── onnx_engine.py
    └── tensorrt_engine.py
```

## 🔧 핵심 구성 요소

### 1. Models (AI 모델) 🤖

#### 얼굴 검출 모델들
다양한 얼굴 검출 모델을 지원합니다:

```python
from domains.face_recognition.infrastructure.models import ModelFactory

# 모델 팩토리를 통한 모델 생성
factory = ModelFactory()

# RetinaFace 검출기 (높은 정확도)
detector = factory.create_detector('retinaface')

# MTCNN 검출기 (빠른 속도)
detector = factory.create_detector('mtcnn')

# YOLO Face 검출기 (실시간 처리)
detector = factory.create_detector('yolo_face')
```

#### 얼굴 인식 모델들
다양한 얼굴 인식 모델을 지원합니다:

```python
# ArcFace 인식기 (높은 정확도)
recognizer = factory.create_recognizer('arcface')

# FaceNet 인식기 (범용성)
recognizer = factory.create_recognizer('facenet')

# InsightFace 인식기 (최신 기술)
recognizer = factory.create_recognizer('insightface')
```

#### 모델 성능 비교

| 모델 | 정확도 | 속도 | 메모리 사용량 | 권장 용도 |
|------|--------|------|---------------|-----------|
| RetinaFace | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 높은 정확도 필요 |
| MTCNN | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 균형잡힌 성능 |
| YOLO Face | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 실시간 처리 |

### 2. Storage (데이터 저장소) 💾

#### 관계형 데이터베이스
인물 정보와 메타데이터를 저장합니다:

```python
from domains.face_recognition.infrastructure.storage import DatabasePersonStorage

# SQLite 저장소 (개발/테스트용)
storage = DatabasePersonStorage('sqlite', 'face_recognition.db')

# PostgreSQL 저장소 (운영용)
storage = DatabasePersonStorage('postgresql', {
    'host': 'localhost',
    'port': 5432,
    'database': 'face_recognition',
    'username': 'user',
    'password': 'password'
})

# 인물 저장
person = Person.create_new("홍길동")
success = storage.save(person)
```

#### 벡터 데이터베이스
얼굴 임베딩을 효율적으로 저장하고 검색합니다:

```python
from domains.face_recognition.infrastructure.storage import VectorDatabaseStorage

# FAISS 벡터 저장소 (로컬)
vector_storage = VectorDatabaseStorage('faiss', {
    'index_type': 'IVF',
    'dimension': 512,
    'metric': 'cosine'
})

# Pinecone 벡터 저장소 (클라우드)
vector_storage = VectorDatabaseStorage('pinecone', {
    'api_key': 'your_api_key',
    'environment': 'us-west1-gcp',
    'index_name': 'face-embeddings'
})

# 임베딩 저장 및 검색
vector_storage.store_embedding(person_id, face_embedding)
similar_persons = vector_storage.search_similar(query_embedding, top_k=5)
```

#### 벡터 데이터베이스 비교

| 저장소 | 성능 | 확장성 | 비용 | 권장 용도 |
|--------|------|--------|------|-----------|
| FAISS | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 로컬/소규모 |
| Pinecone | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 클라우드/대규모 |
| Chroma | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 중간 규모 |

### 3. Detection Engines (검출 엔진) ⚙️

#### 추론 엔진 종류
다양한 추론 엔진을 지원하여 최적의 성능을 제공합니다:

```python
from domains.face_recognition.infrastructure.detection_engines import DetectionEngine

# OpenCV DNN 엔진 (CPU 최적화)
engine = DetectionEngine('opencv', {
    'backend': 'opencv',
    'target': 'cpu'
})

# ONNX Runtime 엔진 (범용성)
engine = DetectionEngine('onnx', {
    'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
    'optimization_level': 'all'
})

# TensorRT 엔진 (NVIDIA GPU 최적화)
engine = DetectionEngine('tensorrt', {
    'precision': 'fp16',
    'max_batch_size': 8,
    'workspace_size': 1024 * 1024 * 1024  # 1GB
})
```

## 💡 사용 방법

### 기본 설정 및 초기화

#### 1. 모델 다운로드 및 설정
```python
from domains.face_recognition.infrastructure import setup_models

# 필요한 모델들 자동 다운로드
setup_models([
    'retinaface_resnet50',
    'arcface_r100',
    'face_alignment'
])

# 모델 경로 확인
model_paths = {
    'detector': 'models/weights/retinaface_resnet50.onnx',
    'recognizer': 'models/weights/arcface_r100.onnx',
    'landmark': 'models/weights/face_alignment.onnx'
}
```

#### 2. 전체 파이프라인 구성
```python
from domains.face_recognition.infrastructure import FaceRecognitionPipeline

# 파이프라인 설정
config = {
    'detector': {
        'model': 'retinaface',
        'confidence_threshold': 0.8,
        'nms_threshold': 0.4
    },
    'recognizer': {
        'model': 'arcface',
        'embedding_size': 512
    },
    'storage': {
        'person_db': 'sqlite:face_recognition.db',
        'vector_db': 'faiss:face_embeddings.index'
    }
}

# 파이프라인 초기화
pipeline = FaceRecognitionPipeline(config)

# 이미지 처리
import cv2
image = cv2.imread('test_image.jpg')
result = pipeline.process_image(image)

print(f"검출된 얼굴 수: {len(result.faces)}")
for face in result.faces:
    if face.person_id:
        print(f"인식됨: {face.person_id} (신뢰도: {face.confidence:.2f})")
```

### 개별 컴포넌트 사용

#### 1. 얼굴 검출만 사용
```python
from domains.face_recognition.infrastructure.models.detectors import RetinaFaceDetector

# 검출기 초기화
detector = RetinaFaceDetector(
    model_path='models/weights/retinaface_resnet50.onnx',
    confidence_threshold=0.8
)

# 얼굴 검출
detections = detector.detect(image)
print(f"검출된 얼굴: {len(detections)}")

for detection in detections:
    bbox = detection.bbox
    confidence = detection.confidence
    landmarks = detection.landmarks
    print(f"위치: {bbox}, 신뢰도: {confidence:.2f}")
```

#### 2. 얼굴 인식만 사용
```python
from domains.face_recognition.infrastructure.models.recognizers import ArcFaceRecognizer

# 인식기 초기화
recognizer = ArcFaceRecognizer(
    model_path='models/weights/arcface_r100.onnx'
)

# 얼굴 임베딩 추출
face_crop = image[y:y+h, x:x+w]  # 얼굴 영역 자르기
embedding = recognizer.extract_embedding(face_crop)
print(f"임베딩 크기: {embedding.shape}")

# 임베딩 비교
similarity = recognizer.compare_embeddings(embedding1, embedding2)
print(f"유사도: {similarity:.2f}")
```

## 🔧 모델 관리

### 모델 다운로드 및 설치

#### 자동 다운로드
```python
from domains.face_recognition.infrastructure.models import ModelDownloader

downloader = ModelDownloader()

# 사용 가능한 모델 목록
available_models = downloader.list_available_models()
print("사용 가능한 모델들:")
for model in available_models:
    print(f"- {model['name']}: {model['description']}")

# 특정 모델 다운로드
downloader.download_model('retinaface_resnet50')
downloader.download_model('arcface_r100')

# 모든 권장 모델 다운로드
downloader.download_recommended_models()
```

#### 수동 모델 설치
```bash
# 스크립트를 통한 모델 다운로드
python scripts/download_models.py --model retinaface_resnet50
python scripts/download_models.py --model arcface_r100
python scripts/download_models.py --all
```

### 모델 성능 벤치마크

#### 성능 테스트 실행
```python
from domains.face_recognition.infrastructure.benchmarks import ModelBenchmark

benchmark = ModelBenchmark()

# 검출 모델 벤치마크
detection_results = benchmark.benchmark_detectors([
    'retinaface',
    'mtcnn',
    'yolo_face'
], test_dataset_path='datasets/face_recognition/test/')

print("검출 모델 성능:")
for model, metrics in detection_results.items():
    print(f"{model}: mAP={metrics['mAP']:.3f}, FPS={metrics['fps']:.1f}")

# 인식 모델 벤치마크
recognition_results = benchmark.benchmark_recognizers([
    'arcface',
    'facenet',
    'insightface'
], test_dataset_path='datasets/face_recognition/verification/')

print("인식 모델 성능:")
for model, metrics in recognition_results.items():
    print(f"{model}: Accuracy={metrics['accuracy']:.3f}, Speed={metrics['speed_ms']:.1f}ms")
```

## ⚡ 성능 최적화

### GPU 가속 설정

#### CUDA 설정
```python
import torch

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU 사용: {torch.cuda.get_device_name()}")
else:
    device = torch.device('cpu')
    print("CPU 사용")

# 모델을 GPU로 이동
detector = RetinaFaceDetector(device=device)
recognizer = ArcFaceRecognizer(device=device)
```

#### TensorRT 최적화 (NVIDIA GPU)
```python
from domains.face_recognition.infrastructure.optimization import TensorRTOptimizer

# TensorRT 엔진 생성
optimizer = TensorRTOptimizer()

# ONNX 모델을 TensorRT로 변환
trt_engine_path = optimizer.convert_onnx_to_tensorrt(
    onnx_model_path='models/weights/retinaface_resnet50.onnx',
    output_path='models/weights/retinaface_resnet50.trt',
    precision='fp16',  # fp32, fp16, int8
    max_batch_size=8
)

# TensorRT 엔진 사용
detector = RetinaFaceDetector(
    model_path=trt_engine_path,
    engine_type='tensorrt'
)
```

### 배치 처리 최적화

#### 배치 추론
```python
# 여러 이미지를 한 번에 처리
images = [image1, image2, image3, image4]

# 배치 검출
batch_results = detector.detect_batch(images, batch_size=4)

# 배치 인식
face_crops = [crop1, crop2, crop3, crop4]
batch_embeddings = recognizer.extract_embeddings_batch(face_crops, batch_size=4)
```

### 메모리 최적화

#### 메모리 사용량 모니터링
```python
from domains.face_recognition.infrastructure.monitoring import MemoryMonitor

monitor = MemoryMonitor()

# 메모리 사용량 추적 시작
monitor.start_monitoring()

# 처리 작업 수행
result = pipeline.process_image(image)

# 메모리 사용량 확인
memory_stats = monitor.get_memory_stats()
print(f"최대 메모리 사용량: {memory_stats['peak_memory_mb']:.1f}MB")
print(f"현재 메모리 사용량: {memory_stats['current_memory_mb']:.1f}MB")
```

## 🔧 문제 해결

### 자주 발생하는 문제들

#### Q: 모델 로딩이 실패해요
A: 다음을 확인해보세요:
```python
# 모델 파일 존재 확인
import os
model_path = 'models/weights/retinaface_resnet50.onnx'
if not os.path.exists(model_path):
    print("모델 파일이 없습니다. 다운로드를 실행하세요.")
    # python scripts/download_models.py --model retinaface_resnet50

# 모델 파일 무결성 확인
from domains.face_recognition.infrastructure.utils import verify_model_integrity
if not verify_model_integrity(model_path):
    print("모델 파일이 손상되었습니다. 재다운로드하세요.")
```

#### Q: GPU 메모리 부족 오류가 발생해요
A: 메모리 사용량을 줄여보세요:
```python
# 배치 크기 줄이기
detector = RetinaFaceDetector(batch_size=1)

# 이미지 크기 줄이기
import cv2
image = cv2.resize(image, (640, 480))

# 메모리 정리
torch.cuda.empty_cache()
```

#### Q: 추론 속도가 너무 느려요
A: 최적화 방법들:
```python
# 1. TensorRT 사용 (NVIDIA GPU)
detector = RetinaFaceDetector(engine_type='tensorrt')

# 2. 이미지 크기 최적화
optimal_size = detector.get_optimal_input_size()
image = cv2.resize(image, optimal_size)

# 3. 신뢰도 임계값 조정
detector.set_confidence_threshold(0.9)  # 높은 임계값으로 후처리 시간 단축
```

## 📚 참고 자료

### 지원되는 모델 형식
- **ONNX**: 범용 모델 형식, 다양한 프레임워크 지원
- **PyTorch**: PyTorch 네이티브 모델
- **TensorRT**: NVIDIA GPU 최적화 모델
- **OpenVINO**: Intel CPU/GPU 최적화 모델

### 성능 최적화 가이드
- [GPU 가속 설정 가이드](docs/gpu_acceleration.md)
- [TensorRT 최적화 가이드](docs/tensorrt_optimization.md)
- [메모리 최적화 가이드](docs/memory_optimization.md)

---

**버전**: 0.1.0  
**최종 업데이트**: 2025-06-28  
**작성자**: Vision System Team 