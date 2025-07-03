# 🔍 Face Recognition Domain

> **얼굴인식 도메인** - 얼굴 검출, 인식, 관리를 위한 완전한 솔루션

## 🎯 개요

이 도메인은 **얼굴인식 시스템**의 핵심 기능을 제공합니다. 최신 AI 모델(RetinaFace + ArcFace)을 사용하여 높은 정확도의 얼굴 검출과 인식을 구현합니다.

### 🌟 주요 기능
- 🔍 **실시간 얼굴 검출** - 이미지/비디오에서 얼굴 위치 탐지
- 👤 **얼굴 인식** - 개인 식별을 위한 얼굴 특징 추출 및 매칭
- 📊 **얼굴 품질 평가** - 블러, 조명, 각도 등 품질 검사
- 🗃️ **인물 데이터베이스 관리** - 등록된 인물 정보 저장 및 관리
- ⚡ **GPU/CPU 최적화** - 하드웨어에 따른 자동 최적화

## 🚀 빠른 시작

### 1. 필수 모델 다운로드
```bash
# 프로젝트 루트에서 실행
python scripts/model_management/download_models.py --essential
```

### 2. 기본 사용법
```python
from domains.face_recognition import FaceRecognitionService

# 서비스 초기화
service = FaceRecognitionService()

# 이미지에서 얼굴 검출
faces = service.detect_faces("path/to/image.jpg")

# 얼굴 인식 (임베딩 추출)
embedding = service.extract_embedding(face_image)

# 인물 등록
person_id = service.register_person("홍길동", face_images)

# 얼굴 매칭
matches = service.identify_face(face_image)
```

## 📁 프로젝트 구조

```
domains/face_recognition/
├── 📄 README.md                    # 이 파일
├── 📄 STRUCTURE.md                 # 상세 구조 설명
├── 📄 CHANGELOG.md                 # 변경 이력
├── 📄 __init__.py                  # 도메인 진입점
│
├── 🧠 core/                        # 핵심 비즈니스 로직 (DDD)
│   ├── 📄 README.md                # 핵심 로직 설명
│   ├── 📦 entities/                # 도메인 엔티티
│   │   ├── face.py                 # 얼굴 엔티티
│   │   ├── person.py               # 인물 엔티티
│   │   └── detection_result.py     # 검출 결과 엔티티
│   ├── 🛠️ services/                # 도메인 서비스
│   │   ├── face_detection_service.py
│   │   ├── face_recognition_service.py
│   │   └── face_matching_service.py
│   ├── 💾 repositories/            # 저장소 인터페이스
│   │   ├── face_repository.py
│   │   └── person_repository.py
│   └── 💎 value_objects/           # 값 객체
│       ├── face_embedding.py
│       ├── bounding_box.py
│       └── confidence_score.py
│
├── 🏗️ infrastructure/              # 기술 구현체
│   ├── 📄 README.md                # 인프라 설명
│   ├── 🤖 models/                  # AI 모델 구현
│   │   ├── retinaface_detector.py  # RetinaFace 검출기
│   │   ├── arcface_recognizer.py   # ArcFace 인식기
│   │   └── base_detector.py        # 기본 검출기 인터페이스
│   ├── 💾 storage/                 # 데이터 저장 구현
│   │   ├── file_face_repository.py
│   │   ├── database_person_repository.py
│   │   └── memory_cache.py
│   └── 🔧 detection_engines/       # 검출 엔진
│       ├── onnx_engine.py
│       ├── opencv_engine.py
│       └── tensorrt_engine.py
│
├── 🌐 interfaces/                  # 외부 인터페이스
│   ├── 📄 README.md                # 인터페이스 설명
│   ├── 🔗 api/                     # REST API
│   │   ├── face_detection_api.py
│   │   ├── face_recognition_api.py
│   │   └── person_management_api.py
│   └── 💻 cli/                     # 명령행 인터페이스
│       ├── face_cli.py
│       └── batch_processor.py
│
├── ⚙️ config/                      # 설정 파일
│   ├── models.yaml                 # 모델 설정
│   ├── hardware.yaml               # 하드웨어 최적화 설정
│   └── thresholds.yaml             # 임계값 설정
│
└── 🧪 tests/                       # 테스트 코드
    ├── unit/                       # 단위 테스트
    ├── integration/                # 통합 테스트
    └── performance/                # 성능 테스트
```

## 🤖 지원 AI 모델

### 얼굴 검출 모델
| 모델명 | 크기 | 성능 | 용도 |
|--------|------|------|------|
| **SCRFD 10G** | 16.1MB | 고성능 | GPU 환경 |
| **RetinaFace MNet025** | 2.4MB | 균형 | 일반 환경 |
| **OpenCV Haar Cascade** | 0.9MB | 기본 | CPU 백업 |

### 얼굴 인식 모델  
| 모델명 | 크기 | 임베딩 | 정확도 |
|--------|------|--------|--------|
| **ArcFace R100** | 166.3MB | 512D | 99.83% |
| **ArcFace R50** | 13.0MB | 512D | 99.75% |
| **MobileFaceNet** | 13.0MB | 128D | 99.40% |

## 🔧 설치 및 설정

### 1. 환경 요구사항
```bash
# Python 3.8+
python --version

# 필수 패키지 설치
pip install -r requirements.txt

# GPU 사용 시 (선택사항)
pip install onnxruntime-gpu
```

### 2. 모델 다운로드
```bash
# 모든 모델 다운로드
python scripts/model_management/download_models.py --all

# 필수 모델만 다운로드
python scripts/model_management/download_models.py --essential

# 특정 모델 다운로드
python scripts/model_management/download_models.py --model scrfd_10g_bnkps
```

### 3. 모델 테스트
```bash
# 모든 모델 테스트
python scripts/test_models.py

# 상세 로그와 함께 테스트
python scripts/test_models.py --verbose
```

## 📚 상세 사용법

### 🔍 얼굴 검출

```python
from domains.face_recognition.core.services import FaceDetectionService

# 서비스 초기화
detector = FaceDetectionService(model_name="scrfd_10g")

# 이미지에서 얼굴 검출
import cv2
image = cv2.imread("photo.jpg")
faces = detector.detect_faces(image)

# 결과 확인
for face in faces:
    print(f"얼굴 위치: {face.bounding_box}")
    print(f"신뢰도: {face.confidence}")
    print(f"랜드마크: {face.landmarks}")
```

### 👤 얼굴 인식

```python
from domains.face_recognition.core.services import FaceRecognitionService

# 서비스 초기화
recognizer = FaceRecognitionService(model_name="arcface_r100_buffalo_l")

# 얼굴 임베딩 추출
face_image = cv2.imread("face.jpg")
embedding = recognizer.extract_embedding(face_image)

# 임베딩 비교
similarity = recognizer.compare_embeddings(embedding1, embedding2)
print(f"유사도: {similarity:.3f}")
```

### 🗃️ 인물 관리

```python
from domains.face_recognition.core.services import PersonManagementService

# 서비스 초기화
person_service = PersonManagementService()

# 새 인물 등록
person_id = person_service.register_person(
    name="홍길동",
    face_images=[image1, image2, image3],
    metadata={"department": "개발팀"}
)

# 인물 검색
person = person_service.get_person(person_id)
print(f"이름: {person.name}")
print(f"등록된 얼굴 수: {len(person.face_embeddings)}")
```

### 🔄 실시간 처리

```python
from domains.face_recognition import FaceRecognitionPipeline

# 파이프라인 초기화
pipeline = FaceRecognitionPipeline(
    detection_model="retinaface_mnet025",
    recognition_model="arcface_r50_buffalo_s"
)

# 웹캠 실시간 처리
import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 얼굴 검출 및 인식
    results = pipeline.process_frame(frame)
    
    # 결과 표시
    for result in results:
        # 바운딩 박스 그리기
        cv2.rectangle(frame, result.bbox.top_left, result.bbox.bottom_right, (0, 255, 0), 2)
        
        # 인물 이름 표시
        if result.person:
            cv2.putText(frame, result.person.name, result.bbox.top_left, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## ⚙️ 설정 가이드

### 하드웨어별 최적화

```yaml
# config/hardware.yaml
gpu_high_performance:
  detection_model: "scrfd_10g"
  recognition_model: "arcface_r100_buffalo_l"
  batch_size: 4
  precision: "fp16"

cpu_optimized:
  detection_model: "retinaface_mnet025"
  recognition_model: "mobilefacenet_buffalo_s"
  batch_size: 1
  precision: "fp32"
```

### 성능 임계값 조정

```yaml
# config/thresholds.yaml
detection:
  min_confidence: 0.5      # 검출 최소 신뢰도
  max_inference_time_ms: 100

recognition:
  min_similarity: 0.6      # 인식 최소 유사도
  max_inference_time_ms: 150
```

## 🚨 문제 해결

### 자주 발생하는 문제

**1. 모델 로딩 실패**
```bash
# 모델 파일 확인
ls -la models/weights/

# 모델 재다운로드
python scripts/model_management/download_models.py --model scrfd_10g_bnkps --force
```

**2. GPU 메모리 부족**
```python
# 배치 크기 줄이기
config = {
    "batch_size": 1,  # 기본값: 4
    "precision": "fp16"  # 메모리 사용량 절반
}
```

**3. 느린 추론 속도**
```python
# 경량 모델 사용
detector = FaceDetectionService(model_name="retinaface_mnet025")
recognizer = FaceRecognitionService(model_name="mobilefacenet_buffalo_s")
```

### 성능 최적화 팁

1. **GPU 사용 시**: SCRFD + ArcFace R100 조합
2. **CPU 사용 시**: RetinaFace MNet025 + MobileFaceNet 조합  
3. **실시간 처리**: 배치 크기 1, FP16 정밀도 사용
4. **높은 정확도**: ArcFace R100 모델 사용
5. **메모리 절약**: MobileFaceNet 모델 사용

## 📊 성능 벤치마크

### 추론 시간 (CPU 환경)
| 모델 | 검출 시간 | 인식 시간 | 총 시간 |
|------|-----------|-----------|---------|
| SCRFD + ArcFace R100 | 435ms | 351ms | 786ms |
| RetinaFace + ArcFace R50 | 82ms | 84ms | 166ms |
| RetinaFace + MobileFaceNet | 82ms | 38ms | 120ms |

### 메모리 사용량
| 모델 조합 | GPU 메모리 | RAM |
|-----------|------------|-----|
| 고성능 (SCRFD + R100) | 1.5GB | 2GB |
| 균형 (RetinaFace + R50) | 800MB | 1GB |
| 경량 (RetinaFace + MobileFaceNet) | 400MB | 512MB |

## 🔗 관련 링크

- **[STRUCTURE.md](./STRUCTURE.md)** - 상세 구조 설명
- **[CHANGELOG.md](./CHANGELOG.md)** - 변경 이력
- **[API 문서](./interfaces/README.md)** - REST API 가이드
- **[모델 가이드](./infrastructure/README.md)** - AI 모델 상세 정보

## 🤝 기여하기

1. 이슈 리포트: 버그나 개선사항 제안
2. 풀 리퀘스트: 코드 기여
3. 문서 개선: README나 주석 개선
4. 테스트 추가: 새로운 테스트 케이스 작성

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**💡 도움이 필요하신가요?**
- 📧 이메일: support@vision-system.com
- 💬 이슈: GitHub Issues 탭에서 질문
- 📚 문서: [전체 문서 보기](../../README.md) 