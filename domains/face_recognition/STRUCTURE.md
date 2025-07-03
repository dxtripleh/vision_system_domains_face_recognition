# 🏗️ Face Recognition Domain - 상세 구조 가이드

> **Domain-Driven Design (DDD)** 패턴을 따르는 얼굴인식 도메인의 상세 구조 설명

## 📋 목차

1. [아키텍처 개요](#-아키텍처-개요)
2. [Core Layer (핵심 계층)](#-core-layer-핵심-계층)
3. [Infrastructure Layer (인프라 계층)](#-infrastructure-layer-인프라-계층)
4. [Interface Layer (인터페이스 계층)](#-interface-layer-인터페이스-계층)
5. [Configuration (설정)](#-configuration-설정)
6. [데이터 흐름](#-데이터-흐름)
7. [의존성 관계](#-의존성-관계)

## 🏛️ 아키텍처 개요

### DDD 헥사고날 아키텍처
```
┌─────────────────────────────────────────────────────────────┐
│                    🌐 Interfaces Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ REST API    │  │ CLI         │  │ WebSocket   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                     🧠 Core Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Entities    │  │ Services    │  │ Repositories│         │
│  │             │  │             │  │ (Interface) │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐                                           │
│  │ Value       │                                           │
│  │ Objects     │                                           │
│  └─────────────┘                                           │
├─────────────────────────────────────────────────────────────┤
│                  🏗️ Infrastructure Layer                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ AI Models   │  │ Storage     │  │ Detection   │         │
│  │             │  │             │  │ Engines     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 계층별 책임
- **Interface**: 외부 세계와의 통신 (API, CLI, UI)
- **Core**: 비즈니스 로직과 도메인 규칙
- **Infrastructure**: 기술적 구현체 (AI 모델, 데이터베이스, 파일시스템)

## 🧠 Core Layer (핵심 계층)

### 📦 Entities (엔티티)
> 도메인의 핵심 객체들 - 고유 식별자를 가지며 생명주기가 있는 객체

#### `face.py` - Face 엔티티
```python
class Face:
    """얼굴 엔티티 - 검출된 개별 얼굴을 나타냄"""
    
    # 속성
    face_id: str              # 고유 식별자
    person_id: Optional[str]  # 연관된 인물 ID
    embedding: FaceEmbedding  # 얼굴 특징 벡터
    confidence: float         # 검출 신뢰도
    bbox: BoundingBox        # 얼굴 위치
    landmarks: List[Point]   # 얼굴 랜드마크
    quality_score: float     # 얼굴 품질 점수
    created_at: datetime     # 생성 시간
    
    # 메서드
    def is_high_quality(self) -> bool
    def calculate_similarity(self, other_face: 'Face') -> float
    def to_dict(self) -> dict
```

#### `person.py` - Person 엔티티
```python
class Person:
    """인물 엔티티 - 등록된 개인을 나타냄"""
    
    # 속성
    person_id: str                    # 고유 식별자
    name: str                        # 이름
    face_embeddings: List[FaceEmbedding]  # 등록된 얼굴들
    metadata: dict                   # 추가 정보
    is_active: bool                  # 활성 상태
    created_at: datetime             # 등록 시간
    updated_at: datetime             # 수정 시간
    
    # 메서드
    def add_face_embedding(self, embedding: FaceEmbedding)
    def get_average_embedding(self) -> FaceEmbedding
    def is_match(self, face_embedding: FaceEmbedding, threshold: float) -> bool
```

#### `detection_result.py` - FaceDetectionResult 엔티티
```python
class FaceDetectionResult:
    """얼굴 검출 결과 엔티티"""
    
    # 속성
    image_id: str                # 이미지 식별자
    faces: List[Face]           # 검출된 얼굴들
    processing_time_ms: float   # 처리 시간
    model_name: str             # 사용된 모델
    image_metadata: dict        # 이미지 메타데이터
    
    # 메서드
    def get_face_count(self) -> int
    def get_highest_confidence_face(self) -> Optional[Face]
    def filter_by_confidence(self, min_confidence: float) -> List[Face]
```

### 🛠️ Services (도메인 서비스)
> 복잡한 비즈니스 로직을 캡슐화하는 서비스들

#### `face_detection_service.py`
```python
class FaceDetectionService:
    """얼굴 검출 서비스"""
    
    def __init__(self, detector: IFaceDetector, quality_checker: IQualityChecker)
    
    # 핵심 메서드
    def detect_faces(self, image: np.ndarray) -> FaceDetectionResult
    def detect_faces_batch(self, images: List[np.ndarray]) -> List[FaceDetectionResult]
    def validate_face_quality(self, face: Face) -> bool
    
    # 비즈니스 규칙
    - 최소 얼굴 크기 검증
    - 품질 점수 계산
    - 중복 검출 제거
```

#### `face_recognition_service.py`
```python
class FaceRecognitionService:
    """얼굴 인식 서비스"""
    
    def __init__(self, recognizer: IFaceRecognizer, person_repo: IPersonRepository)
    
    # 핵심 메서드
    def extract_embedding(self, face_image: np.ndarray) -> FaceEmbedding
    def identify_face(self, face: Face) -> Optional[Person]
    def verify_face(self, face: Face, person_id: str) -> bool
    
    # 비즈니스 규칙
    - 임베딩 정규화
    - 유사도 임계값 적용
    - 다중 얼굴 매칭
```

#### `face_matching_service.py`
```python
class FaceMatchingService:
    """얼굴 매칭 서비스"""
    
    def __init__(self, similarity_calculator: ISimilarityCalculator)
    
    # 핵심 메서드
    def find_matches(self, target_embedding: FaceEmbedding, 
                    candidates: List[Person]) -> List[MatchResult]
    def calculate_similarity(self, emb1: FaceEmbedding, emb2: FaceEmbedding) -> float
    def rank_matches(self, matches: List[MatchResult]) -> List[MatchResult]
```

### 💾 Repositories (저장소 인터페이스)
> 데이터 접근을 추상화하는 인터페이스들

#### `face_repository.py`
```python
class IFaceRepository(ABC):
    """얼굴 저장소 인터페이스"""
    
    @abstractmethod
    def save(self, face: Face) -> str
    
    @abstractmethod
    def find_by_id(self, face_id: str) -> Optional[Face]
    
    @abstractmethod
    def find_by_person_id(self, person_id: str) -> List[Face]
    
    @abstractmethod
    def delete(self, face_id: str) -> bool
```

#### `person_repository.py`
```python
class IPersonRepository(ABC):
    """인물 저장소 인터페이스"""
    
    @abstractmethod
    def save(self, person: Person) -> str
    
    @abstractmethod
    def find_by_id(self, person_id: str) -> Optional[Person]
    
    @abstractmethod
    def find_by_name(self, name: str) -> List[Person]
    
    @abstractmethod
    def get_all_active(self) -> List[Person]
    
    @abstractmethod
    def update(self, person: Person) -> bool
    
    @abstractmethod
    def delete(self, person_id: str) -> bool
```

### 💎 Value Objects (값 객체)
> 불변 객체로 도메인 개념을 표현

#### `face_embedding.py`
```python
@dataclass(frozen=True)
class FaceEmbedding:
    """얼굴 특징 벡터 값 객체"""
    
    vector: Tuple[float, ...]  # 임베딩 벡터 (불변)
    dimension: int             # 차원 수
    model_name: str           # 생성 모델
    
    def __post_init__(self):
        # 벡터 정규화 및 검증
        
    def cosine_similarity(self, other: 'FaceEmbedding') -> float
    def euclidean_distance(self, other: 'FaceEmbedding') -> float
```

#### `bounding_box.py`
```python
@dataclass(frozen=True)
class BoundingBox:
    """바운딩 박스 값 객체"""
    
    x: int      # 좌상단 X
    y: int      # 좌상단 Y  
    width: int  # 너비
    height: int # 높이
    
    @property
    def area(self) -> int
    
    @property
    def center(self) -> Point
    
    def contains_point(self, point: Point) -> bool
    def overlaps_with(self, other: 'BoundingBox') -> bool
```

#### `confidence_score.py`
```python
@dataclass(frozen=True)
class ConfidenceScore:
    """신뢰도 점수 값 객체"""
    
    value: float  # 0.0 ~ 1.0
    
    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("신뢰도는 0.0~1.0 사이여야 합니다")
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool
    def is_low_confidence(self, threshold: float = 0.3) -> bool
```

## 🏗️ Infrastructure Layer (인프라 계층)

### 🤖 Models (AI 모델 구현체)
> Core 계층의 인터페이스를 구현하는 실제 AI 모델들

#### `retinaface_detector.py`
```python
class RetinaFaceDetector(IFaceDetector):
    """RetinaFace 기반 얼굴 검출기"""
    
    def __init__(self, model_path: str, device: str = "auto")
    
    # 인터페이스 구현
    def detect(self, image: np.ndarray) -> List[Face]
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Face]]
    
    # 내부 메서드
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray
    def _postprocess_results(self, raw_output: np.ndarray) -> List[Face]
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]
```

#### `arcface_recognizer.py`
```python
class ArcFaceRecognizer(IFaceRecognizer):
    """ArcFace 기반 얼굴 인식기"""
    
    def __init__(self, model_path: str, device: str = "auto")
    
    # 인터페이스 구현
    def extract_embedding(self, face_image: np.ndarray) -> FaceEmbedding
    def extract_embeddings_batch(self, face_images: List[np.ndarray]) -> List[FaceEmbedding]
    
    # 내부 메서드
    def _align_face(self, face_image: np.ndarray, landmarks: List[Point]) -> np.ndarray
    def _normalize_embedding(self, raw_embedding: np.ndarray) -> np.ndarray
```

### 💾 Storage (데이터 저장 구현체)
> Repository 인터페이스의 실제 구현체들

#### `file_face_repository.py`
```python
class FileFaceRepository(IFaceRepository):
    """파일 시스템 기반 얼굴 저장소"""
    
    def __init__(self, storage_path: str)
    
    def save(self, face: Face) -> str:
        # JSON 파일로 저장
        
    def find_by_id(self, face_id: str) -> Optional[Face]:
        # 파일에서 로드
```

#### `database_person_repository.py`
```python
class DatabasePersonRepository(IPersonRepository):
    """데이터베이스 기반 인물 저장소"""
    
    def __init__(self, connection_string: str)
    
    def save(self, person: Person) -> str:
        # SQL 데이터베이스에 저장
```

### 🔧 Detection Engines (검출 엔진)
> 다양한 추론 엔진 지원

#### `onnx_engine.py`
```python
class ONNXEngine(IInferenceEngine):
    """ONNX Runtime 기반 추론 엔진"""
    
    def __init__(self, model_path: str, providers: List[str])
    
    def load_model(self) -> bool
    def run_inference(self, input_data: np.ndarray) -> np.ndarray
    def get_model_info(self) -> dict
```

## 🌐 Interface Layer (인터페이스 계층)

### 🔗 API (REST API)
> 외부 시스템과의 HTTP 통신

#### `face_detection_api.py`
```python
@router.post("/detect")
async def detect_faces(image: UploadFile):
    """얼굴 검출 API 엔드포인트"""
    
    # 1. 이미지 검증
    # 2. 서비스 호출
    # 3. 결과 반환
```

#### `face_recognition_api.py`
```python
@router.post("/recognize")
async def recognize_face(image: UploadFile):
    """얼굴 인식 API 엔드포인트"""

@router.post("/register")
async def register_person(person_data: PersonCreateRequest):
    """인물 등록 API 엔드포인트"""
```

### 💻 CLI (명령행 인터페이스)
> 배치 처리 및 관리 도구

#### `face_cli.py`
```python
class FaceCLI:
    """얼굴인식 CLI 도구"""
    
    def detect_command(self, image_path: str)
    def register_command(self, name: str, images_dir: str)
    def batch_process_command(self, input_dir: str, output_dir: str)
```

## ⚙️ Configuration (설정)

### `models.yaml` - 모델 설정
```yaml
face_detection:
  default_model: "scrfd_10g"
  models:
    scrfd_10g:
      model_path: "models/weights/face_detection_scrfd_10g_20250628.onnx"
      framework: "onnx"
      input_size: [640, 640]
      confidence_threshold: 0.5
```

### `hardware.yaml` - 하드웨어 최적화
```yaml
gpu_high_performance:
  detection_model: "scrfd_10g"
  recognition_model: "arcface_r100_buffalo_l"
  batch_size: 4
  precision: "fp16"
```

### `thresholds.yaml` - 임계값 설정
```yaml
detection:
  min_confidence: 0.5
  max_inference_time_ms: 100

recognition:
  min_similarity: 0.6
  max_inference_time_ms: 150
```

## 🔄 데이터 흐름

### 1. 얼굴 검출 플로우
```
이미지 입력
    ↓
Interface Layer (API/CLI)
    ↓
Core Layer (FaceDetectionService)
    ↓
Infrastructure Layer (RetinaFaceDetector)
    ↓
ONNX Engine (추론 실행)
    ↓
결과 반환 (Face 엔티티들)
```

### 2. 얼굴 인식 플로우
```
얼굴 이미지 입력
    ↓
Interface Layer (API/CLI)
    ↓
Core Layer (FaceRecognitionService)
    ↓
Infrastructure Layer (ArcFaceRecognizer)
    ↓
ONNX Engine (임베딩 추출)
    ↓
Core Layer (매칭 로직)
    ↓
Repository (인물 검색)
    ↓
결과 반환 (Person 엔티티)
```

### 3. 인물 등록 플로우
```
인물 정보 + 얼굴 이미지들
    ↓
Interface Layer (API/CLI)
    ↓
Core Layer (PersonManagementService)
    ↓
각 얼굴 이미지 → FaceRecognitionService
    ↓
임베딩 추출 및 품질 검증
    ↓
Person 엔티티 생성
    ↓
Repository (저장)
    ↓
person_id 반환
```

## 🔗 의존성 관계

### 의존성 방향 (DDD 원칙)
```
Interface Layer
    ↓ (의존)
Core Layer
    ↑ (구현)
Infrastructure Layer
```

### 핵심 원칙
1. **Interface → Core**: 인터페이스는 핵심 서비스를 사용
2. **Core ← Infrastructure**: 인프라는 핵심 인터페이스를 구현
3. **Core는 독립적**: 외부 기술에 의존하지 않음
4. **의존성 역전**: 추상화에 의존, 구체적 구현에 의존하지 않음

### 의존성 주입 예시
```python
# 잘못된 방법 (강한 결합)
class FaceDetectionService:
    def __init__(self):
        self.detector = RetinaFaceDetector()  # 구체적 구현에 의존

# 올바른 방법 (느슨한 결합)
class FaceDetectionService:
    def __init__(self, detector: IFaceDetector):  # 추상화에 의존
        self.detector = detector
        
# 의존성 주입
detector = RetinaFaceDetector()
service = FaceDetectionService(detector)
```

## 📊 성능 고려사항

### 메모리 관리
- **모델 로딩**: 지연 로딩으로 메모리 절약
- **배치 처리**: 적절한 배치 크기로 메모리 효율성 확보
- **캐싱**: 자주 사용되는 임베딩 캐싱

### 확장성
- **모델 교체**: 인터페이스 기반으로 쉬운 모델 교체
- **저장소 교체**: Repository 패턴으로 저장소 변경 용이
- **수평 확장**: 서비스 단위로 독립적 확장 가능

### 보안
- **데이터 암호화**: 임베딩 및 개인정보 암호화
- **접근 제어**: API 레벨에서 인증/인가
- **감사 로그**: 모든 작업에 대한 로그 기록

---

이 구조는 **확장 가능하고 유지보수가 용이한** 얼굴인식 시스템을 위해 설계되었습니다. 각 계층의 명확한 책임 분리와 의존성 역전 원칙을 통해 유연하고 테스트 가능한 코드를 제공합니다. 