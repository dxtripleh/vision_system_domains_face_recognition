# Recognition 모듈

이 모듈은 비전 시스템의 인식 기능을 제공하는 공통 모듈입니다.

## 📁 폴더 구조

```
recognition/
├── __init__.py              # 모듈 초기화
├── README.md                # 이 파일
├── base_recognizer.py       # 기본 인식기 클래스
├── face_embedder.py         # 얼굴 임베딩 생성기
├── face_matcher.py          # 얼굴 매칭기
└── similarity_matcher.py    # 유사도 매칭기
```

## 🎯 주요 기능

### 1. BaseRecognizer (기본 인식기)
```python
from shared.vision_core.recognition.base_recognizer import BaseRecognizer

class CustomRecognizer(BaseRecognizer):
    """사용자 정의 인식기"""
    
    def __init__(self, model_path: str, config: Dict):
        super().__init__(model_path, config)
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """특징 추출"""
        # 특징 추출 로직 구현
        pass
    
    def match(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """특징 매칭"""
        # 매칭 로직 구현
        pass
```

### 2. FaceEmbedder (얼굴 임베딩 생성기)
```python
from shared.vision_core.recognition.face_embedder import FaceEmbedder

# 얼굴 임베딩 생성기 초기화
embedder = FaceEmbedder(
    model_path="models/weights/face_recognition.onnx",
    config={
        "embedding_size": 512,
        "normalize_embeddings": True,
        "device": "cpu"
    }
)

# 얼굴 이미지에서 임베딩 추출
face_image = cv2.imread("face.jpg")
embedding = embedder.extract_embedding(face_image)

# 임베딩 정보
print(f"임베딩 크기: {embedding.shape}")
print(f"임베딩 범위: {embedding.min():.3f} ~ {embedding.max():.3f}")
```

### 3. FaceMatcher (얼굴 매칭기)
```python
from shared.vision_core.recognition.face_matcher import FaceMatcher

# 얼굴 매칭기 초기화
matcher = FaceMatcher(
    config={
        "similarity_threshold": 0.6,
        "distance_metric": "cosine",  # cosine, euclidean, manhattan
        "enable_face_quality": True
    }
)

# 두 얼굴 이미지 매칭
face1 = cv2.imread("face1.jpg")
face2 = cv2.imread("face2.jpg")

# 임베딩 추출
embedding1 = embedder.extract_embedding(face1)
embedding2 = embedder.extract_embedding(face2)

# 매칭 수행
similarity_score = matcher.match(embedding1, embedding2)
is_same_person = matcher.is_same_person(embedding1, embedding2)

print(f"유사도 점수: {similarity_score:.3f}")
print(f"동일 인물: {is_same_person}")
```

### 4. SimilarityMatcher (유사도 매칭기)
```python
from shared.vision_core.recognition.similarity_matcher import SimilarityMatcher

# 유사도 매칭기 초기화
similarity_matcher = SimilarityMatcher(
    config={
        "distance_metric": "cosine",
        "k_neighbors": 5,
        "enable_approximate_search": True
    }
)

# 데이터베이스에 임베딩 등록
database = {
    "person1": embedding1,
    "person2": embedding2,
    "person3": embedding3
}

# 새로운 얼굴 인식
query_embedding = embedder.extract_embedding(query_face)
matches = similarity_matcher.find_matches(query_embedding, database)

# 매칭 결과
for person_id, similarity in matches:
    print(f"{person_id}: {similarity:.3f}")
```

## 🔧 사용 예시

### 기본 얼굴 인식 파이프라인
```python
import cv2
import numpy as np
from shared.vision_core.recognition import FaceEmbedder, FaceMatcher

def face_recognition_pipeline(face_image: np.ndarray, database: Dict[str, np.ndarray]):
    """얼굴 인식 파이프라인"""
    
    # 얼굴 임베딩 생성기 초기화
    embedder = FaceEmbedder(
        model_path="models/weights/face_recognition.onnx",
        config={"embedding_size": 512}
    )
    
    # 얼굴 매칭기 초기화
    matcher = FaceMatcher(
        config={"similarity_threshold": 0.6}
    )
    
    # 쿼리 얼굴의 임베딩 추출
    query_embedding = embedder.extract_embedding(face_image)
    
    # 데이터베이스와 매칭
    best_match = None
    best_score = 0.0
    
    for person_id, stored_embedding in database.items():
        similarity = matcher.match(query_embedding, stored_embedding)
        if similarity > best_score:
            best_score = similarity
            best_match = person_id
    
    # 결과 반환
    if best_score >= matcher.config["similarity_threshold"]:
        return best_match, best_score
    else:
        return "Unknown", best_score

# 사용 예시
if __name__ == "__main__":
    # 데이터베이스 준비 (실제로는 파일에서 로드)
    database = {
        "person1": np.random.rand(512),  # 실제로는 저장된 임베딩
        "person2": np.random.rand(512),
        "person3": np.random.rand(512)
    }
    
    # 쿼리 얼굴 이미지
    query_face = cv2.imread("query_face.jpg")
    
    # 인식 수행
    person_id, confidence = face_recognition_pipeline(query_face, database)
    
    print(f"인식 결과: {person_id}")
    print(f"신뢰도: {confidence:.3f}")
```

### 실시간 얼굴 인식 시스템
```python
import cv2
import time
from shared.vision_core.detection import FaceDetector
from shared.vision_core.recognition import FaceEmbedder, FaceMatcher

def real_time_face_recognition(camera_id: int = 0):
    """실시간 얼굴 인식"""
    
    # 카메라 초기화
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {camera_id}")
    
    # 얼굴 검출기 초기화
    detector = FaceDetector(
        model_path="models/weights/face_detection.onnx",
        config={"confidence_threshold": 0.5}
    )
    
    # 얼굴 임베딩 생성기 초기화
    embedder = FaceEmbedder(
        model_path="models/weights/face_recognition.onnx",
        config={"embedding_size": 512}
    )
    
    # 얼굴 매칭기 초기화
    matcher = FaceMatcher(
        config={"similarity_threshold": 0.6}
    )
    
    # 등록된 얼굴 데이터베이스 (실제로는 파일에서 로드)
    registered_faces = {
        "Alice": np.random.rand(512),
        "Bob": np.random.rand(512),
        "Charlie": np.random.rand(512)
    }
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break
            
            # 얼굴 검출
            faces = detector.detect(frame)
            
            # 각 얼굴에 대해 인식 수행
            for face in faces:
                x, y, w, h = face.bbox
                
                # 얼굴 영역 추출
                face_roi = frame[y:y+h, x:x+w]
                
                # 얼굴 임베딩 추출
                try:
                    embedding = embedder.extract_embedding(face_roi)
                    
                    # 데이터베이스와 매칭
                    best_match = "Unknown"
                    best_score = 0.0
                    
                    for person_id, stored_embedding in registered_faces.items():
                        similarity = matcher.match(embedding, stored_embedding)
                        if similarity > best_score:
                            best_score = similarity
                            best_match = person_id
                    
                    # 결과 시각화
                    color = (0, 255, 0) if best_score >= matcher.config["similarity_threshold"] else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{best_match}: {best_score:.2f}", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                except Exception as e:
                    # 얼굴 인식 실패 시 검출만 표시
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, "Detection Only", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 화면 표시
            cv2.imshow("Real-time Face Recognition", frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

# 사용 예시
if __name__ == "__main__":
    real_time_face_recognition()
```

### 얼굴 등록 시스템
```python
import cv2
import numpy as np
import json
from pathlib import Path
from shared.vision_core.recognition import FaceEmbedder

class FaceRegistrationSystem:
    """얼굴 등록 시스템"""
    
    def __init__(self, database_path: str = "data/domains/humanoid/face_database.json"):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.embedder = FaceEmbedder(
            model_path="models/weights/face_recognition.onnx",
            config={"embedding_size": 512}
        )
        
        self.database = self.load_database()
    
    def load_database(self) -> Dict[str, np.ndarray]:
        """데이터베이스 로드"""
        if self.database_path.exists():
            with open(self.database_path, 'r') as f:
                data = json.load(f)
                return {k: np.array(v) for k, v in data.items()}
        return {}
    
    def save_database(self):
        """데이터베이스 저장"""
        data = {k: v.tolist() for k, v in self.database.items()}
        with open(self.database_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_face(self, person_id: str, face_image: np.ndarray) -> bool:
        """얼굴 등록"""
        try:
            # 얼굴 임베딩 추출
            embedding = self.embedder.extract_embedding(face_image)
            
            # 데이터베이스에 저장
            self.database[person_id] = embedding
            
            # 파일에 저장
            self.save_database()
            
            print(f"얼굴 등록 완료: {person_id}")
            return True
            
        except Exception as e:
            print(f"얼굴 등록 실패: {str(e)}")
            return False
    
    def list_registered_faces(self) -> List[str]:
        """등록된 얼굴 목록"""
        return list(self.database.keys())
    
    def remove_face(self, person_id: str) -> bool:
        """얼굴 삭제"""
        if person_id in self.database:
            del self.database[person_id]
            self.save_database()
            print(f"얼굴 삭제 완료: {person_id}")
            return True
        else:
            print(f"등록되지 않은 얼굴: {person_id}")
            return False

# 사용 예시
if __name__ == "__main__":
    registration_system = FaceRegistrationSystem()
    
    # 얼굴 등록
    face_image = cv2.imread("new_face.jpg")
    registration_system.register_face("NewPerson", face_image)
    
    # 등록된 얼굴 목록
    registered_faces = registration_system.list_registered_faces()
    print(f"등록된 얼굴: {registered_faces}")
```

## 📊 성능 최적화

### 1. 배치 처리
```python
def batch_embedding_extraction(embedder, images: List[np.ndarray]) -> List[np.ndarray]:
    """배치 임베딩 추출"""
    return embedder.extract_embeddings_batch(images)
```

### 2. GPU 가속
```python
# GPU 사용 설정
embedder = FaceEmbedder(
    model_path="models/weights/face_recognition.onnx",
    config={
        "device": "cuda",  # GPU 사용
        "precision": "fp16"  # 반정밀도 사용
    }
)
```

### 3. 벡터화된 유사도 계산
```python
def vectorized_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """벡터화된 유사도 계산"""
    # 코사인 유사도 계산
    similarity = np.dot(embeddings1, embeddings2.T) / (
        np.linalg.norm(embeddings1, axis=1, keepdims=True) * 
        np.linalg.norm(embeddings2, axis=1, keepdims=True).T
    )
    return similarity
```

## 🔧 설정 옵션

### 얼굴 임베딩 설정
```python
FACE_EMBEDDING_CONFIG = {
    "embedding_size": 512,           # 임베딩 차원
    "normalize_embeddings": True,    # 임베딩 정규화
    "device": "cpu",                 # 실행 디바이스
    "precision": "fp32",             # 정밀도
    "batch_size": 1,                 # 배치 크기
    "enable_face_quality": True,     # 얼굴 품질 검사
    "min_face_size": 80,             # 최소 얼굴 크기
    "preprocessing": {
        "resize": (112, 112),        # 리사이즈 크기
        "normalize": True,           # 정규화
        "mean": [0.5, 0.5, 0.5],     # 평균값
        "std": [0.5, 0.5, 0.5]       # 표준편차
    }
}
```

### 얼굴 매칭 설정
```python
FACE_MATCHING_CONFIG = {
    "similarity_threshold": 0.6,     # 유사도 임계값
    "distance_metric": "cosine",     # 거리 측정 방식
    "enable_face_quality": True,     # 얼굴 품질 검사
    "quality_threshold": 0.5,        # 품질 임계값
    "enable_liveness": False,        # 생체 검증 비활성화
    "matching_strategy": "nearest",  # 매칭 전략
    "k_neighbors": 1,                # k-최근접 이웃
    "enable_approximate_search": False  # 근사 검색 비활성화
}
```

### 유사도 매칭 설정
```python
SIMILARITY_MATCHING_CONFIG = {
    "distance_metric": "cosine",     # 거리 측정 방식
    "k_neighbors": 5,                # k-최근접 이웃
    "enable_approximate_search": True,  # 근사 검색 활성화
    "approximation_config": {
        "method": "lsh",             # LSH (Locality Sensitive Hashing)
        "num_tables": 10,            # 해시 테이블 수
        "num_bits": 64               # 해시 비트 수
    },
    "enable_caching": True,          # 캐싱 활성화
    "cache_size": 1000               # 캐시 크기
}
```

## 🚨 주의사항

### 1. 모델 호환성
- ONNX 모델만 지원 (PyTorch 모델은 변환 필요)
- 임베딩 차원 일치 확인 필수
- 전처리 방식 일치 확인 필요

### 2. 성능 고려사항
- 대용량 데이터베이스 시 근사 검색 사용 권장
- GPU 메모리 부족 시 배치 크기 조정
- 실시간 처리 시 임베딩 캐싱 고려

### 3. 정확도 vs 속도 트레이드오프
- 유사도 임계값 조정으로 정확도/속도 균형
- 임베딩 차원 조정으로 성능 최적화
- 얼굴 품질 검사로 정확도 향상

### 4. 보안 고려사항
- 임베딩 데이터 암호화 저장
- 접근 권한 관리
- 개인정보 보호 규정 준수

## 📞 지원

### 문제 해결
1. **인식 성능 저하**: 모델 경로 및 설정 확인
2. **메모리 부족**: 배치 크기 및 임베딩 차원 조정
3. **GPU 오류**: CUDA 버전 및 드라이버 확인
4. **매칭 실패**: 유사도 임계값 조정

### 추가 도움말
- 각 인식기의 `__init__` 메서드 문서 참조
- 모델 파일의 메타데이터 확인
- 성능 벤치마크 결과 참조 