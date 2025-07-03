# Face Recognition Core Module 🧠

얼굴인식 도메인의 핵심 비즈니스 로직을 담당하는 모듈입니다. 도메인 주도 설계(DDD) 원칙에 따라 구성되어 있습니다.

## 📋 목차

- [개요](#개요)
- [폴더 구조](#폴더-구조)
- [핵심 구성 요소](#핵심-구성-요소)
- [사용 방법](#사용-방법)
- [개발 가이드](#개발-가이드)

## 🎯 개요

### 무엇인가요?
Core 모듈은 얼굴인식 도메인의 **핵심 비즈니스 로직**을 포함합니다. 기술적인 구현 세부사항과 분리되어 있어, 비즈니스 규칙에만 집중할 수 있습니다.

### 왜 중요한가요?
- **기술 독립적**: 특정 프레임워크나 라이브러리에 의존하지 않음
- **테스트 용이**: 비즈니스 로직만 분리되어 테스트하기 쉬움
- **유지보수성**: 비즈니스 규칙 변경 시 이 모듈만 수정
- **재사용성**: 다른 프로젝트에서도 재사용 가능

## 📁 폴더 구조

```
core/
├── README.md           # 👈 현재 파일
├── entities/           # 도메인 엔티티
│   ├── face.py        # 얼굴 엔티티
│   ├── person.py      # 인물 엔티티
│   └── face_detection_result.py  # 검출 결과 엔티티
├── services/           # 도메인 서비스
│   ├── face_detection_service.py
│   ├── face_recognition_service.py
│   └── person_management_service.py
├── repositories/       # 저장소 인터페이스
│   ├── person_repository.py
│   └── face_repository.py
└── value_objects/      # 값 객체
    ├── bounding_box.py
    ├── confidence.py
    └── face_embedding.py
```

## 🔧 핵심 구성 요소

### 1. Entities (엔티티) 📦
**역할**: 도메인의 핵심 개념을 나타내는 객체들

#### `Face` - 얼굴 엔티티
```python
from domains.face_recognition.core.entities import Face

# 얼굴 객체 생성
face = Face(
    face_id="face_001",
    person_id="person_123",  # None이면 미등록 얼굴
    embedding=embedding_vector,
    confidence=0.95,
    bbox=[100, 100, 200, 200]  # [x, y, width, height]
)

# 얼굴 정보 확인
print(f"신원 확인됨: {face.is_identified()}")
print(f"바운딩 박스: {face.get_bbox_coordinates()}")
```

#### `Person` - 인물 엔티티
```python
from domains.face_recognition.core.entities import Person

# 새 인물 생성
person = Person.create_new(
    name="홍길동",
    metadata={"department": "개발팀", "role": "시니어"}
)

# 얼굴 임베딩 추가
person.add_face_embedding(face_embedding)
print(f"등록된 얼굴 수: {person.get_embedding_count()}")
```

#### `FaceDetectionResult` - 검출 결과 엔티티
```python
from domains.face_recognition.core.entities import FaceDetectionResult

# 검출 결과 생성
result = FaceDetectionResult.create_empty(
    image_id="img_001",
    model_name="RetinaFace",
    model_version="1.0"
)

# 얼굴 추가
result.add_face(face)

# 결과 분석
print(f"총 얼굴 수: {result.get_face_count()}")
print(f"신원 확인된 얼굴: {len(result.get_identified_faces())}")
print(f"높은 신뢰도 얼굴: {len(result.get_high_confidence_faces(0.8))}")
```

### 2. Services (서비스) ⚙️
**역할**: 복잡한 비즈니스 로직을 처리하는 서비스들

#### 언제 서비스를 사용하나요?
- 여러 엔티티가 협력해야 하는 작업
- 복잡한 비즈니스 규칙이 있는 작업
- 외부 시스템과의 조정이 필요한 작업

```python
# 예시: 얼굴 인식 서비스
from domains.face_recognition.core.services import FaceRecognitionService

service = FaceRecognitionService(person_repository, face_repository)

# 얼굴 매칭 수행
matching_result = service.match_face(face_embedding)
if matching_result.is_match:
    print(f"매칭된 인물: {matching_result.person.name}")
    print(f"유사도: {matching_result.similarity:.2f}")
```

### 3. Repositories (저장소) 💾
**역할**: 데이터 저장/조회를 위한 인터페이스 정의

#### 왜 인터페이스만 정의하나요?
- **의존성 역전**: Core는 구체적인 저장 방식에 의존하지 않음
- **테스트 용이**: Mock 객체로 쉽게 테스트 가능
- **유연성**: 저장 방식 변경 시 Core 코드는 수정 불필요

```python
# 인터페이스 정의 (추상 클래스)
from abc import ABC, abstractmethod

class PersonRepository(ABC):
    @abstractmethod
    def save(self, person: Person) -> bool:
        """인물 정보 저장"""
        pass
    
    @abstractmethod
    def find_by_id(self, person_id: str) -> Optional[Person]:
        """ID로 인물 조회"""
        pass
    
    @abstractmethod
    def find_by_embedding(self, embedding: np.ndarray, threshold: float) -> List[Person]:
        """임베딩으로 유사한 인물들 조회"""
        pass
```

### 4. Value Objects (값 객체) 💎
**역할**: 값으로만 구별되는 불변 객체들

#### 값 객체의 특징
- **불변성**: 생성 후 변경 불가
- **값 동등성**: 같은 값이면 같은 객체로 취급
- **검증 로직**: 생성 시 유효성 검사

```python
# BoundingBox 값 객체 예시
from domains.face_recognition.core.value_objects import BoundingBox

# 바운딩 박스 생성 (자동 검증)
bbox = BoundingBox(x=100, y=100, width=200, height=200)

# 유용한 메서드들
print(f"면적: {bbox.area()}")
print(f"중심점: {bbox.center()}")
print(f"좌표: {bbox.to_coordinates()}")  # (x1, y1, x2, y2)

# 다른 박스와 교집합 계산
other_bbox = BoundingBox(x=150, y=150, width=200, height=200)
iou = bbox.intersection_over_union(other_bbox)
print(f"IoU: {iou:.2f}")
```

## 💡 사용 방법

### 기본 사용 패턴

#### 1. 새 인물 등록 프로세스
```python
from domains.face_recognition.core.entities import Person
from domains.face_recognition.core.services import PersonManagementService

# 1. 서비스 초기화
person_service = PersonManagementService(person_repository)

# 2. 새 인물 생성
person = Person.create_new("김철수")

# 3. 얼굴 임베딩 추가
for face_image in face_images:
    embedding = extract_embedding(face_image)  # infrastructure에서 구현
    person.add_face_embedding(embedding)

# 4. 저장
success = person_service.register_person(person)
if success:
    print(f"등록 완료: {person.person_id}")
```

#### 2. 얼굴 인식 프로세스
```python
from domains.face_recognition.core.services import FaceRecognitionService

# 1. 서비스 초기화
recognition_service = FaceRecognitionService(person_repository)

# 2. 얼굴 임베딩 추출
face_embedding = extract_embedding(face_image)

# 3. 인물 매칭
match_result = recognition_service.identify_person(face_embedding)

# 4. 결과 처리
if match_result.is_confident_match(threshold=0.8):
    person = match_result.person
    print(f"인식됨: {person.name} (신뢰도: {match_result.confidence:.2f})")
else:
    print("미등록 인물 또는 낮은 신뢰도")
```

### 고급 사용 패턴

#### 1. 배치 처리
```python
# 여러 얼굴을 한 번에 처리
faces = [face1, face2, face3]
results = recognition_service.identify_multiple_faces(faces)

for face, result in zip(faces, results):
    if result.person:
        print(f"얼굴 {face.face_id}: {result.person.name}")
    else:
        print(f"얼굴 {face.face_id}: 미등록")
```

#### 2. 조건부 검색
```python
# 특정 조건의 인물들 검색
filters = {
    'department': '개발팀',
    'min_embeddings': 3,  # 최소 3개 이상의 얼굴 등록
    'created_after': datetime(2025, 1, 1)
}

persons = person_service.find_persons_by_criteria(filters)
print(f"조건에 맞는 인물 수: {len(persons)}")
```

## 🛠️ 개발 가이드

### 새로운 엔티티 추가하기

1. **엔티티 클래스 생성**
```python
# entities/my_entity.py
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class MyEntity:
    entity_id: str
    name: str
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
            
        # 비즈니스 규칙 검증
        if not self.name or len(self.name.strip()) == 0:
            raise ValueError("Name cannot be empty")
    
    def business_method(self) -> bool:
        """비즈니스 로직 메서드"""
        # 구현
        return True
```

2. **엔티티 등록**
```python
# entities/__init__.py에 추가
from .my_entity import MyEntity

__all__ = ["Face", "Person", "FaceDetectionResult", "MyEntity"]
```

### 새로운 서비스 추가하기

1. **서비스 인터페이스 정의**
```python
# services/my_service.py
from abc import ABC, abstractmethod
from typing import List, Optional

class MyService(ABC):
    def __init__(self, repository: MyRepository):
        self._repository = repository
    
    def complex_business_operation(self, param1: str, param2: int) -> bool:
        """복잡한 비즈니스 로직"""
        # 1. 입력 검증
        if not param1 or param2 < 0:
            raise ValueError("Invalid parameters")
        
        # 2. 비즈니스 규칙 적용
        entity = self._repository.find_by_param(param1)
        if entity:
            entity.update_value(param2)
            return self._repository.save(entity)
        
        return False
```

### 새로운 값 객체 추가하기

1. **값 객체 생성**
```python
# value_objects/my_value_object.py
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)  # 불변성 보장
class MyValueObject:
    value1: str
    value2: int
    
    def __post_init__(self):
        # 생성 시 검증
        if self.value2 < 0:
            raise ValueError("value2 must be positive")
    
    def calculate_something(self) -> float:
        """값 기반 계산"""
        return self.value2 * 1.5
    
    def __eq__(self, other: Any) -> bool:
        """값 동등성 비교"""
        if not isinstance(other, MyValueObject):
            return False
        return self.value1 == other.value1 and self.value2 == other.value2
```

### 테스트 작성하기

```python
# tests/core/test_entities.py
import unittest
from domains.face_recognition.core.entities import Face, Person

class TestFaceEntity(unittest.TestCase):
    def test_face_creation(self):
        """얼굴 엔티티 생성 테스트"""
        face = Face(
            face_id="test_face",
            person_id=None,
            embedding=np.random.rand(512),
            confidence=0.9,
            bbox=[0, 0, 100, 100]
        )
        
        self.assertEqual(face.face_id, "test_face")
        self.assertFalse(face.is_identified())
        self.assertEqual(face.get_bbox_coordinates(), (0, 0, 100, 100))
    
    def test_invalid_confidence(self):
        """잘못된 신뢰도 값 테스트"""
        with self.assertRaises(ValueError):
            Face(
                face_id="test",
                person_id=None,
                embedding=np.random.rand(512),
                confidence=1.5,  # 잘못된 값
                bbox=[0, 0, 100, 100]
            )
```

## 📚 참고 자료

### 도메인 주도 설계 (DDD) 개념
- **Entity**: 고유 식별자를 가진 객체
- **Value Object**: 값으로만 구별되는 불변 객체
- **Service**: 엔티티나 값 객체에 속하지 않는 비즈니스 로직
- **Repository**: 데이터 저장/조회 추상화

### 설계 원칙
- **단일 책임 원칙**: 각 클래스는 하나의 책임만
- **의존성 역전 원칙**: 추상화에 의존, 구체화에 의존하지 않음
- **개방-폐쇄 원칙**: 확장에는 열려있고, 수정에는 닫혀있음

---

**버전**: 0.1.0  
**최종 업데이트**: 2025-06-28  
**작성자**: Vision System Team 