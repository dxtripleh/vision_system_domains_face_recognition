# Services - 얼굴인식 서비스 모듈

## 📋 개요

이 폴더는 얼굴인식 기능의 비즈니스 로직을 담당하는 서비스 클래스들을 포함합니다. 모델과 유틸리티를 조합하여 실제 얼굴인식 서비스를 제공합니다.

## 🏗️ 폴더 구조

```
services/
├── __init__.py                    # 서비스 패키지 초기화
├── README.md                      # 이 파일
├── service.py                     # 기본 얼굴인식 서비스
└── face_recognition_service.py    # 통합 얼굴인식 서비스
```

## 🔍 포함된 서비스들

### 1. FaceRecognitionService (기본 서비스)
- **파일**: `service.py`
- **목적**: 기본적인 얼굴인식 기능 제공
- **기능**: 얼굴 검출, 특징 추출, 신원 매칭
- **사용처**: 단순한 얼굴인식 요구사항

### 2. IntegratedFaceRecognitionService (통합 서비스)
- **파일**: `face_recognition_service.py`
- **목적**: 고급 얼굴인식 기능 제공
- **기능**: 실시간 처리, 성능 모니터링, 데이터베이스 연동
- **사용처**: 프로덕션 환경, 실시간 얼굴인식

## 🚀 사용법

### 기본 서비스 사용법
```python
from domains.humanoid.face_recognition.services import FaceRecognitionService

# 서비스 초기화
service = FaceRecognitionService()

# 이미지에서 얼굴 인식
result = service.recognize_faces(image)

# 결과 확인
for face in result['faces']:
    print(f"신원: {face['identity']}")
    print(f"신뢰도: {face['confidence']:.2f}")
```

### 통합 서비스 사용법
```python
from domains.humanoid.face_recognition.services import IntegratedFaceRecognitionService

# 서비스 초기화
service = IntegratedFaceRecognitionService(
    config={
        'database_path': 'data/face_database.json',
        'performance_monitoring': True,
        'real_time_mode': True
    }
)

# 실시간 얼굴인식
service.start_realtime_recognition(camera_id=0)

# 성능 통계 확인
stats = service.get_performance_stats()
print(f"평균 FPS: {stats['avg_fps']:.1f}")
```

## 🔧 서비스 설정

### 기본 서비스 설정
```python
basic_config = {
    'detection_confidence': 0.5,    # 검출 신뢰도 임계값
    'recognition_confidence': 0.6,  # 인식 신뢰도 임계값
    'max_faces': 10,               # 최대 처리 얼굴 수
    'face_database_path': None     # 얼굴 데이터베이스 경로
}
```

### 통합 서비스 설정
```python
integrated_config = {
    'detection': {
        'confidence_threshold': 0.5,
        'min_face_size': 80,
        'max_faces': 10
    },
    'recognition': {
        'confidence_threshold': 0.6,
        'distance_threshold': 0.6,
        'embedding_dim': 512
    },
    'performance': {
        'target_fps': 30,
        'enable_monitoring': True,
        'log_level': 'INFO'
    },
    'database': {
        'path': 'data/face_database.json',
        'auto_save': True,
        'backup_interval': 3600
    }
}
```

## 📊 서비스 기능

### 기본 서비스 기능
- **얼굴 검출**: 이미지에서 얼굴 영역 검출
- **특징 추출**: 얼굴 이미지에서 특징 벡터 추출
- **신원 매칭**: 데이터베이스와 비교하여 신원 식별
- **결과 반환**: 검출 및 인식 결과 반환

### 통합 서비스 기능
- **실시간 처리**: 웹캠 스트림 실시간 처리
- **성능 모니터링**: FPS, 정확도, 지연시간 모니터링
- **데이터베이스 관리**: 얼굴 데이터베이스 CRUD 작업
- **이벤트 처리**: 얼굴 검출/인식 이벤트 처리
- **로깅**: 구조화된 로깅 시스템

## 🔗 의존성

### 내부 의존성
- `../models/`: 얼굴인식 모델들
- `../utils/`: 유틸리티 함수들
- `common/`: 공통 유틸리티
- `shared/vision_core/`: 비전 알고리즘 공통 기능

### 외부 의존성
```python
# requirements.txt
opencv-python>=4.5.0
numpy>=1.21.0
onnxruntime>=1.12.0
sqlite3  # Python 내장
json     # Python 내장
```

## 🧪 테스트

### 서비스 테스트 실행
```bash
# 전체 서비스 테스트
python -m pytest tests/test_services.py -v

# 특정 서비스 테스트
python -m pytest tests/test_services.py::TestFaceRecognitionService -v
python -m pytest tests/test_services.py::TestIntegratedFaceRecognitionService -v
```

### 테스트 예시
```python
def test_face_recognition_service():
    """얼굴인식 서비스 테스트"""
    service = FaceRecognitionService()
    
    # 테스트 이미지 로드
    test_image = load_test_image()
    
    # 얼굴 인식 수행
    result = service.recognize_faces(test_image)
    
    # 결과 검증
    assert 'faces' in result
    assert isinstance(result['faces'], list)
    assert all('identity' in face for face in result['faces'])
```

## 📝 데이터베이스 관리

### 얼굴 데이터베이스 구조
```json
{
    "faces": [
        {
            "id": "person_001",
            "name": "홍길동",
            "embedding": [0.1, 0.2, 0.3, ...],
            "metadata": {
                "age": 30,
                "gender": "male",
                "created_at": "2024-01-01T00:00:00Z"
            }
        }
    ],
    "metadata": {
        "version": "1.0.0",
        "created_at": "2024-01-01T00:00:00Z",
        "total_faces": 1
    }
}
```

### 데이터베이스 작업
```python
# 얼굴 추가
service.add_face(image, "홍길동", metadata={"age": 30})

# 얼굴 삭제
service.remove_face("person_001")

# 얼굴 업데이트
service.update_face("person_001", new_image, new_metadata)

# 얼굴 검색
faces = service.search_faces(query_embedding, top_k=5)
```

## 🔧 개발 가이드

### 새로운 서비스 추가
1. **서비스 클래스 생성**: `new_service.py` 파일 생성
2. **기본 인터페이스 구현**: `process()`, `initialize()` 메서드 구현
3. **설정 관리**: 서비스별 설정 클래스 구현
4. **테스트 작성**: 단위 테스트 및 통합 테스트 작성
5. **문서화**: 클래스 및 메서드 문서화

### 서비스 확장
```python
class CustomFaceRecognitionService(FaceRecognitionService):
    """커스텀 얼굴인식 서비스"""
    
    def __init__(self, custom_config):
        super().__init__()
        self.custom_config = custom_config
    
    def process_with_custom_logic(self, image):
        """커스텀 로직으로 처리"""
        # 커스텀 처리 로직
        result = super().recognize_faces(image)
        # 추가 처리
        return self.apply_custom_postprocessing(result)
```

## 🐛 문제 해결

### 일반적인 문제들

#### 1. 얼굴이 검출되지 않음
```python
# 해결 방법
service = FaceRecognitionService(
    config={'detection_confidence': 0.3}  # 신뢰도 임계값 낮추기
)
```

#### 2. 인식 정확도가 낮음
```python
# 해결 방법
service = FaceRecognitionService(
    config={'recognition_confidence': 0.4}  # 인식 임계값 낮추기
)
```

#### 3. 처리 속도가 느림
```python
# 해결 방법
service = IntegratedFaceRecognitionService(
    config={
        'performance': {
            'target_fps': 15,  # 목표 FPS 낮추기
            'enable_gpu': True  # GPU 사용 활성화
        }
    }
)
```

## 📈 성능 모니터링

### 성능 지표
- **처리 속도**: FPS (Frames Per Second)
- **정확도**: 검출/인식 정확도
- **지연시간**: 입력부터 출력까지의 시간
- **메모리 사용량**: RAM/GPU 메모리 사용량

### 성능 측정
```python
# 성능 측정
start_time = time.time()
result = service.recognize_faces(image)
processing_time = time.time() - start_time

print(f"처리 시간: {processing_time*1000:.2f}ms")
print(f"FPS: {1.0/processing_time:.1f}")
```

## 🔒 보안 및 개인정보 보호

### 데이터 보호
- **암호화**: 얼굴 데이터 암호화 저장
- **익명화**: 기본적으로 얼굴 데이터 익명화
- **접근 제어**: 데이터베이스 접근 권한 관리
- **보존 정책**: 30일 자동 삭제 정책

### GDPR 준수
- **동의 관리**: 사용자 동의 기반 데이터 처리
- **데이터 이식성**: 사용자 데이터 내보내기 기능
- **삭제 권리**: 사용자 데이터 삭제 요청 처리

## 🔗 관련 문서

- [얼굴인식 기능 문서](../README.md)
- [모델 문서](../models/README.md)
- [유틸리티 문서](../utils/README.md)
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