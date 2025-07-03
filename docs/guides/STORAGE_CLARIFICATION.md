# 📁 Storage 폴더 구분 설명

## 🤔 **사용자 질문**
`domains/face_recognition/data/storage`와 `domains/face_recognition/infrastructure/storage/` 이 두 폴더는 다른 것인가?

## ✅ **답변: 완전히 다른 목적의 폴더입니다**

### 1️⃣ **`domains/face_recognition/data/storage/`** (실제 데이터)
```
domains/face_recognition/data/storage/
├── faces/           # 실제 얼굴 데이터 JSON 파일들
│   ├── face_001.json
│   ├── face_002.json
│   └── face_index.json
└── persons/         # 실제 인물 데이터 JSON 파일들
    ├── person_001.json
    ├── person_002.json
    └── person_index.json
```

**목적**: 
- 실제 얼굴 인식 데이터를 저장하는 **데이터 저장소**
- JSON 파일 형태로 저장된 얼굴 임베딩, 인물 정보 등
- 런타임에 읽고 쓰는 실제 데이터

### 2️⃣ **`domains/face_recognition/infrastructure/storage/`** (코드)
```
domains/face_recognition/infrastructure/storage/
├── __init__.py
├── file_storage.py      # 파일 저장소 구현 클래스
└── database_storage.py  # 데이터베이스 저장소 구현 클래스
```

**목적**:
- 데이터를 어떻게 저장할지 정의하는 **코드 모듈**
- Repository 패턴의 구현체들
- 저장소 인터페이스를 구현하는 클래스들

## 🏗️ **DDD 아키텍처 관점에서의 구분**

### Infrastructure Layer (코드)
```python
# domains/face_recognition/infrastructure/storage/file_storage.py
class FileStorage:
    """파일 시스템에 데이터를 저장하는 방법을 정의"""
    def save(self, data):
        # 저장 로직
        pass
    
    def load(self, id):
        # 로드 로직  
        pass
```

### Data Layer (실제 데이터)
```
# domains/face_recognition/data/storage/faces/
실제로 저장된 얼굴 데이터 파일들
```

## 📊 **비교표**

| 구분 | Infrastructure/Storage | Data/Storage |
|------|----------------------|--------------|
| **타입** | Python 코드 파일 | JSON 데이터 파일 |
| **목적** | 저장 방법 정의 | 실제 데이터 저장 |
| **내용** | 클래스, 함수 | 얼굴 데이터, 인물 정보 |
| **역할** | "어떻게 저장할까?" | "무엇을 저장할까?" |
| **변경빈도** | 개발 시에만 | 런타임에 계속 |

## ✅ **결론**

- **Infrastructure/Storage**: 저장소 구현 코드 (개발자가 작성)
- **Data/Storage**: 실제 저장된 데이터 (시스템이 생성)

**완전히 다른 목적의 폴더이므로 둘 다 필요합니다!** 