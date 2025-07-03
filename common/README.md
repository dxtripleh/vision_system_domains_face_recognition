# 📁 COMMON 폴더 - 범용 유틸리티

## 🎯 **목적**
프로젝트 전체에서 공통으로 사용되는 범용 유틸리티 모듈들을 제공합니다.
모든 도메인과 모듈에서 자유롭게 사용할 수 있는 기반 기능들이 포함되어 있습니다.

## 📂 **구조**
```
common/
├── __init__.py          # 패키지 초기화 파일
├── config_loader.py     # 설정 파일 로딩 유틸리티
└── logging.py          # 로깅 시스템 설정
```

## 🚀 **주요 기능**

### 1. **설정 관리 (config_loader.py)**
- YAML/JSON 설정 파일 자동 로딩
- 환경별 설정 병합 (개발/운영)
- 설정 검증 및 기본값 처리

```python
from common.config_loader import load_config

# 설정 파일 로딩
config = load_config('face_recognition_api.yaml')
print(config['model']['detection']['confidence_threshold'])
```

### 2. **로깅 시스템 (logging.py)**
- 구조화된 로깅 설정
- 파일/콘솔 출력 관리
- 로그 레벨별 필터링

```python
from common.logging import setup_logging

# 로깅 설정
logger = setup_logging()
logger.info("시스템 시작")
logger.error("오류 발생")
```

## 🔄 **사용 패턴**

### 일반적인 import 방식
```python
# 도메인에서 사용
from common.config_loader import load_config
from common.logging import setup_logging

# 설정 로딩 및 로깅 초기화
config = load_config('my_config.yaml')
logger = setup_logging()
```

## 📝 **개발 가이드라인**

### ✅ **허용되는 것들**
- 모든 도메인에서 자유롭게 사용 가능
- 프로젝트 전체에서 공통으로 필요한 기능
- 외부 라이브러리에 대한 의존성 최소화

### ❌ **금지되는 것들**
- 특정 도메인에만 필요한 기능
- 비즈니스 로직 포함
- 다른 도메인 모듈에 대한 의존성

## 🔗 **관련 문서**
- [프로젝트 개요](../README.md)
- [설정 관리 가이드](../docs/guides/CONFIG_GUIDE.md)
- [로깅 시스템 가이드](../docs/guides/LOGGING_GUIDE.md)

## 💡 **초보자 팁**
1. **common 모듈 우선 활용**: 새로운 유틸리티가 필요할 때 먼저 common에 있는지 확인하세요
2. **import 경로**: `from common.모듈명 import 함수명` 형태로 사용하세요
3. **설정 파일**: YAML 형식을 권장하며, config/ 폴더에 저장하세요
4. **로깅**: 개발 시 적절한 로그 레벨(DEBUG, INFO, WARNING, ERROR)을 사용하세요

---
*이 문서는 프로젝트 구조 검증 시스템에 의해 자동 생성되고 수동으로 개선되었습니다.*
