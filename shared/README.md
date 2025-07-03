# 📁 SHARED 폴더 - 공유 모듈

## 🎯 **목적**
모든 도메인에서 공통으로 사용하는 핵심 모듈들을 제공합니다.
비전 시스템의 공통 기능과 보안 기능을 중앙 집중식으로 관리합니다.

## 📂 **구조**
```
shared/
├── __init__.py          # 패키지 초기화
├── vision_core/         # 비전 알고리즘 공통 모듈
│   ├── detection/       # 검출 관련 공통 기능
│   ├── recognition/     # 인식 관련 공통 기능
│   ├── quality/         # 품질 평가 공통 기능
│   └── utils/           # 비전 유틸리티
└── security/            # 보안 모듈 (GDPR 준수)
    ├── data_protection/ # 데이터 보호
    ├── access_control/  # 접근 제어
    └── compliance/      # 규정 준수
```

## 🚀 **주요 기능**

### 1. **비전 코어 (vision_core/)**
모든 도메인에서 사용하는 비전 알고리즘 공통 기능

```python
from shared.vision_core.detection.base_detector import BaseDetector
from shared.vision_core.utils.fps_counter import FPSCounter

# 공통 인터페이스 사용
class MyDetector(BaseDetector):
    def detect(self, image):
        # 검출 로직 구현
        pass
    
# 공통 유틸리티 사용
fps_counter = FPSCounter()
current_fps = fps_counter.tick()
```

### 2. **보안 모듈 (security/)**
개인정보 보호 및 GDPR 준수 기능

```python
from shared.security.data_protection import anonymize_face_data
from shared.security.compliance import check_gdpr_compliance

# 얼굴 데이터 익명화
anonymized = anonymize_face_data(face_data)

# GDPR 준수 확인
is_compliant = check_gdpr_compliance(processing_purpose)
```

## 🔄 **사용 패턴**

### 도메인에서 shared 모듈 사용
```python
# 얼굴인식 도메인에서 사용
from shared.vision_core.detection.base_detector import BaseDetector
from shared.vision_core.quality.face_quality_assessor import FaceQualityAssessor
from shared.security.data_protection import encrypt_face_data

# 공장 불량 검출 도메인에서 사용 (향후)
from shared.vision_core.detection.base_detector import BaseDetector
from shared.vision_core.utils.performance_optimizer import PerformanceOptimizer
```

## 📝 **개발 가이드라인**

### ✅ **허용되는 것들**
- 2개 이상 도메인에서 사용하는 공통 기능
- 비전 시스템 표준 인터페이스
- 보안 및 개인정보 보호 기능
- 성능 최적화 공통 도구

### ❌ **금지되는 것들**
- 특정 도메인에만 필요한 기능 (→ 해당 도메인으로 이동)
- 비즈니스 로직 (→ domains/로 이동)
- 설정 파일 (→ config/로 이동)
- 실행 파일 (→ 도메인 내부로 이동)

## 🔗 **관련 문서**
- [프로젝트 개요](../README.md)
- [구조 문서](STRUCTURE.md)
- [비전 코어 가이드](vision_core/README.md)
- [보안 모듈 가이드](security/README.md)

## 💡 **초보자 팁**

### 1. **공통 기능 우선 확인**
새로운 기능이 필요할 때 먼저 shared에 있는지 확인하세요
```python
# 예시: FPS 카운터가 필요할 때
from shared.vision_core.utils.fps_counter import FPSCounter
```

### 2. **표준 인터페이스 사용**
새로운 검출기나 인식기를 만들 때는 shared의 기본 인터페이스를 상속하세요
```python
from shared.vision_core.detection.base_detector import BaseDetector

class MyDetector(BaseDetector):
    # 표준 인터페이스 구현
    pass
```

### 3. **보안 기능 활용**
개인정보를 다룰 때는 반드시 shared/security 모듈을 사용하세요
```python
from shared.security.data_protection import anonymize_face_data
```

### 4. **공통 모듈 기여**
새로운 공통 기능을 개발했다면 shared로 이동하여 다른 도메인에서도 사용할 수 있게 하세요

## ⚠️ **주의사항**
1. **의존성 관리**: shared 모듈은 다른 도메인에 의존하면 안 됩니다
2. **하위 호환성**: 기존 인터페이스 변경 시 모든 도메인에 영향을 줄 수 있습니다
3. **테스트 필수**: 공통 모듈 변경 시 모든 도메인에서 테스트 필요
4. **문서화**: 새로운 공통 기능 추가 시 반드시 문서화

---
*이 문서는 프로젝트 구조 검증 시스템에 의해 자동 생성되고 수동으로 개선되었습니다.* 