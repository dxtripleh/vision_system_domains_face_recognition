# Vision System Development Rules

## 📋 **규칙 체계 개요**

이 프로젝트는 **얼굴인식을 시작으로 확장 가능한 비전 시스템**을 구축하는 프로젝트입니다.  
현재 **Phase 1: 얼굴인식 도메인 AI 모델 통합 단계**에 있습니다.

---

## 🎯 **현재 개발 상황 (2025-06-28 기준)**

### ✅ **완료된 작업**
- [x] DDD 기반 도메인 구조 설계 완료
- [x] `domains/face_recognition/` 핵심 엔티티 정의
- [x] Infrastructure 및 Interfaces 계층 구조 생성
- [x] 기본 문서화 완료 (README, 구조 설명)

### 🔄 **현재 진행 중 (Week 3-4)**
- [ ] **실제 AI 모델 통합** (RetinaFace, ArcFace)
- [ ] **실시간 처리 파이프라인** 구현
- [ ] **하드웨어 연결 및 검증**

### 🔮 **다음 단계 (Week 5-8)**
- [ ] API 인터페이스 구현
- [ ] 두 번째 도메인 추가 준비
- [ ] Shared 모듈 확장 및 최적화

---

## 📂 **규칙 분류 체계 (Rule Type별)**

### 🔥 **HIGH PRIORITY (현재 필수 적용)**

#### **CORE_MANDATORY** - 절대 준수 규칙
| 파일 | Rule Type | 적용 범위 | 상태 |
|------|-----------|----------|------|
| `repo_specific_rule.mdc` | CORE_MANDATORY | 모든 개발 작업 | ✅ **즉시 적용** |

#### **DOMAIN_SPECIFIC** - 비전 시스템 특화
| 파일 | Rule Type | 적용 범위 | 상태 |
|------|-----------|----------|------|
| `02-vision-specific/vision-system-specific-rules.mdc` | DOMAIN_SPECIFIC | 비전 시스템 개발 | ✅ **즉시 적용** |

#### **IMPLEMENTATION_GUIDE** - 현재 진행 중인 구현
| 파일 | Rule Type | 적용 범위 | 상태 |
|------|-----------|----------|------|
| `04-roadmap/face-recognition-detailed-implementation.mdc` | IMPLEMENTATION_GUIDE | 얼굴인식 도메인 | 🔄 **현재 진행** |

#### **CODING_STANDARDS** - 코딩 표준
| 파일 | Rule Type | 적용 범위 | 상태 |
|------|-----------|----------|------|
| `01-universal/common-development-rules.mdc` | CODING_STANDARDS | 모든 코드 작성 | ✅ **즉시 적용** |

#### **ARCHITECTURE_RULES** - 프로젝트 구조
| 파일 | Rule Type | 적용 범위 | 상태 |
|------|-----------|----------|------|
| `01-universal/project-structure-rules.mdc` | ARCHITECTURE_RULES | 프로젝트 구조 | ✅ **즉시 적용** |

#### **SECURITY_COMPLIANCE** - 보안 및 규정 준수
| 파일 | Rule Type | 적용 범위 | 상태 |
|------|-----------|----------|------|
| `02-vision-specific/vision-system-security-rules.mdc` | SECURITY_COMPLIANCE | 보안 및 GDPR | ✅ **즉시 적용** |

### 🟡 **MEDIUM PRIORITY (단계별 적용)**

#### **ARCHITECTURE_STRATEGY** - 확장 전략
| 파일 | Rule Type | 적용 시점 | 상태 |
|------|-----------|----------|------|
| `01-universal/clean-slate-architecture-strategy.mdc` | ARCHITECTURE_STRATEGY | 도메인 확장 시 | 🔮 **확장 시 적용** |

#### **DOCUMENTATION_STANDARDS** - 문서화 시스템
| 파일 | Rule Type | 적용 시점 | 상태 |
|------|-----------|----------|------|
| `01-universal/documentation-system-rules.mdc` | DOCUMENTATION_STANDARDS | API 개발 시 | 🔮 **API 개발 시** |

#### **QUALITY_ASSURANCE** - 품질 보증
| 파일 | Rule Type | 적용 시점 | 상태 |
|------|-----------|----------|------|
| `01-universal/pre-development-checklist.mdc` | QUALITY_ASSURANCE | 새 기능 개발 시 | 🔮 **필요 시 적용** |
| `01-universal/common-folder-management-strategy.mdc` | QUALITY_ASSURANCE | 공통 모듈 관리 시 | 🔮 **Shared 확장 시** |

#### **MONITORING_SYSTEMS** - 모니터링
| 파일 | Rule Type | 적용 시점 | 상태 |
|------|-----------|----------|------|
| `02-vision-specific/vision-system-monitoring-rules.mdc` | MONITORING_SYSTEMS | 프로덕션 준비 시 | 🔮 **프로덕션 준비 시** |

### 🔵 **LOW PRIORITY (향후 적용)**

#### **DEPLOYMENT_AUTOMATION** - 배포 자동화
| 파일 | Rule Type | 적용 시점 | 상태 |
|------|-----------|----------|------|
| `01-universal/deployment-automation-rules.mdc` | DEPLOYMENT_AUTOMATION | 프로덕션 배포 시 | 💤 **향후 적용** |

#### **HARDWARE_OPTIMIZATION** - 하드웨어 최적화
| 파일 | Rule Type | 적용 시점 | 상태 |
|------|-----------|----------|------|
| `01-universal/hardware-requirements-rules.mdc` | HARDWARE_OPTIMIZATION | 성능 최적화 시 | 💤 **향후 적용** |

#### **DEVELOPMENT_TOOLS** - 개발 도구
| 파일 | Rule Type | 적용 시점 | 상태 |
|------|-----------|----------|------|
| `01-universal/automation-tools-rules.mdc` | DEVELOPMENT_TOOLS | 팀 확장 시 | 💤 **향후 적용** |

#### **LEGACY_ANALYSIS** - 레거시 분석
| 파일 | Rule Type | 적용 시점 | 상태 |
|------|-----------|----------|------|
| `01-universal/legacy-code-analysis-rules.mdc` | LEGACY_ANALYSIS | 리팩토링 시 | 💤 **향후 적용** |

#### **EXPANSION_ROADMAP** - 확장 로드맵
| 파일 | Rule Type | 적용 시점 | 상태 |
|------|-----------|----------|------|
| `04-roadmap/vision-system-complete-expansion-roadmap.mdc` | EXPANSION_ROADMAP | 3개 이상 도메인 시 | 💤 **대규모 확장 시** |

---

## 🎯 **현재 단계별 적용 가이드**

### **현재 Week 3-4: AI 모델 통합**
```python
CURRENT_ACTIVE_RULES = {
    'MANDATORY': [
        'repo_specific_rule.mdc',                           # 기본 규칙
        'vision-system-specific-rules.mdc',                 # 비전 특화
        'common-development-rules.mdc',                     # 코딩 표준
        'project-structure-rules.mdc',                      # 구조 규칙
        'vision-system-security-rules.mdc'                  # 보안 규칙
    ],
    'IMPLEMENTATION_FOCUS': [
        'face-recognition-detailed-implementation.mdc'      # 구현 가이드
    ]
}
```

### **다음 Week 5-6: API 및 인터페이스**
```python
NEXT_PHASE_RULES = {
    'ADD_TO_ACTIVE': [
        'documentation-system-rules.mdc',                   # API 문서화
        'vision-system-monitoring-rules.mdc',               # 기본 모니터링
    ],
    'CONTINUE_MANDATORY': [
        # 기존 HIGH PRIORITY 규칙들 계속 적용
    ]
}
```

### **향후 Week 7-8: 도메인 확장**
```python
EXPANSION_PHASE_RULES = {
    'ADD_TO_ACTIVE': [
        'clean-slate-architecture-strategy.mdc',            # 아키텍처 확장
        'common-folder-management-strategy.mdc',            # 공통 모듈 관리
        'pre-development-checklist.mdc'                     # 개발 전 체크
    ]
}
```

---

## 🔧 **규칙 적용 체크리스트**

### **매일 확인해야 할 규칙 (HIGH PRIORITY)**
- [ ] ✅ 루트 디렉토리에 임시 파일 생성 금지
- [ ] ✅ Type hints 및 Docstring 작성
- [ ] ✅ 도메인 독립성 유지
- [ ] ✅ 파일 저장 위치 규칙 준수
- [ ] ✅ 보안 기본 사항 확인

### **주간 확인해야 할 규칙 (MEDIUM PRIORITY)**
- [ ] 🟡 문서화 상태 점검
- [ ] 🟡 코드 품질 메트릭 확인
- [ ] 🟡 테스트 커버리지 확인

### **단계별 확인해야 할 규칙 (LOW PRIORITY)**
- [ ] 🔵 성능 최적화 필요성 검토
- [ ] 🔵 배포 준비 상태 점검
- [ ] 🔵 확장성 요구사항 검토

---

## 📊 **규칙 적용 현황 대시보드**

### **현재 적용 상태**
```
HIGH PRIORITY (6개 규칙)
├── ✅ CORE_MANDATORY (1/1) - 100%
├── ✅ DOMAIN_SPECIFIC (1/1) - 100%
├── 🔄 IMPLEMENTATION_GUIDE (1/1) - 진행 중
├── ✅ CODING_STANDARDS (1/1) - 100%
├── ✅ ARCHITECTURE_RULES (1/1) - 100%
└── ✅ SECURITY_COMPLIANCE (1/1) - 100%

MEDIUM PRIORITY (5개 규칙)
├── 🔮 ARCHITECTURE_STRATEGY (0/1) - 대기
├── 🔮 DOCUMENTATION_STANDARDS (0/1) - 대기
├── 🔮 QUALITY_ASSURANCE (0/2) - 대기
└── 🔮 MONITORING_SYSTEMS (0/1) - 대기

LOW PRIORITY (5개 규칙)
└── 💤 ALL (0/5) - 향후 적용
```

---

## 💡 **규칙 사용 가이드**

### **코드 작성 시**
1. `repo_specific_rule.mdc` 기본 규칙 확인
2. `vision-system-specific-rules.mdc` 특화 규칙 적용
3. `common-development-rules.mdc` 코딩 표준 준수

### **새 기능 개발 시**
1. `face-recognition-detailed-implementation.mdc` 구현 가이드 참조
2. `project-structure-rules.mdc` 구조 규칙 확인
3. `vision-system-security-rules.mdc` 보안 요구사항 검토

### **코드 리뷰 시**
1. HIGH PRIORITY 규칙 모두 준수 확인
2. 해당 단계의 MEDIUM PRIORITY 규칙 적용 확인
3. 문서화 및 테스트 완성도 검토

---

**현재 개발 단계에 맞는 HIGH PRIORITY 규칙들을 우선적으로 적용하여 안정적이고 확장 가능한 얼굴인식 시스템을 완성해 나가세요! 🎯**

## 🚀 시작하기

### 1️⃣ **개발 환경 준비**
```bash
# 1. 필수 소프트웨어 설치
python --version  # Python 3.9+ 필요
git --version     # Git 최신 버전
docker --version  # Docker (선택사항)

# 2. 가상환경 생성
python -m venv vision_env
source vision_env/bin/activate  # Linux/Mac
# vision_env\Scripts\activate   # Windows

# 3. 기본 패키지 설치
pip install --upgrade pip
pip install torch torchvision opencv-python numpy
```

### 2️⃣ **프로젝트 구조 생성**
```bash
# 기본 프로젝트 구조 생성
mkdir -p vision_system/{common,config,models,features,modules,applications,scripts,tests,data}
cd vision_system

# 필수 __init__.py 파일 생성
touch common/__init__.py features/__init__.py modules/__init__.py applications/__init__.py
```

### 3️⃣ **개발 전 체크리스트 실행**
`01-universal/pre-development-checklist.mdc`를 참조하여 7단계 체크리스트를 완료하세요:

1. **환경 및 인프라 준비** ✅
2. **기술 스택 및 의존성 검토** ✅
3. **데이터 전략 수립** ✅
4. **성능 및 품질 기준 설정** ✅
5. **프로젝트 관리 및 협업** ✅
6. **위험 관리 및 비상 계획** ✅
7. **최종 검토 및 승인** ✅

## 📖 주요 개발 가이드

### 🧑 **얼굴인식 도메인 개발 (현재 우선순위)**

#### **Phase 1.1: Face Detection (Month 1-2)**
```python
# 구현 순서
features/face_recognition_domain/
├── face_detection/
│   ├── detectors/          # MediaPipe, MTCNN, RetinaFace
│   ├── processors/         # 전처리, 후처리
│   ├── interfaces/         # 표준 인터페이스
│   └── optimization/       # 성능 최적화
```

#### **Phase 1.2: Face Recognition (Month 2-4)**
```python
# 구현 순서
features/face_recognition_domain/
├── face_recognition/
│   ├── embedders/          # FaceNet, ArcFace, InsightFace
│   ├── matchers/           # 유사도 매칭
│   ├── databases/          # 얼굴 DB 관리
│   ├── quality_control/    # 데이터 품질 검증
│   └── privacy_protection/ # GDPR 준수
```

### 🏭 **공장불량인식 도메인 개발 (Phase 2)**

#### **Phase 2.1: Defect Detection (Month 6-8)**
```python
# 구현 순서
features/factory_defect_domain/
├── defect_detection/
│   ├── anomaly_detectors/  # 이상 탐지
│   ├── surface_inspectors/ # 표면 검사
│   ├── dimension_checkers/ # 치수 검사
│   └── quality_assessors/  # 품질 평가
```

#### **Phase 2.2: Defect Classification (Month 8-10)**
```python
# 구현 순서
features/factory_defect_domain/
├── defect_classification/
│   ├── scratch_classifiers/    # 스크래치 분류
│   ├── dent_classifiers/       # 찌그러짐 분류
│   ├── crack_classifiers/      # 균열 분류
│   └── contamination_classifiers/ # 오염 분류
```

### ⚡ **전선불량인식 도메인 개발 (Phase 3)**

#### **Phase 3.1: Component Detection (Month 12-14)**
```python
# 구현 순서
features/powerline_defect_domain/
├── component_detection/
│   ├── wire_detectors/     # 전선 검출
│   ├── insulator_detectors/ # 절연체 검출
│   ├── tower_detectors/    # 철탑 검출
│   └── equipment_detectors/ # 설비 검출
```

## 🔧 개발 도구 및 환경

### 💻 **권장 개발 환경**
```yaml
IDE: Visual Studio Code
Extensions:
  - Python
  - Pylance
  - Black Formatter
  - GitLens
  - Docker
  - Jupyter

Code Quality:
  - black (코드 포맷팅)
  - isort (import 정렬)
  - flake8 (린팅)
  - mypy (타입 체킹)
  - pytest (테스팅)
```

### 🐳 **Docker 환경 (선택사항)**
```dockerfile
# Dockerfile 예시
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

### 📊 **모니터링 및 로깅**
```python
# 로깅 설정 예시
import logging
import structlog

# 구조화된 로깅
logger = structlog.get_logger()
logger.info("Face detection started", 
           model="mediapipe", 
           fps=30, 
           resolution="1080p")
```

## 🔒 보안 및 컴플라이언스

### 🛡️ **데이터 보호**
- **GDPR 준수**: 개인정보 보호 규정 완전 준수
- **데이터 암호화**: 저장 및 전송 데이터 암호화
- **접근 제어**: 역할 기반 접근 제어 (RBAC)
- **감사 로깅**: 모든 접근 기록 및 추적

### 🔐 **생체정보 보안**
```python
# 얼굴 데이터 보안 처리 예시
class SecureFaceData:
    def __init__(self):
        self.encryption_key = self._generate_key()
    
    def encrypt_embedding(self, embedding):
        """얼굴 임베딩 암호화"""
        return self._encrypt(embedding, self.encryption_key)
    
    def anonymize_data(self, face_data):
        """개인정보 익명화"""
        return self._remove_identifiers(face_data)
```

## 📈 성능 최적화

### ⚡ **실시간 처리 최적화**
```python
# 성능 최적화 예시
class OptimizedProcessor:
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device = "cuda" if self.gpu_available else "cpu"
        
    def process_batch(self, images, batch_size=8):
        """배치 처리로 성능 향상"""
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_results = self._process_on_device(batch)
            results.extend(batch_results)
        return results
```

### 🎯 **성능 목표**
```yaml
얼굴인식:
  - FPS: 30fps 이상
  - 정확도: 99.5% 이상
  - 지연시간: 100ms 이하

공장불량인식:
  - 처리속도: 1000장/분 이상
  - 정확도: 95% 이상
  - 오탐률: 5% 이하

전선불량인식:
  - 원거리 검출: 100m 이상
  - 정확도: 90% 이상
  - 안전성: 99.9% 이상
```

## 🧪 테스트 전략

### 📋 **테스트 레벨**
```python
# 테스트 구조
tests/
├── unit/               # 단위 테스트
│   ├── test_detectors.py
│   ├── test_recognizers.py
│   └── test_classifiers.py
├── integration/        # 통합 테스트
│   ├── test_pipelines.py
│   └── test_apis.py
├── performance/        # 성능 테스트
│   ├── test_speed.py
│   └── test_accuracy.py
└── e2e/               # 종단간 테스트
    └── test_workflows.py
```

### 🔄 **CI/CD 파이프라인**
```yaml
# GitHub Actions 예시
name: Vision System CI/CD
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ --cov=src/
      - name: Quality checks
        run: |
          black --check src/
          flake8 src/
          mypy src/
```

## 📚 추가 리소스

### 📖 **학습 자료**
- [OpenCV 공식 문서](https://docs.opencv.org/)
- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)
- [MediaPipe 가이드](https://mediapipe.dev/)
- [YOLO 공식 문서](https://docs.ultralytics.com/)

### 🤝 **커뮤니티 및 지원**
- **이슈 트래킹**: GitHub Issues 사용
- **토론**: GitHub Discussions 활용
- **문서 기여**: Pull Request 환영
- **질문**: 각 도메인별 전문가 지원

### 📝 **문서 기여 가이드**
```markdown
# 문서 기여 방법
1. 이슈 생성 또는 확인
2. 브랜치 생성: feature/document-update
3. 문서 수정 및 테스트
4. Pull Request 생성
5. 리뷰 및 병합
```

## 🎯 다음 단계

### ✅ **현재 완료된 작업**
- [x] 프로젝트 구조 설계
- [x] 문서화 시스템 구축
- [x] 개발 가이드라인 수립
- [x] 보안 및 컴플라이언스 계획
- [x] 테스트 전략 수립

### 🚀 **즉시 시작 가능한 작업**
- [ ] **얼굴 검출 구현** (Phase 1.1 시작)
- [ ] 개발 환경 설정
- [ ] 첫 번째 프로토타입 개발
- [ ] 성능 벤치마크 수립

### 📅 **향후 계획**
1. **Week 1-2**: 얼굴 검출 기본 구현
2. **Week 3-4**: 얼굴 검출 최적화
3. **Month 2**: 얼굴 인식 구현 시작
4. **Month 3-6**: 얼굴인식 도메인 완성

---

## 📞 연락처 및 지원

**프로젝트 관리자**: Vision System Development Team  
**문서 업데이트**: 2025-06-28  
**버전**: v1.0.0  

**지원 채널**:
- 📧 이메일: [프로젝트 이메일]
- 💬 채팅: [팀 채팅 링크]
- 📋 이슈: [GitHub Issues 링크]

---

*이 문서는 지속적으로 업데이트되며, 각 개발 단계에서 실제 경험을 바탕으로 개선됩니다. 모든 기여와 피드백을 환영합니다!* 🚀 